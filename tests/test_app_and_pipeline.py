"""Verification tests for T2D Genome Risk Prediction (Real Clinical + Hybrid GWAS-PRS)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.t2d_pipeline import (
    FEATURE_ORDER,
    build_panel_metadata,
    calculate_prs_from_genotypes,
    compute_cross_cohort_prs_comparison,
    compute_pima_real_evaluation,
    compute_simulated_test_evaluation,
    get_artifact_status,
    get_curated_real_pima_patients,
    get_curated_synthetic_profiles,
    load_gwas_summary_stats,
    load_model_bundle,
    load_pima_real_dataset,
    load_pima_real_model,
    load_simulated_cohort,
    load_simulated_cohort_summary,
    predict_pima_real_risk,
    predict_t2d_risk,
    simulate_hwe_genotypes,
    validate_clinical_inputs,
    validate_pima_clinical_inputs,
)


def test_required_artifacts_exist():
    status = get_artifact_status()
    for key in [
        "realistic_model",
        "realistic_scaler",
        "baseline_model",
        "baseline_scaler",
        "gwas_cleaned",
        "prs_weights",
        "simulated_dataset",
        "evaluation_results",
        "pima_real_data",
        "pima_real_model",
    ]:
        assert status[key] is True, f"Missing required artifact: {key}"


def test_model_and_scaler_loading():
    for key in ("realistic", "baseline"):
        bundle = load_model_bundle(key)
        assert bundle.feature_names == FEATURE_ORDER
        assert hasattr(bundle.model, "predict_proba")
        assert hasattr(bundle.scaler, "transform")

    pima_model = load_pima_real_model()
    assert hasattr(pima_model, "predict_proba")

    with pytest.raises(ValueError):
        load_model_bundle("nonexistent")


def test_real_pima_clinical_pipeline():
    pima_df = load_pima_real_dataset()
    assert len(pima_df) == 768
    pima_model = load_pima_real_model()

    presets = get_curated_real_pima_patients(pima_df)
    assert len(presets) == 4

    probs = []
    for rec in presets:
        ok, errs, cleaned = validate_pima_clinical_inputs(rec["values"])
        assert ok is True and errs == []
        res = predict_pima_real_risk(cleaned, pima_model, include_shap=True)
        assert 0.0 <= res["probability"] <= 1.0
        assert res["shap"] is not None
        assert len(res["shap"]["shap_df"]) == 8
        probs.append(res["probability"])
    assert probs[-1] > probs[0]

    ev = compute_pima_real_evaluation(pima_df, pima_model)
    assert ev["auc"] > 0.80
    assert ev["cv_auc_mean"] > 0.80


def test_pima_input_validation():
    ok, errs, _ = validate_pima_clinical_inputs(
        {
            "Pregnancies": 1,
            "Glucose": 20.0,
            "BloodPressure": 70.0,
            "SkinThickness": 20.0,
            "Insulin": 80.0,
            "BMI": 5.0,
            "DiabetesPedigreeFunction": 0.4,
            "Age": 12,
        }
    )
    assert ok is False
    assert len(errs) == 3


def test_prs_calculation_matches_simulated_cohort():
    gwas_df = load_gwas_summary_stats()
    sim_df, panel_snps, sim_stats = load_simulated_cohort()
    panel_df = build_panel_metadata(gwas_df, panel_snps)
    _, fast_stats = load_simulated_cohort_summary()
    assert abs(fast_stats["mean"] - sim_stats["mean"]) < 1e-12

    for idx in (0, 1, 42, 999):
        row = sim_df.iloc[idx]
        dosages = {snp: int(row[snp]) for snp in panel_snps}
        res = calculate_prs_from_genotypes(dosages, panel_df, cohort_stats=sim_stats)
        assert abs(res["prs_raw"] - float(row["PRS"])) < 1e-12
        assert res["observed_snps"] == 200
        assert res["imputed_snps"] == 0


def test_prs_dosage_and_missing_validation():
    gwas_df = load_gwas_summary_stats()
    _, panel_snps, _ = load_simulated_cohort()
    panel_df = build_panel_metadata(gwas_df, panel_snps)
    dosages = simulate_hwe_genotypes(panel_df, profile="average", seed=42)

    bad_dosages = dict(dosages)
    bad_dosages[panel_snps[0]] = 3
    with pytest.raises(ValueError):
        calculate_prs_from_genotypes(bad_dosages, panel_df)

    missing_dosages = dict(dosages)
    del missing_dosages[panel_snps[0]]
    with pytest.raises(ValueError):
        calculate_prs_from_genotypes(missing_dosages, panel_df, allow_eaf_imputation=False)

    imputed_res = calculate_prs_from_genotypes(missing_dosages, panel_df, allow_eaf_imputation=True)
    assert imputed_res["imputed_snps"] == 1
    assert imputed_res["observed_snps"] == 199


def test_clinical_input_validation():
    ok, errs, cleaned = validate_clinical_inputs(45, 26.5, 92.0, 1, 0)
    assert ok is True
    assert errs == []
    assert cleaned["age"] == 45

    ok_bad, errs_bad, _ = validate_clinical_inputs(10, 5.0, 200.0, 2, np.nan)
    assert ok_bad is False
    assert len(errs_bad) == 5


def test_prediction_across_synthetic_profiles():
    bundle = load_model_bundle("realistic")
    sim_df, _, _ = load_simulated_cohort()
    profiles = get_curated_synthetic_profiles(sim_df)
    assert len(profiles) >= 3

    probs = {}
    for prof in profiles:
        res = predict_t2d_risk(prof["clinical"], prof["prs_raw"], bundle, include_shap=True)
        assert 0.0 <= res["probability"] <= 1.0
        assert res["shap"] is not None
        assert len(res["shap"]["shap_df"]) == len(FEATURE_ORDER)
        probs[prof["label"]] = res["probability"]

    low_prob = next(v for k, v in probs.items() if "Low Risk" in k)
    high_prob = next(v for k, v in probs.items() if "Very High Combined Risk" in k)
    assert high_prob > low_prob


def test_simulated_and_cross_cohort_evaluations():
    bundle = load_model_bundle("realistic")
    sim_df, _, _ = load_simulated_cohort()
    ev = compute_simulated_test_evaluation(sim_df, bundle)
    assert ev["n_test"] == 1000
    assert ev["auc"] > 0.90

    cc_df = compute_cross_cohort_prs_comparison(load_pima_real_dataset(), bundle)
    assert len(cc_df) == 2
    assert cc_df.iloc[1]["Specificity"] > cc_df.iloc[0]["Specificity"]


def test_streamlit_all_pages_and_tracks_render():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"Page 1 exception: {at.exception}"

    radio = at.sidebar.radio[0]
    for option in radio.options:
        radio.set_value(option).run()
        assert not at.exception, f"Exception on {option}: {at.exception}"

    # Verify both tracks & interactive modes on Page 2 render cleanly
    radio.set_value("2. Patient Risk Assessment").run()
    assert len(at.tabs) == 2
    patient_source_radio = next(r for r in at.radio if r.label == "Patient Source")
    patient_source_radio.set_value("Custom Clinical Input").run()
    assert not at.exception, f"Exception on Real Clinical Custom Input: {at.exception}"

    hybrid_mode_radio = next(r for r in at.radio if r.label == "Genetic Demonstration Mode")
    hybrid_mode_radio.set_value("Simulated Genotype Input").run()
    assert not at.exception, f"Exception on Hybrid Simulated Genotype Input: {at.exception}"
