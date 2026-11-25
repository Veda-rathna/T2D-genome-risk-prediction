"""
Core backend module for the T2D Genome Risk Prediction research demonstration.

Reuses the repository's existing trained models, scalers, GWAS summary statistics,
PRS weights, and simulated cohort artifacts while providing strict input validation,
allele-consistent PRS calculation, SHAP explainability, and safe fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Resolve paths relative to repository root regardless of current working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "processed_data"
DATA_DIR = PROJECT_ROOT / "data"

PIMA_FEATURES: list[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

PIMA_ZERO_MISSING_COLS: list[str] = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

PIMA_FEATURE_LABELS: dict[str, str] = {
    "Pregnancies": "Pregnancies (count)",
    "Glucose": "Plasma Glucose (mg/dL)",
    "BloodPressure": "Diastolic BP (mm Hg)",
    "SkinThickness": "Triceps Skin Fold (mm)",
    "Insulin": "2-Hr Serum Insulin (mu U/mL)",
    "BMI": "BMI (kg/m²)",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
    "Age": "Age (years)",
}

PIMA_BOUNDS: dict[str, tuple[float, float]] = {
    "Pregnancies": (0, 20),
    "Glucose": (40.0, 260.0),
    "BloodPressure": (24.0, 140.0),
    "SkinThickness": (5.0, 100.0),
    "Insulin": (10.0, 900.0),
    "BMI": (14.0, 70.0),
    "DiabetesPedigreeFunction": (0.05, 2.6),
    "Age": (18, 95),
}

FEATURE_ORDER: list[str] = [
    "age",
    "bmi",
    "waist_cm",
    "family_history",
    "physical_activity",
    "PRS",
]

FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "age": "Age (years)",
    "bmi": "BMI (kg/m²)",
    "waist_cm": "Waist Circumference (cm)",
    "family_history": "Family History of T2D",
    "physical_activity": "Regular Physical Activity",
    "PRS": "Polygenic Risk Score (Raw)",
}

INPUT_BOUNDS: dict[str, tuple[float, float]] = {
    "age": (18, 95),
    "bmi": (14.0, 60.0),
    "waist_cm": (50.0, 160.0),
    "family_history": (0, 1),
    "physical_activity": (0, 1),
}

# Demonstration risk tiers matching T2D_Risk_Prediction_Demo.ipynb
# These are NOT clinically validated diagnostic cutoffs.
RISK_TIERS: list[tuple[float, str, str]] = [
    (0.75, "Very High Risk", "#b91c1c"),
    (0.50, "High Risk", "#d97706"),
    (0.25, "Moderate Risk", "#ca8a04"),
    (0.00, "Low Risk", "#15803d"),
]


@dataclass
class ModelBundle:
    """Loaded model, scaler, and schema metadata."""

    model: Any
    scaler: Any
    model_key: str
    model_label: str
    model_path: Path
    scaler_path: Path
    is_calibrated: bool
    feature_names: list[str]


def get_artifact_status() -> dict[str, bool]:
    """Check presence of all expected repository artifacts."""
    paths = {
        "realistic_model": MODELS_DIR / "realistic_model.pkl",
        "realistic_scaler": MODELS_DIR / "realistic_scaler.pkl",
        "baseline_model": MODELS_DIR / "best_model_ml.pkl",
        "baseline_scaler": MODELS_DIR / "scaler.pkl",
        "gwas_cleaned": PROCESSED_DIR / "cleaned_gwas_summary_stats.csv",
        "prs_weights": PROCESSED_DIR / "prs_weights.txt",
        "simulated_dataset": PROCESSED_DIR / "simulated_t2d_dataset.csv",
        "evaluation_results": PROCESSED_DIR / "model_evaluation_results.csv",
        "evaluation_report": PROCESSED_DIR / "evaluation_report.txt",
        "preprocessing_log": PROCESSED_DIR / "preprocessing_log.txt",
        "pima_metrics": PROCESSED_DIR / "pima_metrics.csv",
        "pima_predictions": PROCESSED_DIR / "pima_predictions.csv",
        "pima_real_data": DATA_DIR / "pima_real_clinical.csv",
        "pima_real_model": MODELS_DIR / "pima_real_clinical_model.pkl",
    }
    return {name: p.exists() for name, p in paths.items()}


def _rebuild_hybrid_bundle(model_key: str) -> tuple[Any, Any]:
    """Rebuild hybrid model & scaler locally if pickled artifacts fail across sklearn versions."""
    sim_df, _ = load_simulated_cohort_summary()
    X = sim_df[FEATURE_ORDER].copy()
    y = sim_df["T2D_status"].astype(int).copy()

    if model_key == "realistic":
        rng = np.random.default_rng(42)
        X_noisy = X.copy()
        for col in FEATURE_ORDER:
            if col != "family_history":
                X_noisy[col] = X[col] + rng.normal(0, 0.1 * X[col].std(), size=len(X))
        X_noisy["age"] = X_noisy["age"].clip(18, 90)
        X_noisy["bmi"] = X_noisy["bmi"].clip(15, 50)
        X_noisy["waist_cm"] = X_noisy["waist_cm"].clip(60, 140)
        X_noisy["physical_activity"] = X_noisy["physical_activity"].clip(0, 7)

        case_idx = np.where(y == 1)[0]
        ctrl_idx = np.where(y == 0)[0]
        n_cases = int(len(ctrl_idx) * 0.12 / 0.88)
        sel_cases = rng.choice(case_idx, size=n_cases, replace=False)
        sel = np.concatenate([ctrl_idx, sel_cases])
        rng.shuffle(sel)
        X_bal, y_bal = X_noisy.iloc[sel], y.iloc[sel]

        X_train, _, y_train, _ = train_test_split(
            X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
        )
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        base = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            scale_pos_weight=7,
            random_state=42,
            eval_metric="logloss",
        )
        model = CalibratedClassifierCV(base, method="sigmoid", cv=5)
        model.fit(X_train_sc, y_train)
        return model, scaler

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    model = XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42)
    model.fit(X_train_sc, y_train)
    return model, scaler


def load_model_bundle(model_key: str = "realistic") -> ModelBundle:
    """Load the requested trained model and scaler, rebuilding cleanly if pickle fails."""
    if model_key == "realistic":
        model_path = MODELS_DIR / "realistic_model.pkl"
        scaler_path = MODELS_DIR / "realistic_scaler.pkl"
        label = "Calibrated Regularized XGBoost (realistic_model.pkl)"
        is_cal = True
    elif model_key == "baseline":
        model_path = MODELS_DIR / "best_model_ml.pkl"
        scaler_path = MODELS_DIR / "scaler.pkl"
        label = "Uncalibrated Baseline XGBoost (best_model_ml.pkl)"
        is_cal = False
    else:
        raise ValueError(f"Unknown model_key '{model_key}'. Expected 'realistic' or 'baseline'.")

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler artifact not found: {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Smoke-test unpickled estimators against installed scikit-learn version
    smoke_df = pd.DataFrame(
        [[48, 27.5, 92.0, 0, 1, -8.66]], columns=FEATURE_ORDER
    )
    try:
        _ = model.predict_proba(scaler.transform(smoke_df))
    except Exception:
        model, scaler = _rebuild_hybrid_bundle(model_key)

    scaler_features = list(getattr(scaler, "feature_names_in_", FEATURE_ORDER))
    if scaler_features != FEATURE_ORDER:
        raise ValueError(
            f"Scaler feature schema mismatch. Expected {FEATURE_ORDER}, got {scaler_features}"
        )

    return ModelBundle(
        model=model,
        scaler=scaler,
        model_key=model_key,
        model_label=label,
        model_path=model_path,
        scaler_path=scaler_path,
        is_calibrated=is_cal,
        feature_names=scaler_features,
    )


def load_gwas_summary_stats() -> pd.DataFrame:
    """Load cleaned DIAMANTE South Asian T2D GWAS summary statistics (1,113 variants)."""
    path = PROCESSED_DIR / "cleaned_gwas_summary_stats.csv"
    if not path.exists():
        raise FileNotFoundError(f"GWAS summary statistics not found: {path}")
    df = pd.read_csv(path)
    df["neg_log10_pval"] = -np.log10(np.clip(df["pval"].astype(float), 1e-300, 1.0))
    return df


def load_prs_weights() -> pd.DataFrame:
    """Load PRS weights generated by gwas_preprocessing.py."""
    path = PROCESSED_DIR / "prs_weights.txt"
    if not path.exists():
        raise FileNotFoundError(f"PRS weights file not found: {path}")
    return pd.read_csv(path, sep="\t")


def load_simulated_cohort_summary() -> tuple[pd.DataFrame, dict[str, float]]:
    """Fast load of only the 7 summary columns from `simulated_t2d_dataset.csv`."""
    path = PROCESSED_DIR / "simulated_t2d_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Simulated cohort not found: {path}")
    cols = FEATURE_ORDER + ["T2D_status"]
    df = pd.read_csv(path, usecols=cols)
    all_cols = pd.read_csv(path, nrows=0).columns.tolist()
    n_snps = sum(1 for c in all_cols if c.startswith("rs"))
    prs_series = df["PRS"].astype(float)
    stats = {
        "mean": float(prs_series.mean()),
        "std": float(prs_series.std(ddof=0)),
        "min": float(prs_series.min()),
        "max": float(prs_series.max()),
        "q25": float(prs_series.quantile(0.25)),
        "q50": float(prs_series.quantile(0.50)),
        "q75": float(prs_series.quantile(0.75)),
        "n_individuals": int(len(df)),
        "n_snps": int(n_snps),
        "prevalence": float(df["T2D_status"].mean()),
    }
    return df, stats


def load_simulated_cohort() -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    """Load the 5,000-individual simulated cohort and compute reference PRS stats."""
    path = PROCESSED_DIR / "simulated_t2d_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Simulated cohort not found: {path}")
    df = pd.read_csv(path)
    panel_snps = [c for c in df.columns if c.startswith("rs")]
    prs_series = df["PRS"].astype(float)
    stats = {
        "mean": float(prs_series.mean()),
        "std": float(prs_series.std(ddof=0)),
        "min": float(prs_series.min()),
        "max": float(prs_series.max()),
        "q25": float(prs_series.quantile(0.25)),
        "q50": float(prs_series.quantile(0.50)),
        "q75": float(prs_series.quantile(0.75)),
        "n_individuals": int(len(df)),
        "n_snps": int(len(panel_snps)),
        "prevalence": float(df["T2D_status"].mean()),
    }
    return df, panel_snps, stats


def build_panel_metadata(gwas_df: pd.DataFrame, panel_snps: list[str]) -> pd.DataFrame:
    """Align GWAS metadata (alleles, EAF, beta, pval) to the exact 200-SNP panel order."""
    indexed = gwas_df.drop_duplicates(subset=["rsID"]).set_index("rsID")
    missing = [snp for snp in panel_snps if snp not in indexed.index]
    if missing:
        raise ValueError(f"Panel SNPs missing from GWAS summary stats: {missing[:5]}")
    panel_df = indexed.loc[panel_snps].reset_index()
    panel_df["abs_beta"] = panel_df["beta"].abs()
    return panel_df


def validate_clinical_inputs(
    age: Any,
    bmi: Any,
    waist_cm: Any,
    family_history: Any,
    physical_activity: Any,
) -> tuple[bool, list[str], dict[str, float | int]]:
    """Validate clinical inputs against physiological bounds and schema types."""
    errors: list[str] = []
    raw_values = {
        "age": age,
        "bmi": bmi,
        "waist_cm": waist_cm,
        "family_history": family_history,
        "physical_activity": physical_activity,
    }
    cleaned: dict[str, float | int] = {}

    for key, val in raw_values.items():
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            errors.append(f"{FEATURE_DISPLAY_NAMES[key]} is missing or non-finite.")
            continue
        try:
            num = float(val)
        except (TypeError, ValueError):
            errors.append(f"{FEATURE_DISPLAY_NAMES[key]} must be numeric.")
            continue

        lo, hi = INPUT_BOUNDS[key]
        if num < lo or num > hi:
            errors.append(
                f"{FEATURE_DISPLAY_NAMES[key]} ({num:g}) is outside valid range [{lo}, {hi}]."
            )
            continue

        if key in ("family_history", "physical_activity"):
            if int(num) not in (0, 1) or num != int(num):
                errors.append(f"{FEATURE_DISPLAY_NAMES[key]} must be 0 or 1.")
                continue
            cleaned[key] = int(num)
        elif key == "age":
            cleaned[key] = int(round(num))
        else:
            cleaned[key] = float(num)

    return len(errors) == 0, errors, cleaned


def calculate_prs_from_genotypes(
    dosage_map: dict[str, int | float | None],
    panel_df: pd.DataFrame,
    cohort_stats: dict[str, float] | None = None,
    allow_eaf_imputation: bool = False,
) -> dict[str, Any]:
    """
    Calculate raw PRS = sum(dosage_i * beta_i) using exact GWAS effect alleles and betas.

    Dosage represents the count (0, 1, or 2) of the GWAS `effect_allele`.
    Missing genotypes are rejected unless `allow_eaf_imputation=True`, in which case
    2 * EAF is used explicitly and reported in `imputed_count`.
    """
    contributions: list[dict[str, Any]] = []
    total_prs = 0.0
    imputed_count = 0
    observed_count = 0

    for row in panel_df.itertuples(index=False):
        rsid = row.rsID
        beta = float(row.beta)
        eaf = float(row.eaf)
        raw_dosage = dosage_map.get(rsid, None)

        if raw_dosage is None or (isinstance(raw_dosage, float) and np.isnan(raw_dosage)):
            if not allow_eaf_imputation:
                raise ValueError(f"Missing genotype dosage for {rsid}.")
            dosage = 2.0 * eaf
            is_imputed = True
            imputed_count += 1
        else:
            dosage_num = float(raw_dosage)
            if dosage_num not in (0.0, 1.0, 2.0):
                raise ValueError(
                    f"Invalid genotype dosage {raw_dosage} for {rsid}; expected 0, 1, or 2."
                )
            dosage = int(dosage_num)
            is_imputed = False
            observed_count += 1

        contrib = dosage * beta
        total_prs += contrib
        contributions.append(
            {
                "rsID": rsid,
                "chr": int(row.chr),
                "pos": int(row.pos),
                "effect_allele": row.effect_allele,
                "other_allele": row.other_allele,
                "eaf": eaf,
                "beta": beta,
                "pval": float(row.pval),
                "dosage": dosage,
                "contribution": contrib,
                "imputed": is_imputed,
            }
        )

    contrib_df = pd.DataFrame(contributions)
    z_score = None
    percentile = None
    if cohort_stats is not None and cohort_stats.get("std", 0) > 0:
        z_score = (total_prs - cohort_stats["mean"]) / cohort_stats["std"]

    return {
        "prs_raw": float(total_prs),
        "prs_z": float(z_score) if z_score is not None else None,
        "percentile": percentile,
        "observed_snps": observed_count,
        "imputed_snps": imputed_count,
        "contributions": contrib_df,
    }


def compute_prs_percentile(prs_raw: float, prs_series: pd.Series) -> float:
    """Compute percentile rank (0-100) of raw PRS in the simulated reference cohort."""
    return float((prs_series <= prs_raw).mean() * 100.0)


def simulate_hwe_genotypes(
    panel_df: pd.DataFrame,
    profile: str = "average",
    seed: int = 42,
) -> dict[str, int]:
    """
    Simulate 200 SNP dosages (0, 1, 2 copies of effect_allele) under Hardy-Weinberg Equilibrium.

    Profiles:
      - 'average': Unbiased HWE using DIAMANTE SAS EAF.
      - 'low_genetic_risk': Modest EAF shift away from risk-increasing alleles (demonstration only).
      - 'high_genetic_risk': Modest EAF shift toward risk-increasing alleles (demonstration only).
    """
    rng = np.random.default_rng(seed)
    dosages: dict[str, int] = {}

    shift_mag = 0.0
    if profile == "low_genetic_risk":
        shift_mag = -0.10
    elif profile == "high_genetic_risk":
        shift_mag = 0.12
    elif profile != "average":
        raise ValueError(f"Unknown simulation profile: {profile}")

    for row in panel_df.itertuples(index=False):
        eaf = float(row.eaf)
        if shift_mag != 0.0:
            direction = 1.0 if float(row.beta) >= 0 else -1.0
            eaf = float(np.clip(eaf + shift_mag * direction, 0.02, 0.98))
        p0 = (1.0 - eaf) ** 2
        p1 = 2.0 * eaf * (1.0 - eaf)
        p2 = eaf**2
        probs = np.array([p0, p1, p2], dtype=float)
        probs /= probs.sum()
        dosages[row.rsID] = int(rng.choice([0, 1, 2], p=probs))

    return dosages


def classify_risk_tier(probability: float) -> tuple[str, str]:
    """Map predicted probability to demonstration risk tier."""
    for cutoff, label, color in RISK_TIERS:
        if probability >= cutoff:
            return label, color
    return "Low Risk", "#15803d"


def compute_shap_explanation(
    bundle: ModelBundle,
    X_scaled: np.ndarray,
    feature_values: dict[str, float | int],
) -> dict[str, Any]:
    """
    Compute TreeSHAP contributions in log-odds space.

    For CalibratedClassifierCV (`realistic_model.pkl`), averages SHAP values across the
    5 fold XGBoost base estimators so the explanation reflects the calibrated ensemble's
    underlying tree model.
    """
    if hasattr(bundle.model, "calibrated_classifiers_"):
        shap_arrays = []
        base_values = []
        for cc in bundle.model.calibrated_classifiers_:
            est = getattr(cc, "estimator", None)
            if est is None:
                continue
            explainer = shap.TreeExplainer(est)
            sv = np.asarray(explainer.shap_values(X_scaled))
            if sv.ndim == 2:
                shap_arrays.append(sv[0])
            else:
                shap_arrays.append(sv)
            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                ev = float(np.asarray(ev).ravel()[0])
            base_values.append(float(ev))
        if not shap_arrays:
            raise RuntimeError("No base estimators available inside CalibratedClassifierCV.")
        mean_shap = np.mean(shap_arrays, axis=0)
        base_val = float(np.mean(base_values))
    else:
        explainer = shap.TreeExplainer(bundle.model)
        sv = np.asarray(explainer.shap_values(X_scaled))
        mean_shap = sv[0] if sv.ndim == 2 else sv
        ev = explainer.expected_value
        base_val = float(np.asarray(ev).ravel()[0]) if isinstance(ev, (list, np.ndarray)) else float(ev)

    rows = []
    for idx, feat in enumerate(FEATURE_ORDER):
        val = float(mean_shap[idx])
        rows.append(
            {
                "feature": feat,
                "display_name": FEATURE_DISPLAY_NAMES[feat],
                "input_value": feature_values[feat],
                "shap_value": val,
                "abs_shap": abs(val),
                "direction": "Increases Risk" if val >= 0 else "Decreases Risk",
            }
        )
    shap_df = pd.DataFrame(rows).sort_values("abs_shap", ascending=False).reset_index(drop=True)
    return {
        "base_log_odds": base_val,
        "shap_sum": float(np.sum(mean_shap)),
        "shap_df": shap_df,
    }


def predict_t2d_risk(
    clinical_inputs: dict[str, float | int],
    prs_raw: float,
    bundle: ModelBundle,
    include_shap: bool = True,
) -> dict[str, Any]:
    """Run validated scaling, prediction, demonstration tiering, and optional SHAP explanation."""
    if prs_raw is None or not np.isfinite(float(prs_raw)):
        raise ValueError("PRS must be a finite numeric value.")

    row = {
        "age": clinical_inputs["age"],
        "bmi": clinical_inputs["bmi"],
        "waist_cm": clinical_inputs["waist_cm"],
        "family_history": clinical_inputs["family_history"],
        "physical_activity": clinical_inputs["physical_activity"],
        "PRS": float(prs_raw),
    }
    input_df = pd.DataFrame([row], columns=FEATURE_ORDER)
    X_scaled = bundle.scaler.transform(input_df)

    prob = float(bundle.model.predict_proba(X_scaled)[0, 1])
    pred_class = int(bundle.model.predict(X_scaled)[0])
    tier_label, tier_color = classify_risk_tier(prob)

    shap_result = None
    shap_error = None
    if include_shap:
        try:
            shap_result = compute_shap_explanation(bundle, X_scaled, row)
        except Exception as exc:  # noqa: BLE001
            shap_error = str(exc)

    return {
        "probability": prob,
        "predicted_class": pred_class,
        "risk_tier": tier_label,
        "tier_color": tier_color,
        "feature_row": row,
        "scaled_values": X_scaled[0].tolist(),
        "shap": shap_result,
        "shap_error": shap_error,
    }


def get_curated_synthetic_profiles(sim_df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Select diverse, reproducible synthetic patient records directly from `simulated_t2d_dataset.csv`
    so every preset has genuine simulated 200-SNP genotypes and exact raw PRS.
    """
    df = sim_df.copy()
    df["row_id"] = df.index
    prs_q = df["PRS"]

    candidates = [
        (
            "Low Risk — Young & Active",
            "Young adult, normal BMI, active lifestyle, low polygenic risk (from simulated cohort).",
            df[(df["age"] <= 35) & (df["bmi"] < 23) & (df["physical_activity"] == 1) & (df["family_history"] == 0) & (prs_q < prs_q.quantile(0.20))],
        ),
        (
            "Moderate Risk — Middle-Aged Profile",
            "Middle-aged individual with slightly elevated BMI and borderline PRS (from simulated cohort).",
            df.iloc[[469]] if len(df) > 469 else df[(df["age"].between(42, 55)) & (df["bmi"].between(24.5, 28.5))],
        ),
        (
            "High Genetic Risk — Younger Adult",
            "Younger individual with elevated polygenic risk score driving risk above 50% (from simulated cohort).",
            df.iloc[[144]] if len(df) > 144 else df[(df["age"] <= 48) & (prs_q >= prs_q.quantile(0.85))],
        ),
        (
            "Very High Combined Risk — Clinical + Genetic",
            "Older sedentary individual with elevated BMI, family history, and high PRS (from simulated cohort).",
            df[(df["age"] >= 55) & (df["bmi"] >= 28) & (df["family_history"] == 1) & (df["physical_activity"] == 0) & (prs_q >= prs_q.quantile(0.85))],
        ),
        (
            "Older Healthy — Protective Lifestyle & Genetics",
            "Older adult maintaining healthy BMI, activity, and low genetic risk (from simulated cohort).",
            df[(df["age"] >= 60) & (df["bmi"] < 24) & (df["physical_activity"] == 1) & (prs_q <= prs_q.quantile(0.25))],
        ),
    ]

    profiles: list[dict[str, Any]] = []
    for title, desc, subset in candidates:
        row = subset.iloc[0] if len(subset) > 0 else df.iloc[len(profiles)]
        profiles.append(
            {
                "label": title,
                "description": desc,
                "cohort_index": int(row["row_id"]),
                "clinical": {
                    "age": int(round(row["age"])),
                    "bmi": float(round(row["bmi"], 1)),
                    "waist_cm": float(round(row["waist_cm"], 1)),
                    "family_history": int(row["family_history"]),
                    "physical_activity": int(row["physical_activity"]),
                },
                "prs_raw": float(row["PRS"]),
                "simulated_t2d_label": int(row["T2D_status"]),
                "dosages": {col: int(row[col]) for col in df.columns if col.startswith("rs")},
            }
        )
    return profiles


def compute_simulated_test_evaluation(
    sim_df: pd.DataFrame, bundle: ModelBundle
) -> dict[str, Any]:
    """
    Evaluate the loaded model on the 20% stratified holdout split of `simulated_t2d_dataset.csv`
    (matching the split seed=42 used in simulate_and_train.py / evaluate_and_improve.py).
    """
    X = sim_df[FEATURE_ORDER]
    y = sim_df["T2D_status"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_test_scaled = bundle.scaler.transform(X_test)
    y_proba = bundle.model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="uniform")

    # Feature importance from base estimator(s)
    if hasattr(bundle.model, "calibrated_classifiers_"):
        imps = [
            cc.estimator.feature_importances_
            for cc in bundle.model.calibrated_classifiers_
            if hasattr(cc.estimator, "feature_importances_")
        ]
        feat_imp = np.mean(imps, axis=0)
    elif hasattr(bundle.model, "feature_importances_"):
        feat_imp = bundle.model.feature_importances_
    else:
        feat_imp = np.zeros(len(FEATURE_ORDER))

    imp_df = pd.DataFrame(
        {
            "Feature": [FEATURE_DISPLAY_NAMES[f] for f in FEATURE_ORDER],
            "RawFeature": FEATURE_ORDER,
            "Importance": feat_imp,
        }
    ).sort_values("Importance", ascending=False)

    # Global SHAP on 300-sample subset of test set for fast UI rendering
    sample_n = min(300, len(X_test))
    X_sample_scaled = X_test_scaled[:sample_n]
    if hasattr(bundle.model, "calibrated_classifiers_"):
        svs = [
            np.asarray(shap.TreeExplainer(cc.estimator).shap_values(X_sample_scaled))
            for cc in bundle.model.calibrated_classifiers_
        ]
        shap_matrix = np.mean(svs, axis=0)
    else:
        shap_matrix = np.asarray(shap.TreeExplainer(bundle.model).shap_values(X_sample_scaled))

    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    global_shap_df = pd.DataFrame(
        {
            "Feature": [FEATURE_DISPLAY_NAMES[f] for f in FEATURE_ORDER],
            "Mean |SHAP|": mean_abs_shap,
        }
    ).sort_values("Mean |SHAP|", ascending=False)

    return {
        "n_test": int(len(y_test)),
        "auc": float(roc_auc_score(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "fpr": fpr,
        "tpr": tpr,
        "cm": cm,
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "y_test": y_test.values,
        "y_proba": y_proba,
        "importance_df": imp_df,
        "global_shap_df": global_shap_df,
        "shap_matrix": shap_matrix,
        "X_shap_sample": X_test.iloc[:sample_n],
    }


def load_saved_evaluation_artifacts() -> dict[str, Any]:
    """Load pre-computed ablation CSV, text reports, and PIMA external clinical outputs."""
    data: dict[str, Any] = {}
    eval_csv = PROCESSED_DIR / "model_evaluation_results.csv"
    if eval_csv.exists():
        data["ablation_df"] = pd.read_csv(eval_csv)

    pima_metrics = PROCESSED_DIR / "pima_metrics.csv"
    if pima_metrics.exists():
        data["pima_metrics_df"] = pd.read_csv(pima_metrics)

    pima_preds = PROCESSED_DIR / "pima_predictions.csv"
    if pima_preds.exists():
        data["pima_predictions_df"] = pd.read_csv(pima_preds)

    prep_log = PROCESSED_DIR / "preprocessing_log.txt"
    if prep_log.exists():
        data["preprocessing_log"] = prep_log.read_text(encoding="utf-8")

    eval_rep = PROCESSED_DIR / "evaluation_report.txt"
    if eval_rep.exists():
        data["evaluation_report"] = eval_rep.read_text(encoding="utf-8")

    return data


# ---------------------------------------------------------------------------
# Real Clinical Dataset & Model (UCI PIMA Indians Diabetes, n = 768)
# ---------------------------------------------------------------------------
def load_pima_real_dataset() -> pd.DataFrame:
    """Load the real 768-patient UCI PIMA Indians Diabetes clinical dataset."""
    path = DATA_DIR / "pima_real_clinical.csv"
    if not path.exists():
        raise FileNotFoundError(f"Real PIMA clinical dataset not found: {path}")
    return pd.read_csv(path)


def _fit_pima_real_pipeline() -> Pipeline:
    """Fit the real-data PIMA clinical XGBoost pipeline using the active sklearn version."""
    df = load_pima_real_dataset()
    X = df[PIMA_FEATURES].copy()
    for col in PIMA_ZERO_MISSING_COLS:
        X[col] = X[col].replace(0, np.nan)
    y = df["Outcome"].astype(int)

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                XGBClassifier(
                    n_estimators=120,
                    max_depth=3,
                    learning_rate=0.05,
                    min_child_weight=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=1.0,
                    reg_lambda=2.0,
                    random_state=42,
                    eval_metric="logloss",
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def load_pima_real_model() -> Any:
    """
    Load the scikit-learn Pipeline trained on real PIMA clinical data.

    Runs a 1-row smoke test after unpickling; if an incompatible scikit-learn version
    (e.g., SimpleImputer missing `_fill_dtype` in sklearn 1.9+) raises an error,
    transparently re-fits the pipeline from `data/pima_real_clinical.csv` (~0.2s).
    """
    path = MODELS_DIR / "pima_real_clinical_model.pkl"
    smoke_df = pd.DataFrame(
        [[1, 118.0, 72.0, 28.0, 110.0, 28.5, 0.42, 42]], columns=PIMA_FEATURES
    )
    if path.exists():
        try:
            pipeline = joblib.load(path)
            _ = pipeline.predict_proba(smoke_df)
            return pipeline
        except Exception:
            pass
    return _fit_pima_real_pipeline()


def validate_pima_clinical_inputs(values: dict[str, Any]) -> tuple[bool, list[str], dict[str, float]]:
    """Validate the 8 real PIMA clinical parameters against physiological bounds."""
    errors: list[str] = []
    cleaned: dict[str, float] = {}
    for feat in PIMA_FEATURES:
        val = values.get(feat)
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            errors.append(f"{PIMA_FEATURE_LABELS[feat]} is missing or non-finite.")
            continue
        try:
            num = float(val)
        except (TypeError, ValueError):
            errors.append(f"{PIMA_FEATURE_LABELS[feat]} must be numeric.")
            continue
        lo, hi = PIMA_BOUNDS[feat]
        if num < lo or num > hi:
            errors.append(f"{PIMA_FEATURE_LABELS[feat]} ({num:g}) is outside [{lo}, {hi}].")
            continue
        cleaned[feat] = int(round(num)) if feat in ("Pregnancies", "Age") else float(num)
    return len(errors) == 0, errors, cleaned


def predict_pima_real_risk(
    clinical_values: dict[str, float], pipeline: Any, include_shap: bool = True
) -> dict[str, Any]:
    """Predict T2D risk probability using the real-data PIMA clinical XGBoost pipeline."""
    row = {k: clinical_values[k] for k in PIMA_FEATURES}
    X_df = pd.DataFrame([row], columns=PIMA_FEATURES)
    for col in PIMA_ZERO_MISSING_COLS:
        X_df[col] = X_df[col].replace(0, np.nan)

    prob = float(pipeline.predict_proba(X_df)[0, 1])
    pred_class = int(prob >= 0.5)
    tier_label, tier_color = classify_risk_tier(prob)

    shap_result = None
    shap_error = None
    if include_shap:
        try:
            X_imp = pipeline.named_steps["imputer"].transform(X_df)
            X_sc = pipeline.named_steps["scaler"].transform(X_imp)
            clf = pipeline.named_steps["clf"]
            explainer = shap.TreeExplainer(clf)
            sv = np.asarray(explainer.shap_values(X_sc))
            sv_row = sv[0] if sv.ndim == 2 else sv
            ev = explainer.expected_value
            base_val = float(np.asarray(ev).ravel()[0]) if isinstance(ev, (list, np.ndarray)) else float(ev)
            rows = [
                {
                    "feature": f,
                    "display_name": PIMA_FEATURE_LABELS[f],
                    "input_value": row[f],
                    "shap_value": float(sv_row[i]),
                    "abs_shap": abs(float(sv_row[i])),
                    "direction": "Increases Risk" if sv_row[i] >= 0 else "Decreases Risk",
                }
                for i, f in enumerate(PIMA_FEATURES)
            ]
            shap_result = {
                "base_log_odds": base_val,
                "shap_sum": float(np.sum(sv_row)),
                "shap_df": pd.DataFrame(rows).sort_values("abs_shap", ascending=False).reset_index(drop=True),
            }
        except Exception as exc:  # noqa: BLE001
            shap_error = str(exc)

    return {
        "probability": prob,
        "predicted_class": pred_class,
        "risk_tier": tier_label,
        "tier_color": tier_color,
        "feature_row": row,
        "shap": shap_result,
        "shap_error": shap_error,
    }


def get_curated_real_pima_patients(pima_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Select 4 diverse real patient records from the 768-patient PIMA dataset."""
    clean = pima_df[(pima_df["BMI"] > 0) & (pima_df["Glucose"] > 0) & (pima_df["BloodPressure"] > 0)].copy()
    clean["row_id"] = clean.index

    specs = [
        (
            "Real Patient #1 — Healthy Young Adult (Outcome: No T2D)",
            clean[(clean["Outcome"] == 0) & (clean["Age"] <= 28) & (clean["Glucose"] < 100) & (clean["BMI"] < 26)],
        ),
        (
            "Real Patient #2 — Borderline Metabolic Profile (Outcome: No T2D)",
            clean[(clean["Outcome"] == 0) & (clean["Age"].between(32, 48)) & (clean["Glucose"].between(110, 138)) & (clean["BMI"].between(27, 34))],
        ),
        (
            "Real Patient #3 — Elevated Glucose & Pedigree (Outcome: T2D)",
            clean[(clean["Outcome"] == 1) & (clean["Glucose"].between(140, 175)) & (clean["BMI"].between(28, 38))],
        ),
        (
            "Real Patient #4 — High-Risk Hyperglycemic Case (Outcome: T2D)",
            clean[(clean["Outcome"] == 1) & (clean["Glucose"] >= 175) & (clean["Age"] >= 38)],
        ),
    ]

    out: list[dict[str, Any]] = []
    for title, subset in specs:
        row = subset.iloc[0] if len(subset) > 0 else clean.iloc[len(out)]
        values = {
            feat: int(row[feat]) if feat in ("Pregnancies", "Age") else float(row[feat])
            for feat in PIMA_FEATURES
        }
        # Fill 0 insulin/skinfold with clean column median for UI display clarity
        for zc in ("SkinThickness", "Insulin"):
            if values[zc] == 0:
                values[zc] = float(round(clean[clean[zc] > 0][zc].median(), 1))
        out.append(
            {
                "label": title,
                "patient_index": int(row["row_id"]),
                "values": values,
                "actual_outcome": int(row["Outcome"]),
            }
        )
    return out


def compute_pima_real_evaluation(pima_df: pd.DataFrame, pipeline: Any) -> dict[str, Any]:
    """Compute held-out test split (20%) & 5-fold CV metrics on real PIMA data."""
    X = pima_df[PIMA_FEATURES].copy()
    for col in PIMA_ZERO_MISSING_COLS:
        X[col] = X[col].replace(0, np.nan)
    y = pima_df["Outcome"].astype(int)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=8, strategy="uniform")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")

    X_imp = pipeline.named_steps["imputer"].transform(X_test)
    X_sc = pipeline.named_steps["scaler"].transform(X_imp)
    clf = pipeline.named_steps["clf"]
    shap_matrix = np.asarray(shap.TreeExplainer(clf).shap_values(X_sc))
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    shap_df = pd.DataFrame(
        {
            "Feature": [PIMA_FEATURE_LABELS[f] for f in PIMA_FEATURES],
            "Mean |SHAP|": mean_abs_shap,
        }
    ).sort_values("Mean |SHAP|", ascending=False)

    return {
        "n_total": int(len(pima_df)),
        "n_test": int(len(y_test)),
        "prevalence": float(y.mean()),
        "auc": float(roc_auc_score(y_test, y_proba)),
        "cv_auc_mean": float(cv_auc.mean()),
        "cv_auc_std": float(cv_auc.std()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "fpr": fpr,
        "tpr": tpr,
        "cm": cm,
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "global_shap_df": shap_df,
    }


def compute_cross_cohort_prs_comparison(pima_df: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    """Compare hybrid South Asian model behavior on real PIMA clinical data with PRS=0 vs neutral PRS."""
    df = pima_df[pima_df["BMI"] > 0].copy()
    y = df["Outcome"].astype(int).values
    sim_prs_mean = -8.6588

    rows = []
    for label, prs_val, waist_series in [
        ("Original Script Fallback (PRS = 0.0, waist = 3.2*BMI + 50)", 0.0, df["BMI"] * 3.2 + 50),
        ("Cohort-Neutral PRS Imputation (PRS = -8.66, waist = 2.1*BMI + 28)", sim_prs_mean, df["BMI"] * 2.1 + 28),
    ]:
        X = pd.DataFrame(
            {
                "age": df["Age"].values,
                "bmi": df["BMI"].values,
                "waist_cm": waist_series.values,
                "family_history": (df["DiabetesPedigreeFunction"] > 0.5).astype(int).values,
                "physical_activity": 0,
                "PRS": prs_val,
            },
            columns=FEATURE_ORDER,
        )
        probs = bundle.model.predict_proba(bundle.scaler.transform(X))[:, 1]
        preds = (probs >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
        rows.append(
            {
                "Evaluation Strategy": label,
                "AUC-ROC": float(roc_auc_score(y, probs)),
                "Accuracy": float(accuracy_score(y, preds)),
                "Sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                "Specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                "Mean Pred Prob": float(probs.mean()),
            }
        )
    return pd.DataFrame(rows)
