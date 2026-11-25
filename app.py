"""
T2D Genome Risk Prediction — DBT Research Demonstration Application.

Integrates:
1. Real clinical risk modeling (UCI PIMA Indians Diabetes Dataset, n=768, 5-Fold CV AUC 0.839).
2. Real DIAMANTE South Asian GWAS summary statistics (1,113 QC SNPs, 200-SNP PRS panel).
3. Hybrid GWAS-PRS + clinical proof-of-concept prototype (5,000 simulated cohort).
"""

from __future__ import annotations

import html
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import src.t2d_pipeline as _t2d_pipeline

# Ensure submodule updates are picked up cleanly if Streamlit hot-reloads app.py
if not hasattr(_t2d_pipeline, "PIMA_FEATURE_LABELS") or not hasattr(
    _t2d_pipeline, "load_simulated_cohort_summary"
):
    _t2d_pipeline = importlib.reload(_t2d_pipeline)

from src.t2d_pipeline import (
    PIMA_FEATURES,
    PROCESSED_DIR,
    build_panel_metadata,
    calculate_prs_from_genotypes,
    compute_cross_cohort_prs_comparison,
    compute_pima_real_evaluation,
    compute_prs_percentile,
    compute_simulated_test_evaluation,
    get_artifact_status,
    get_curated_real_pima_patients,
    get_curated_synthetic_profiles,
    load_gwas_summary_stats,
    load_model_bundle,
    load_pima_real_dataset,
    load_pima_real_model,
    load_saved_evaluation_artifacts,
    load_simulated_cohort,
    load_simulated_cohort_summary,
    predict_pima_real_risk,
    predict_t2d_risk,
    simulate_hwe_genotypes,
    validate_clinical_inputs,
    validate_pima_clinical_inputs,
)

# ---------------------------------------------------------------------------
# Page Configuration & Clean Scientific Design System
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="T2D Genome Risk Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOTLY_CONFIG = {"displayModeBar": False}

PALETTE = {
    "navy": "#0f2942",
    "blue": "#1d4ed8",
    "teal": "#0f766e",
    "crimson": "#be123c",
    "amber": "#b45309",
    "slate": "#475569",
    "muted": "#94a3b8",
    "grid": "#e2e8f0",
}

DESIGN_CSS = """
<style>
    /* Suppress default Streamlit chrome for clean presentation mode */
    #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; height: 2.2rem; }

    .main .block-container {
        padding-top: 0.75rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }

    /* Typography refinement */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #0f172a;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.25rem;
    }
    .sb-brand {
        border-left: 4px solid #0f2942;
        padding: 0.25rem 0 0.25rem 0.75rem;
        margin-bottom: 1.1rem;
    }
    .sb-brand-title {
        font-size: 1.0rem;
        font-weight: 700;
        color: #0f2942;
        line-height: 1.2;
    }
    .sb-brand-sub {
        font-size: 0.76rem;
        color: #64748b;
        margin-top: 0.2rem;
    }
    .sb-section-label {
        font-size: 0.70rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        margin: 0.9rem 0 0.4rem 0;
    }
    .health-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.80rem;
        color: #334155;
        padding: 0.28rem 0;
        border-bottom: 1px solid #f1f5f9;
    }
    .status-dot {
        width: 7px; height: 7px; border-radius: 50%; display: inline-block;
    }
    .dot-ok { background-color: #0f766e; }
    .dot-bad { background-color: #be123c; }

    /* Institutional Header */
    .top-header {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-top: 4px solid #0f2942;
        border-radius: 8px;
        padding: 1.15rem 1.5rem;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.75rem;
    }
    .top-header h1 {
        margin: 0;
        font-size: 1.55rem;
        font-weight: 700;
        color: #0f2942;
        letter-spacing: -0.015em;
    }
    .top-header .subtitle {
        margin: 0.2rem 0 0 0;
        font-size: 0.90rem;
        color: #475569;
    }
    .header-meta {
        font-size: 0.76rem;
        font-weight: 600;
        color: #0f2942;
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 0.35rem 0.75rem;
        white-space: nowrap;
    }

    /* Subtle Scope Callout */
    .notice-bar {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #1d4ed8;
        border-radius: 6px;
        padding: 0.65rem 1rem;
        margin-bottom: 1rem;
        font-size: 0.85rem;
        color: #334155;
        line-height: 1.48;
    }
    .notice-bar strong { color: #0f2942; }

    /* KPI Grid */
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.9rem 1.05rem;
        margin-bottom: 0.75rem;
    }
    .kpi-title {
        font-size: 0.71rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
    }
    .kpi-val {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0f2942;
        margin: 0.22rem 0 0.12rem 0;
        line-height: 1.15;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #475569;
    }

    /* Section Heading */
    .sec-head {
        font-size: 1.02rem;
        font-weight: 700;
        color: #0f2942;
        margin: 1.05rem 0 0.6rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #e2e8f0;
    }

    /* Workflow Steps */
    .wf-step {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 0.65rem;
        height: 108px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .wf-idx {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #1d4ed8;
    }
    .wf-name {
        font-size: 0.84rem;
        font-weight: 700;
        color: #0f2942;
        margin: 0.18rem 0;
    }
    .wf-detail {
        font-size: 0.74rem;
        color: #475569;
        line-height: 1.3;
    }

    /* Clean Content Panel */
    .panel {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.75rem;
    }
    .panel-title {
        font-size: 0.90rem;
        font-weight: 700;
        color: #0f2942;
        margin-bottom: 0.35rem;
    }
    .panel-body {
        font-size: 0.84rem;
        color: #334155;
        line-height: 1.5;
    }

    /* Tag */
    .tag {
        display: inline-block;
        font-size: 0.71rem;
        font-weight: 600;
        padding: 0.14rem 0.5rem;
        border-radius: 4px;
        margin-right: 0.35rem;
        vertical-align: middle;
    }
    .tag-real { background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; }
    .tag-gwas { background: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe; }
    .tag-sim  { background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }

    /* Risk Output Card */
    .risk-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.05rem 1.2rem;
        margin-bottom: 0.75rem;
    }
    .risk-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.65rem;
    }
    .risk-score-num {
        font-size: 2.05rem;
        font-weight: 700;
        color: #0f2942;
        line-height: 1;
    }
    .risk-tier-badge {
        font-size: 0.78rem;
        font-weight: 700;
        padding: 0.25rem 0.7rem;
        border-radius: 4px;
        color: #ffffff;
    }
    .bar-bg {
        position: relative;
        height: 10px;
        background-color: #e2e8f0;
        border-radius: 999px;
        overflow: hidden;
        margin-bottom: 0.4rem;
    }
    .bar-fill {
        height: 100%;
        border-radius: 999px;
    }
    .bar-ticks {
        display: flex;
        justify-content: space-between;
        font-size: 0.71rem;
        color: #64748b;
        font-weight: 600;
    }
    .risk-sub {
        margin-top: 0.5rem;
        font-size: 0.78rem;
        color: #475569;
    }

    .formula-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #0f2942;
        border-radius: 4px;
        padding: 0.65rem 0.95rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.85rem;
        color: #0f2942;
        margin: 0.5rem 0;
    }

    /* Tighter tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.25rem;
        border-bottom: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.45rem 0.15rem;
        font-weight: 600;
        font-size: 0.90rem;
    }
</style>
"""
st.markdown(DESIGN_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached Data & Model Loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def cached_model_bundle(model_key: str):
    return load_model_bundle(model_key)


@st.cache_resource(show_spinner=False)
def cached_pima_model():
    return load_pima_real_model()


@st.cache_data(show_spinner=False)
def cached_pima_data():
    return load_pima_real_dataset()


@st.cache_data(show_spinner=False)
def cached_gwas():
    return load_gwas_summary_stats()


@st.cache_data(show_spinner=False)
def cached_sim_summary():
    return load_simulated_cohort_summary()


@st.cache_data(show_spinner=False)
def cached_simulated_cohort():
    return load_simulated_cohort()


@st.cache_data(show_spinner=False)
def cached_panel_metadata():
    gwas_df = cached_gwas()
    _, panel_snps, _ = cached_simulated_cohort()
    return build_panel_metadata(gwas_df, panel_snps)


@st.cache_data(show_spinner=False)
def cached_saved_evals():
    return load_saved_evaluation_artifacts()


@st.cache_data(show_spinner=False)
def cached_pima_eval():
    return compute_pima_real_evaluation(cached_pima_data(), cached_pima_model())


@st.cache_data(show_spinner=False)
def cached_sim_eval(model_key: str):
    sim_df, _, _ = cached_simulated_cohort()
    return compute_simulated_test_evaluation(sim_df, cached_model_bundle(model_key))


@st.cache_data(show_spinner=False)
def cached_cross_cohort(model_key: str):
    return compute_cross_cohort_prs_comparison(cached_pima_data(), cached_model_bundle(model_key))


# ---------------------------------------------------------------------------
# Reusable Components
# ---------------------------------------------------------------------------
def style_fig(fig: go.Figure, height: int = 330, margin: dict | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a", size=12),
        title_font=dict(size=13.5, color="#0f2942"),
        margin=margin or dict(l=16, r=20, t=42, b=42),
    )
    fig.update_xaxes(gridcolor="#f1f5f9", zerolinecolor="#cbd5e1", automargin=True)
    fig.update_yaxes(gridcolor="#f1f5f9", zerolinecolor="#cbd5e1", automargin=True)
    return fig


def render_kpis(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items), gap="medium")
    for col, (title, val, sub) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">{html.escape(title)}</div>
                    <div class="kpi-val">{html.escape(val)}</div>
                    <div class="kpi-sub">{html.escape(sub)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_risk_card(prob: float, tier_label: str, subtitle: str) -> None:
    pct = float(np.clip(prob * 100.0, 0.0, 100.0))
    tier_palette = {
        "Low Risk": PALETTE["teal"],
        "Moderate Risk": PALETTE["amber"],
        "High Risk": "#ea580c",
        "Very High Risk": PALETTE["crimson"],
    }
    fill_color = tier_palette.get(tier_label, PALETTE["navy"])
    st.markdown(
        f"""
        <div class="risk-card">
            <div class="risk-top">
                <div>
                    <div style="font-size:0.71rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;color:#64748b;">
                        Model Risk Estimate
                    </div>
                    <div class="risk-score-num">{pct:.1f}%</div>
                </div>
                <span class="risk-tier-badge" style="background-color:{fill_color};">{html.escape(tier_label)}</span>
            </div>
            <div class="bar-bg">
                <div class="bar-fill" style="width:{pct:.2f}%;background-color:{fill_color};"></div>
            </div>
            <div class="bar-ticks">
                <span>0% Low</span>
                <span>25% Moderate</span>
                <span>50% High</span>
                <span>75% Very High</span>
                <span>100%</span>
            </div>
            <div class="risk-sub">{html.escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_shap_chart(shap_df: pd.DataFrame, title: str) -> None:
    ordered = shap_df.sort_values("shap_value", ascending=True)
    colors = [PALETTE["crimson"] if v >= 0 else PALETTE["teal"] for v in ordered["shap_value"]]
    fig = go.Figure(
        go.Bar(
            x=ordered["shap_value"],
            y=ordered["display_name"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in ordered["shap_value"]],
            textposition="outside",
        )
    )
    fig.add_vline(x=0, line_width=1.2, line_color="#475569")
    max_abs = max(0.25, float(ordered["abs_shap"].max()) * 1.32)
    fig.update_layout(
        title=title,
        xaxis_title="Log-Odds Impact (Teal = Protective · Crimson = Risk-Increasing)",
        yaxis_title="",
    )
    fig.update_xaxes(range=[-max_abs, max_abs])
    style_fig(fig, height=300, margin=dict(l=12, r=24, t=40, b=40))
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)


def render_header():
    st.markdown(
        """
        <div class="top-header">
            <div>
                <h1>T2D Genome Risk Prediction</h1>
                <div class="subtitle">An AI-assisted research framework integrating genetic and clinical risk factors.</div>
            </div>
            <div class="header-meta">DBT Prototype · Offline Ready</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_notice(content_html: str):
    st.markdown(f'<div class="notice-bar">{content_html}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page 1: Research Dashboard
# ---------------------------------------------------------------------------
def page_research_dashboard(model_key: str):
    render_notice(
        "<strong>Scientific Scope &amp; Provenance:</strong> Combines <strong>real DIAMANTE South Asian GWAS summary statistics</strong> "
        "(1,113 QC-passed variants) with a <strong>real 768-patient clinical dataset</strong> (UCI PIMA Indians Diabetes, 5-Fold CV AUC 0.839). "
        "Because individual-level patient genotypes linked to clinical outcomes require controlled biobank access (e.g., GenomeIndia / UK Biobank), "
        "the multi-modal PRS+Clinical model uses synthetically simulated genotypes as a proof-of-concept and is not for clinical diagnosis."
    )

    gwas_df = cached_gwas()
    _, sim_stats = cached_sim_summary()
    pima_df = cached_pima_data()
    pima_eval = cached_pima_eval()
    evals = cached_saved_evals()
    bundle = cached_model_bundle(model_key)

    render_kpis(
        [
            ("Raw DIAMANTE GWAS", "10.40M", "South Asian summary SNPs"),
            ("QC-Retained GWAS SNPs", f"{len(gwas_df):,}", "p < 1e-5, MAF > 0.01, unambiguous"),
            ("Real Clinical Cohort", f"{len(pima_df):,}", f"PIMA Patients · 5-Fold CV AUC {pima_eval['cv_auc_mean']:.3f}"),
            ("Hybrid Sim. Cohort", f"{sim_stats['n_individuals']:,}", f"{sim_stats['n_snps']} SNPs · HWE Proof-of-Concept"),
        ]
    )

    st.markdown('<div class="sec-head">End-to-End Research Workflow</div>', unsafe_allow_html=True)
    steps = [
        ("Stage 1", "GWAS Summary Stats", "DIAMANTE SAS T2D (10.4M SNPs)"),
        ("Stage 2", "SNP Selection", "Strand QC, MAF > 0.01, p < 1e-5"),
        ("Stage 3", "PRS Calculation", "200 SNPs: sum(dosage × beta)"),
        ("Stage 4", "Clinical Integration", "Real PIMA + IDRS risk factors"),
        ("Stage 5", "ML Prediction", "Calibrated Regularized XGBoost"),
        ("Stage 6", "Risk Interpretation", "TreeSHAP & variant attribution"),
    ]
    wf_cols = st.columns(len(steps), gap="small")
    for col, (idx, name, detail) in zip(wf_cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="wf-step">
                    <div class="wf-idx">{idx}</div>
                    <div class="wf-name">{name}</div>
                    <div class="wf-detail">{detail}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    c_left, c_right = st.columns([1.05, 1.0], gap="large")

    with c_left:
        qc_steps = pd.DataFrame(
            [
                {"Stage": "1. Raw Summary Stats", "SNPs": 10_401_621},
                {"Stage": "2. Complete Critical Fields", "SNPs": 10_401_548},
                {"Stage": "3. Strand Ambiguous Removed", "SNPs": 8_853_170},
                {"Stage": "4. MAF > 0.01 Filter", "SNPs": 7_377_446},
                {"Stage": "5. Retained (p < 1e-5)", "SNPs": 1_113},
                {"Stage": "6. PRS Simulation Panel", "SNPs": 200},
            ]
        )
        fig_qc = px.bar(
            qc_steps.iloc[::-1],
            x="SNPs",
            y="Stage",
            orientation="h",
            log_x=True,
            text="SNPs",
            title="DIAMANTE South Asian GWAS Preprocessing Funnel",
            color_discrete_sequence=[PALETTE["navy"]],
        )
        fig_qc.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_qc.update_layout(xaxis_title="Variant Count (Log10 Scale)", yaxis_title="")
        fig_qc.update_xaxes(range=[1.5, 8.3])
        style_fig(fig_qc, height=340, margin=dict(l=10, r=60, t=42, b=40))
        st.plotly_chart(fig_qc, width="stretch", config=PLOTLY_CONFIG)

    with c_right:
        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-title">
                    Track A — Real Clinical Model <span class="tag tag-real">Real Data (n = 768)</span>
                </div>
                <div class="panel-body">
                    Trained on <strong>UCI PIMA Indians Diabetes</strong> records using 8 measured clinical features
                    (median imputation + <code>StandardScaler</code> + regularized XGBoost).<br/>
                    <strong>Metrics:</strong> 5-Fold CV AUC = <strong>{pima_eval['cv_auc_mean']:.3f}</strong> ·
                    Test AUC = <strong>{pima_eval['auc']:.3f}</strong> · Specificity = <strong>{pima_eval['specificity']:.1%}</strong>.
                </div>
            </div>
            <div class="panel">
                <div class="panel-title">
                    Track B — Hybrid GWAS-PRS + Clinical <span class="tag tag-sim">Synthetic Cohort (n = 5,000)</span>
                </div>
                <div class="panel-body">
                    Active artifact: <code>{html.escape(bundle.model_path.name)}</code>. Integrates
                    <strong>200 real DIAMANTE GWAS weights</strong> (<code>PRS_raw = sum(dosage * beta)</code>)
                    with 5 IDRS clinical features across 5,000 HWE-simulated individuals.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "ablation_df" in evals:
            ab_xgb = evals["ablation_df"][evals["ablation_df"]["Model"] == "XGBoost"][
                ["Feature Set", "AUC-ROC", "Accuracy", "F1-Score"]
            ].copy()
            st.caption("Synthetic Cohort Ablation (XGBoost, n = 5,000 simulated individuals):")
            st.dataframe(
                ab_xgb.style.format({"AUC-ROC": "{:.4f}", "Accuracy": "{:.4f}", "F1-Score": "{:.4f}"}),
                width="stretch",
                hide_index=True,
            )

    st.markdown('<div class="sec-head">Data Provenance &amp; Research Limitations</div>', unsafe_allow_html=True)
    provenance_df = pd.DataFrame(
        [
            {
                "Component": "Real Clinical Patient Cohort",
                "Source": "data/pima_real_clinical.csv & models/pima_real_clinical_model.pkl",
                "Classification": "Real Clinical Data (n = 768)",
                "Notes": "Measured Glucose, BMI, Age, BP, Insulin, SkinFold, Pregnancies, Pedigree + verified outcomes.",
            },
            {
                "Component": "South Asian GWAS Weights",
                "Source": "processed_data/cleaned_gwas_summary_stats.csv & prs_weights.txt",
                "Classification": "Real Population GWAS (1,113 SNPs)",
                "Notes": "Real DIAMANTE SAS effect alleles, EAF, p-values, and log-odds betas; top-N approximates LD clumping.",
            },
            {
                "Component": "Hybrid Genotypes & Labels",
                "Source": "processed_data/simulated_t2d_dataset.csv & realistic_model.pkl",
                "Classification": "Simulated Proof-of-Concept (n = 5,000)",
                "Notes": "HWE-sampled 200-SNP genotypes & synthetic T2D labels; high AUC (~0.99+) is not clinical accuracy.",
            },
        ]
    )
    st.dataframe(provenance_df, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Page 2: Patient Risk Assessment
# ---------------------------------------------------------------------------
def page_risk_assessment(model_key: str):
    tab_real, tab_hybrid = st.tabs(
        [
            "Real Clinical Patient Assessment (PIMA, n = 768)",
            "Hybrid GWAS-PRS + Clinical Prototype (200 SNPs)",
        ]
    )
    with tab_real:
        _render_real_clinical_track()
    with tab_hybrid:
        _render_hybrid_gwas_track(model_key)


def _render_real_clinical_track():
    pima_df = cached_pima_data()
    pima_model = cached_pima_model()
    presets = get_curated_real_pima_patients(pima_df)

    render_notice(
        '<span class="tag tag-real">Real Clinical Data</span> '
        "Evaluated by the regularized XGBoost pipeline trained on <strong>768 real patients</strong> "
        "(5-Fold CV AUC = 0.839, Test AUC = 0.826). Inspect a verified cohort patient or enter custom clinical values."
    )

    col_in, col_out = st.columns([0.95, 1.15], gap="large")

    with col_in:
        with st.container(border=True):
            source_mode = st.radio(
                "Patient Source",
                ["Curated Cohort Cases", "Browse Cohort Index (0–767)", "Custom Clinical Input"],
                horizontal=True,
            )

            actual_outcome = None
            note = ""
            if source_mode == "Curated Cohort Cases":
                chosen = st.selectbox("Select Real Patient", [p["label"] for p in presets])
                rec = next(p for p in presets if p["label"] == chosen)
                vals = rec["values"]
                actual_outcome = rec["actual_outcome"]
                note = f"Cohort Row #{rec['patient_index']} · Verified Diagnosis: {'T2D Positive (1)' if actual_outcome == 1 else 'No T2D (0)'}"
            elif source_mode.startswith("Browse Cohort"):
                idx = st.number_input("Cohort Index (0–767)", 0, len(pima_df) - 1, 0, step=1)
                row = pima_df.iloc[int(idx)]
                actual_outcome = int(row["Outcome"])
                vals = {f: int(row[f]) if f in ("Pregnancies", "Age") else float(row[f]) for f in PIMA_FEATURES}
                for zc, fb in [("Glucose", 117.0), ("BloodPressure", 72.0), ("SkinThickness", 29.0), ("Insulin", 125.0), ("BMI", 32.0)]:
                    if vals[zc] == 0:
                        vals[zc] = fb
                note = f"Cohort Row #{int(idx)} · Verified Diagnosis: {'T2D Positive (1)' if actual_outcome == 1 else 'No T2D (0)'}"
            else:
                vals = {
                    "Pregnancies": 2,
                    "Glucose": 118.0,
                    "BloodPressure": 74.0,
                    "SkinThickness": 28.0,
                    "Insulin": 110.0,
                    "BMI": 28.5,
                    "DiabetesPedigreeFunction": 0.42,
                    "Age": 42,
                }
                note = "Custom patient profile evaluated against the real PIMA-trained model."

            st.caption(note)
            c1, c2 = st.columns(2)
            with c1:
                glucose = st.number_input("Plasma Glucose (mg/dL)", 40.0, 260.0, float(np.clip(vals["Glucose"], 40.0, 260.0)), step=1.0)
                bmi = st.number_input("BMI (kg/m²)", 14.0, 70.0, float(np.clip(vals["BMI"], 14.0, 70.0)), step=0.5)
                age = st.number_input("Age (years)", 18, 95, int(np.clip(vals["Age"], 18, 95)), step=1)
                pregnancies = st.number_input("Pregnancies", 0, 20, int(np.clip(vals["Pregnancies"], 0, 20)), step=1)
            with c2:
                bp = st.number_input("Diastolic BP (mm Hg)", 24.0, 140.0, float(np.clip(vals["BloodPressure"], 24.0, 140.0)), step=1.0)
                insulin = st.number_input("2-Hr Insulin (mu U/mL)", 10.0, 900.0, float(np.clip(vals["Insulin"], 10.0, 900.0)), step=5.0)
                skin = st.number_input("Skin Fold (mm)", 5.0, 100.0, float(np.clip(vals["SkinThickness"], 5.0, 100.0)), step=1.0)
                dpf = st.number_input("Diabetes Pedigree Fn", 0.05, 2.60, float(np.clip(vals["DiabetesPedigreeFunction"], 0.05, 2.60)), step=0.05)

    user_vals = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }
    valid, errs, cleaned = validate_pima_clinical_inputs(user_vals)

    with col_out:
        if not valid:
            for e in errs:
                st.error(e)
            return

        res = predict_pima_real_risk(cleaned, pima_model, include_shap=True)
        sub = "Model: pima_real_clinical_model.pkl (trained on 768 real patients; 5-Fold CV AUC 0.839)."
        if actual_outcome is not None and source_mode != "Custom Clinical Input":
            sub += f" Verified cohort label: {'T2D Positive' if actual_outcome == 1 else 'No T2D'}."
        render_risk_card(res["probability"], res["risk_tier"], sub)

        render_kpis(
            [
                ("Plasma Glucose", f"{cleaned['Glucose']:.0f} mg/dL", "Primary metabolic driver"),
                ("BMI / Age", f"{cleaned['BMI']:.1f} / {cleaned['Age']}y", f"BP: {cleaned['BloodPressure']:.0f} mm Hg"),
                ("Pedigree Score", f"{cleaned['DiabetesPedigreeFunction']:.2f}", "Familial risk index"),
            ]
        )

        if res["shap"] is not None:
            render_shap_chart(res["shap"]["shap_df"], "Real-Data Clinical Attribution (TreeSHAP)")
            st.caption(f"Base log-odds: {res['shap']['base_log_odds']:+.3f} · Net SHAP shift: {res['shap']['shap_sum']:+.3f}")


def _render_hybrid_gwas_track(model_key: str):
    bundle = cached_model_bundle(model_key)
    sim_df, _, sim_stats = cached_simulated_cohort()
    panel_df = cached_panel_metadata()
    profiles = get_curated_synthetic_profiles(sim_df)

    render_notice(
        '<span class="tag tag-gwas">Real DIAMANTE GWAS Weights (200 SNPs)</span> '
        '<span class="tag tag-sim">Simulated Individual Genotypes</span> '
        "Demonstrates how additive SNP allele dosages (<code>0/1/2</code>) aggregate into Raw PRS and integrate with IDRS clinical factors."
    )

    top_editable_snps = panel_df.sort_values("abs_beta", ascending=False).head(10).reset_index(drop=True)
    if "rs7903146" in panel_df["rsID"].values and "rs7903146" not in top_editable_snps["rsID"].values:
        tcf7l2 = panel_df[panel_df["rsID"] == "rs7903146"].iloc[[0]]
        top_editable_snps = pd.concat([tcf7l2, top_editable_snps.iloc[:9]], ignore_index=True)

    col_in, col_out = st.columns([0.95, 1.15], gap="large")

    with col_in:
        with st.container(border=True):
            demo_mode = st.radio(
                "Genetic Demonstration Mode",
                ["Sample Synthetic Patient", "Simulated Genotype Input"],
                horizontal=True,
            )

            if demo_mode == "Sample Synthetic Patient":
                sub_mode = st.radio("Cohort Selection", ["5 Tiered Profiles", "Cohort Index (0–4999)"], horizontal=True)
                if sub_mode == "5 Tiered Profiles":
                    chosen_label = st.selectbox("Select Synthetic Profile", [p["label"] for p in profiles])
                    prof = next(p for p in profiles if p["label"] == chosen_label)
                    default_clin = prof["clinical"]
                    dosage_map = dict(prof["dosages"])
                    st.caption(f"{prof['description']} (Row #{prof['cohort_index']}, synthetic label={prof['simulated_t2d_label']})")
                else:
                    idx = st.number_input("Simulated Cohort Row", 0, len(sim_df) - 1, 0, step=1)
                    row = sim_df.iloc[int(idx)]
                    default_clin = {
                        "age": int(round(row["age"])),
                        "bmi": float(round(row["bmi"], 1)),
                        "waist_cm": float(round(row["waist_cm"], 1)),
                        "family_history": int(row["family_history"]),
                        "physical_activity": int(row["physical_activity"]),
                    }
                    dosage_map = {c: int(row[c]) for c in panel_df["rsID"]}
                    st.caption(f"Row #{int(idx)} · Synthetic label: {'T2D (1)' if int(row['T2D_status']) == 1 else 'No T2D (0)'}")

                c1, c2 = st.columns(2)
                with c1:
                    age = st.number_input("Age (years)", 18, 95, int(np.clip(default_clin["age"], 18, 95)), step=1)
                    bmi = st.number_input("BMI (kg/m²)", 14.0, 60.0, float(np.clip(default_clin["bmi"], 14.0, 60.0)), step=0.5)
                    waist_cm = st.number_input("Waist (cm)", 50.0, 160.0, float(np.clip(default_clin["waist_cm"], 50.0, 160.0)), step=1.0)
                with c2:
                    family_history = st.selectbox("Family History", [0, 1], index=int(default_clin["family_history"]), format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
                    physical_activity = st.selectbox("Physical Activity", [0, 1], index=int(default_clin["physical_activity"]), format_func=lambda x: "Active (1)" if x == 1 else "Sedentary (0)")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    age = st.slider("Age (years)", 18, 95, 48, 1)
                    bmi = st.slider("BMI (kg/m²)", 14.0, 55.0, 27.5, 0.5)
                    waist_cm = st.slider("Waist (cm)", 50.0, 150.0, 92.0, 1.0)
                with c2:
                    family_history = st.radio("Family History", [0, 1], index=0, format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
                    physical_activity = st.radio("Physical Activity", [0, 1], index=1, format_func=lambda x: "Active (1)" if x == 1 else "Sedentary (0)")

                g1, g2 = st.columns([1.45, 1.0])
                with g1:
                    gen_profile = st.selectbox(
                        "HWE Background Preset",
                        ["average", "low_genetic_risk", "high_genetic_risk"],
                        format_func=lambda x: {
                            "average": "Average Population Risk (Unbiased HWE)",
                            "low_genetic_risk": "Low Genetic Risk Preset",
                            "high_genetic_risk": "High Genetic Risk Preset",
                        }[x],
                    )
                with g2:
                    sim_seed = st.number_input("HWE Seed", 1, 9999, 42, 1)
                dosage_map = simulate_hwe_genotypes(panel_df, profile=gen_profile, seed=int(sim_seed))

            with st.expander("Override Top 10 Variant Dosages (0, 1, or 2 Effect Alleles)", expanded=False):
                for r in top_editable_snps.itertuples(index=False):
                    dosage_map[r.rsID] = int(
                        st.selectbox(
                            f"{r.rsID} · Chr{r.chr} · {r.effect_allele}/{r.other_allele} · EAF {r.eaf:.3f} · β={r.beta:+.4f}",
                            options=[0, 1, 2],
                            index=int(dosage_map[r.rsID]),
                            key=f"hyb_{demo_mode[:5]}_{r.rsID}",
                        )
                    )

    ok, errs, cleaned = validate_clinical_inputs(age, bmi, waist_cm, family_history, physical_activity)
    with col_out:
        if not ok:
            for e in errs:
                st.error(e)
            return

        prs_out = calculate_prs_from_genotypes(dosage_map, panel_df, cohort_stats=sim_stats)
        prs_raw = prs_out["prs_raw"]
        prs_z = prs_out["prs_z"]
        prs_pct = compute_prs_percentile(prs_raw, sim_df["PRS"])
        pred = predict_t2d_risk(cleaned, prs_raw, bundle, include_shap=True)

        render_risk_card(
            pred["probability"],
            pred["risk_tier"],
            f"Demonstration tiers derived from synthetic model output ({bundle.model_path.name}).",
        )
        render_kpis(
            [
                ("Raw PRS (200 SNPs)", f"{prs_raw:.3f}", f"Cohort Z-score: {prs_z:+.2f}"),
                ("PRS Percentile", f"{prs_pct:.1f}th", "Rank in 5,000 sim. cohort"),
                ("Demo Classification", "High Risk" if pred["predicted_class"] == 1 else "Low Risk", "At 0.50 demo threshold"),
            ]
        )

        if pred["shap"] is not None:
            render_shap_chart(pred["shap"]["shap_df"], "Feature Contributions (5-Fold TreeSHAP Log-Odds)")
            st.caption(f"Base log-odds: {pred['shap']['base_log_odds']:+.3f} · Net SHAP shift: {pred['shap']['shap_sum']:+.3f}")

    st.markdown('<div class="sec-head">Top Variant Contributions to Simulated PRS</div>', unsafe_allow_html=True)
    contrib_df = prs_out["contributions"].copy()
    contrib_df["abs_contrib"] = contrib_df["contribution"].abs()
    top_c = contrib_df.sort_values("abs_contrib", ascending=False).head(12)

    vc1, vc2 = st.columns([1.1, 1.0], gap="large")
    with vc1:
        fig_c = px.bar(
            top_c.sort_values("contribution"),
            x="contribution",
            y="rsID",
            orientation="h",
            color="contribution",
            color_continuous_scale=[PALETTE["teal"], "#cbd5e1", PALETTE["crimson"]],
            hover_data=["chr", "effect_allele", "dosage", "beta", "eaf"],
            title="Top 12 Variant Dosage Contributions (dosage × GWAS beta)",
        )
        fig_c.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Contribution to Raw PRS")
        style_fig(fig_c, height=340, margin=dict(l=12, r=18, t=40, b=40))
        st.plotly_chart(fig_c, width="stretch", config=PLOTLY_CONFIG)
    with vc2:
        disp_c = top_c[["rsID", "chr", "effect_allele", "eaf", "beta", "dosage", "contribution"]].copy()
        disp_c.columns = ["rsID", "Chr", "Effect Allele", "EAF", "Beta", "Dosage", "Contribution"]
        st.dataframe(
            disp_c.style.format({"EAF": "{:.3f}", "Beta": "{:+.4f}", "Contribution": "{:+.4f}"}),
            width="stretch",
            hide_index=True,
            height=340,
        )


# ---------------------------------------------------------------------------
# Page 3: Genetic Risk Explorer
# ---------------------------------------------------------------------------
def page_genetic_risk_explorer():
    render_notice(
        '<span class="tag tag-gwas">Real DIAMANTE South Asian GWAS</span> '
        "Exploring all <strong>1,113 QC-passed variants</strong> from <code>cleaned_gwas_summary_stats.csv</code> "
        "and the <strong>200-SNP PRS panel</strong>."
    )

    gwas_df = cached_gwas().copy()
    panel_df = cached_panel_metadata()
    panel_set = set(panel_df["rsID"])
    gwas_df["in_200_panel"] = gwas_df["rsID"].isin(panel_set)
    lead = gwas_df.sort_values("pval").iloc[0]

    render_kpis(
        [
            ("Retained GWAS SNPs", f"{len(gwas_df):,}", "p < 1e-5 & MAF > 0.01"),
            ("Model PRS Panel", f"{len(panel_df)}", "Top SNPs ranked by p-value"),
            ("Chromosomes", f"{gwas_df['chr'].nunique()}", "Autosomal chromosomes"),
            ("Lead Variant", f"{lead['rsID']}", f"p = {lead['pval']:.2e} (Chr {int(lead['chr'])})"),
        ]
    )

    with st.expander("PRS Formula & Allele Orientation Reference", expanded=False):
        st.write(
            "All strand-ambiguous A/T and C/G SNPs (1,548,378 variants) were removed during QC so each variant's "
            "effect allele is unambiguously oriented. Genotype dosage G_i in {0, 1, 2} counts copies of effect_allele:"
        )
        st.markdown('<div class="formula-box">PRS_raw = sum_{i=1..200} ( dosage_i * beta_i )</div>', unsafe_allow_html=True)
        st.caption(
            "Because many common DIAMANTE effect alleles carry negative log-odds betas, raw PRS values center around "
            "-8.66 +/- 1.31 (range -12.95 to -3.12). Less negative values correspond to higher genetic risk."
        )

    st.markdown('<div class="sec-head">Searchable GWAS Variant Catalog</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns([1.0, 0.85, 1.25])
    with f1:
        scope = st.selectbox("Variant Scope", ["200-SNP PRS Model Panel", "All 1,113 Retained GWAS SNPs"])
    with f2:
        chroms = ["All"] + [str(c) for c in sorted(gwas_df["chr"].unique())]
        chosen_chr = st.selectbox("Chromosome", chroms)
    with f3:
        rs_query = st.text_input("Search rsID (e.g., rs7903146)", "").strip().lower()

    view_df = gwas_df[gwas_df["in_200_panel"]].copy() if scope.startswith("200-SNP") else gwas_df.copy()
    if chosen_chr != "All":
        view_df = view_df[view_df["chr"] == int(chosen_chr)]
    if rs_query:
        view_df = view_df[view_df["rsID"].str.lower().str.contains(rs_query)]
    view_df = view_df.sort_values("pval").reset_index(drop=True)

    st.caption(f"Showing {len(view_df):,} variants.")
    tbl = view_df[["rsID", "chr", "pos", "effect_allele", "other_allele", "eaf", "beta", "se", "pval", "neg_log10_pval", "in_200_panel"]].copy()
    tbl.columns = ["rsID", "Chr", "Position", "Effect Allele", "Other Allele", "EAF", "Beta", "SE", "P-value", "-log10(P)", "In 200-SNP Panel"]
    st.dataframe(
        tbl.style.format({"EAF": "{:.4f}", "Beta": "{:+.4f}", "SE": "{:.4f}", "P-value": "{:.2e}", "-log10(P)": "{:.2f}"}),
        width="stretch",
        hide_index=True,
        height=310,
    )

    st.markdown('<div class="sec-head">Single-Variant Dosage Impact &amp; Genome Overview</div>', unsafe_allow_html=True)
    chosen_snp = st.selectbox("Select Panel Variant to Inspect 0 / 1 / 2 Dosage Impact", panel_df.sort_values("pval")["rsID"].tolist())
    snp_row = panel_df[panel_df["rsID"] == chosen_snp].iloc[0]

    d1, d2 = st.columns([1.0, 1.1], gap="large")
    with d1:
        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-title">{snp_row['rsID']} · Chr {int(snp_row['chr'])}:{int(snp_row['pos']):,}</div>
                <div class="panel-body">
                    <strong>Effect Allele:</strong> <code>{snp_row['effect_allele']}</code> (Other: <code>{snp_row['other_allele']}</code>) ·
                    <strong>EAF:</strong> {snp_row['eaf']:.4f}<br/>
                    <strong>GWAS Beta:</strong> {snp_row['beta']:+.4f} (SE {snp_row['se']:.4f}) ·
                    <strong>P-value:</strong> {snp_row['pval']:.2e}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        dosage_df = pd.DataFrame(
            [
                {"Genotype": f"{snp_row['other_allele']}/{snp_row['other_allele']}", "Dosage": 0, "HWE Freq": (1 - snp_row["eaf"]) ** 2, "PRS Contribution": 0.0},
                {"Genotype": f"{snp_row['effect_allele']}/{snp_row['other_allele']}", "Dosage": 1, "HWE Freq": 2 * snp_row["eaf"] * (1 - snp_row["eaf"]), "PRS Contribution": float(snp_row["beta"])},
                {"Genotype": f"{snp_row['effect_allele']}/{snp_row['effect_allele']}", "Dosage": 2, "HWE Freq": snp_row["eaf"] ** 2, "PRS Contribution": float(2 * snp_row["beta"])},
            ]
        )
        st.dataframe(dosage_df.style.format({"HWE Freq": "{:.1%}", "PRS Contribution": "{:+.4f}"}), width="stretch", hide_index=True)
    with d2:
        fig_d = px.bar(
            dosage_df,
            x="Dosage",
            y="PRS Contribution",
            text=[f"{v:+.4f}" for v in dosage_df["PRS Contribution"]],
            title=f"Additive PRS Contribution for {snp_row['rsID']}",
            color_discrete_sequence=[PALETTE["navy"]],
        )
        fig_d.update_xaxes(tickvals=[0, 1, 2], ticktext=["0 (Ref/Ref)", "1 (Het)", "2 (Eff/Eff)"])
        style_fig(fig_d, height=280, margin=dict(l=14, r=18, t=40, b=40))
        st.plotly_chart(fig_d, width="stretch", config=PLOTLY_CONFIG)

    v1, v2 = st.columns(2, gap="large")
    with v1:
        top20 = panel_df.sort_values("abs_beta", ascending=False).head(20).sort_values("beta")
        fig_b = px.bar(
            top20,
            x="beta",
            y="rsID",
            orientation="h",
            color="beta",
            color_continuous_scale=[PALETTE["teal"], "#cbd5e1", PALETTE["crimson"]],
            hover_data=["chr", "effect_allele", "eaf", "pval"],
            title="Top 20 Panel SNPs by |Beta| Effect Size",
        )
        fig_b.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="GWAS Beta")
        style_fig(fig_b, height=400, margin=dict(l=12, r=18, t=42, b=42))
        st.plotly_chart(fig_b, width="stretch", config=PLOTLY_CONFIG)
    with v2:
        gwas_plot = gwas_df.copy()
        gwas_plot["Set"] = np.where(gwas_plot["in_200_panel"], "200-SNP Model Panel", "Other Retained (p<1e-5)")
        fig_m = px.scatter(
            gwas_plot,
            x="chr",
            y="neg_log10_pval",
            color="Set",
            color_discrete_map={"200-SNP Model Panel": PALETTE["crimson"], "Other Retained (p<1e-5)": PALETTE["blue"]},
            hover_data=["rsID", "pos", "beta", "eaf", "pval"],
            title="Chromosomal Significance (-log10 P-value)",
            labels={"chr": "Chromosome", "neg_log10_pval": "-log10(p-value)"},
        )
        fig_m.add_hline(y=-np.log10(5e-8), line_dash="dash", line_color=PALETTE["navy"], annotation_text="p = 5e-8")
        fig_m.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, title=""))
        style_fig(fig_m, height=400, margin=dict(l=12, r=18, t=54, b=42))
        st.plotly_chart(fig_m, width="stretch", config=PLOTLY_CONFIG)


# ---------------------------------------------------------------------------
# Page 4: Model Insights
# ---------------------------------------------------------------------------
def page_model_insights(model_key: str):
    tab_real, tab_sim, tab_transfer = st.tabs(
        [
            "Real Clinical Model (PIMA, n = 768)",
            "Hybrid GWAS-PRS Prototype (Simulated, n = 5,000)",
            "Cross-Cohort Transfer & Saved Figures",
        ]
    )

    with tab_real:
        render_notice(
            '<span class="tag tag-real">Real Clinical Dataset (UCI PIMA, n = 768)</span> '
            "Regularized XGBoost trained with median imputation on an 80% stratified split and evaluated on 20% held-out real patients (n = 154)."
        )
        pe = cached_pima_eval()
        render_kpis(
            [
                ("5-Fold CV AUC", f"{pe['cv_auc_mean']:.3f}", f"+/- {pe['cv_auc_std']:.3f} across all 768 patients"),
                ("Holdout Test AUC", f"{pe['auc']:.3f}", f"n = {pe['n_test']} real test patients"),
                ("Test Accuracy", f"{pe['accuracy']:.1%}", f"F1-Score: {pe['f1']:.3f}"),
                ("Sensitivity / Specificity", f"{pe['recall']:.1%} / {pe['specificity']:.1%}", "At 0.50 threshold"),
            ]
        )

        r1, r2, r3 = st.columns(3, gap="medium")
        with r1:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=pe["fpr"], y=pe["tpr"], mode="lines", name=f"XGBoost (AUC {pe['auc']:.3f})", line=dict(color=PALETTE["blue"], width=2.5)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color=PALETTE["muted"], dash="dash")))
            fig_roc.update_layout(
                title="ROC Curve (Real Test Split)",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                legend=dict(x=0.36, y=0.06, bgcolor="rgba(255,255,255,0.90)", bordercolor="#e2e8f0", borderwidth=1),
            )
            style_fig(fig_roc, height=340, margin=dict(l=12, r=14, t=42, b=46))
            st.plotly_chart(fig_roc, width="stretch", config=PLOTLY_CONFIG)
        with r2:
            fig_cm = px.imshow(pe["cm"], text_auto=True, x=["Pred 0", "Pred 1"], y=["True 0", "True 1"], color_continuous_scale="Blues", title="Confusion Matrix (n = 154)")
            fig_cm.update_layout(coloraxis_showscale=False)
            style_fig(fig_cm, height=340, margin=dict(l=12, r=14, t=42, b=46))
            st.plotly_chart(fig_cm, width="stretch", config=PLOTLY_CONFIG)
        with r3:
            fig_gs = px.bar(
                pe["global_shap_df"].sort_values("Mean |SHAP|"),
                x="Mean |SHAP|",
                y="Feature",
                orientation="h",
                title="Global Mean |SHAP| (Real Data)",
                color_discrete_sequence=[PALETTE["navy"]],
            )
            fig_gs.update_layout(yaxis_title="")
            style_fig(fig_gs, height=340, margin=dict(l=12, r=14, t=42, b=46))
            st.plotly_chart(fig_gs, width="stretch", config=PLOTLY_CONFIG)

    with tab_sim:
        render_notice(
            '<span class="tag tag-sim">Simulated Cohort Evaluation (n = 5,000)</span> '
            "Genotypes and T2D labels were generated from a mathematical risk model; AUC (~0.99+) validates pipeline integration rather than clinical accuracy."
        )
        evals = cached_saved_evals()
        if "ablation_df" in evals:
            ab_df = evals["ablation_df"]
            fig_ab = px.bar(
                ab_df,
                x="Model",
                y="AUC-ROC",
                color="Feature Set",
                barmode="group",
                text_auto=".3f",
                color_discrete_sequence=[PALETTE["muted"], PALETTE["blue"], PALETTE["navy"]],
                title="Ablation Study: Clinical Only vs Genetic Only vs Combined (Simulated Cohort)",
            )
            fig_ab.update_yaxes(range=[0.88, 1.015])
            fig_ab.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, title=""))
            style_fig(fig_ab, height=360, margin=dict(l=12, r=16, t=58, b=46))
            st.plotly_chart(fig_ab, width="stretch", config=PLOTLY_CONFIG)

            st.dataframe(
                ab_df[["Feature Set", "Model", "AUC-ROC", "Accuracy", "Recall", "Specificity", "F1-Score"]].style.format(
                    {"AUC-ROC": "{:.4f}", "Accuracy": "{:.4f}", "Recall": "{:.4f}", "Specificity": "{:.4f}", "F1-Score": "{:.4f}"}
                ),
                width="stretch",
                hide_index=True,
            )

        bundle = cached_model_bundle(model_key)
        se = cached_sim_eval(model_key)
        st.markdown(f'<div class="sec-head">Holdout Split Evaluation ({html.escape(bundle.model_path.name)}, n = {se["n_test"]})</div>', unsafe_allow_html=True)
        render_kpis(
            [
                ("Test AUC-ROC", f"{se['auc']:.4f}", "Simulated holdout"),
                ("Accuracy", f"{se['accuracy']:.4f}", f"F1-Score: {se['f1']:.4f}"),
                ("Sensitivity", f"{se['recall']:.4f}", "True positive rate"),
                ("Specificity", f"{se['specificity']:.4f}", "True negative rate"),
            ]
        )

        s1, s2, s3 = st.columns(3, gap="medium")
        with s1:
            fig_roc_s = go.Figure()
            fig_roc_s.add_trace(go.Scatter(x=se["fpr"], y=se["tpr"], mode="lines", name=f"AUC {se['auc']:.4f}", line=dict(color=PALETTE["blue"], width=2.5)))
            fig_roc_s.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color=PALETTE["muted"], dash="dash")))
            fig_roc_s.update_layout(
                title="ROC Curve (Simulated Test)",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                legend=dict(x=0.45, y=0.06, bgcolor="rgba(255,255,255,0.90)", bordercolor="#e2e8f0", borderwidth=1),
            )
            style_fig(fig_roc_s, height=330, margin=dict(l=12, r=14, t=42, b=46))
            st.plotly_chart(fig_roc_s, width="stretch", config=PLOTLY_CONFIG)
        with s2:
            fig_cm_s = px.imshow(se["cm"], text_auto=True, x=["Pred 0", "Pred 1"], y=["True 0", "True 1"], color_continuous_scale="Blues", title="Confusion Matrix")
            fig_cm_s.update_layout(coloraxis_showscale=False)
            style_fig(fig_cm_s, height=330, margin=dict(l=12, r=14, t=42, b=46))
            st.plotly_chart(fig_cm_s, width="stretch", config=PLOTLY_CONFIG)
        with s3:
            fig_gs_s = px.bar(
                se["global_shap_df"].sort_values("Mean |SHAP|"),
                x="Mean |SHAP|",
                y="Feature",
                orientation="h",
                title=f"Global Mean |SHAP| (n = {len(se['X_shap_sample'])})",
                color_discrete_sequence=[PALETTE["navy"]],
            )
            fig_gs_s.update_layout(yaxis_title="")
            style_fig(fig_gs_s, height=330, margin=dict(l=12, r=14, t=42, b=46))
            st.plotly_chart(fig_gs_s, width="stretch", config=PLOTLY_CONFIG)

    with tab_transfer:
        render_notice(
            '<span class="tag tag-real">Cross-Cohort Diagnostic</span> '
            "Explains why evaluating the South Asian hybrid model on PIMA (<code>PRS = 0.0</code>) gave AUC 0.752 yet 0% specificity at threshold 0.50, vs. cohort-neutral PRS imputation."
        )
        cc_df = cached_cross_cohort(model_key)
        st.dataframe(
            cc_df.style.format({"AUC-ROC": "{:.3f}", "Accuracy": "{:.3f}", "Sensitivity": "{:.3f}", "Specificity": "{:.3f}", "Mean Pred Prob": "{:.3f}"}),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Passing PRS = 0.0 to a scaler centered at raw PRS = -8.93 shifts every patient +7.6 SD above the mean, "
            "collapsing specificity to 0.000. Imputing neutral raw PRS (-8.66) restores 99.2% specificity (67.1% accuracy), "
            "while training directly on PIMA's real 8 clinical features achieves 82.6% test AUC and 75.3% accuracy."
        )

        with st.expander("Pre-Generated Repository Figures (processed_data/)", expanded=False):
            plot_specs = [
                ("qc_plots.png", "GWAS Preprocessing QC Plots (Real DIAMANTE SAS Summary Stats)"),
                ("model_comparison.png", "4-Algorithm Ablation Comparison (Simulated Cohort)"),
                ("probability_calibration.png", "Probability Calibration Before vs After (Simulated Cohort)"),
                ("calibration_plot.png", "Calibration Reliability Curve (Simulated Cohort)"),
                ("shap_summary.png", "SHAP Feature Summary Plot (Baseline XGBoost)"),
                ("pima_evaluation_plots.png", "Original PIMA Evaluation Plots (PRS = 0.0 Fallback)"),
            ]
            cols = st.columns(2)
            idx = 0
            for fname, caption in plot_specs:
                fpath = PROCESSED_DIR / fname
                if fpath.exists():
                    with cols[idx % 2]:
                        st.image(str(fpath), caption=caption, width="stretch")
                    idx += 1


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main():
    render_header()

    with st.sidebar:
        st.markdown(
            """
            <div class="sb-brand">
                <div class="sb-brand-title">T2D Risk Framework</div>
                <div class="sb-brand-sub">DBT Research Demonstration</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sb-section-label">Navigation</div>', unsafe_allow_html=True)
        page = st.radio(
            "Navigation",
            [
                "1. Research Dashboard",
                "2. Patient Risk Assessment",
                "3. Genetic Risk Explorer",
                "4. Model Insights",
            ],
            label_visibility="collapsed",
        )

        st.markdown('<div class="sb-section-label">Hybrid Model Artifact</div>', unsafe_allow_html=True)
        model_choice = st.selectbox(
            "Hybrid Model",
            options=["realistic", "baseline"],
            format_func=lambda k: "Calibrated XGBoost (Realistic)" if k == "realistic" else "Uncalibrated XGBoost (Baseline)",
            label_visibility="collapsed",
        )

        st.markdown('<div class="sb-section-label">Artifact Health</div>', unsafe_allow_html=True)
        status = get_artifact_status()
        checks = [
            ("pima_real_data", "Real Clinical Data (768)"),
            ("pima_real_model", "Real Clinical Model"),
            ("gwas_cleaned", "DIAMANTE GWAS (1,113)"),
            ("realistic_model", "Hybrid Calibrated Model"),
            ("simulated_dataset", "Simulated Cohort (5,000)"),
        ]
        for key, label in checks:
            dot = "dot-ok" if status.get(key) else "dot-bad"
            st.markdown(
                f'<div class="health-item"><span>{label}</span><span class="status-dot {dot}"></span></div>',
                unsafe_allow_html=True,
            )

    try:
        if page.startswith("1."):
            page_research_dashboard(model_choice)
        elif page.startswith("2."):
            page_risk_assessment(model_choice)
        elif page.startswith("3."):
            page_genetic_risk_explorer()
        elif page.startswith("4."):
            page_model_insights(model_choice)
    except FileNotFoundError as exc:
        st.error(f"Required data or model artifact missing: {exc}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected error rendering view: {exc}")


if __name__ == "__main__":
    main()
