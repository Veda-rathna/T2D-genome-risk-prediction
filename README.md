# T2D Genome Risk Prediction

*An AI-assisted research framework integrating genetic and clinical risk factors.*

## Quick Start

Double-click **`run_app.bat`** or launch from the repository root:

```powershell
python -m streamlit run app.py --server.port 8501
```

Then open **`http://localhost:8501`** in your browser (100% offline-ready).

---

## Repository Structure

```text
T2D-genome-risk-prediction/
├── app.py                        # Four-page Streamlit research application
├── run_app.bat                   # Windows one-click startup script
├── requirements.txt              # Python dependencies
├── .streamlit/config.toml        # Unified navy/slate Streamlit theme
├── src/
│   └── t2d_pipeline.py           # Core PRS, validation, ML prediction & SHAP backend
├── data/
│   └── pima_real_clinical.csv    # Real UCI PIMA clinical cohort (n = 768)
├── models/
│   ├── pima_real_clinical_model.pkl  # Regularized XGBoost trained on real PIMA data
│   ├── realistic_model.pkl           # Calibrated regularized XGBoost (hybrid prototype)
│   ├── realistic_scaler.pkl          # StandardScaler for realistic_model.pkl
│   ├── best_model_ml.pkl             # Baseline uncalibrated XGBoost (hybrid prototype)
│   └── scaler.pkl                    # StandardScaler for best_model_ml.pkl
├── processed_data/               # Cleaned DIAMANTE SAS GWAS (1,113 SNPs), PRS weights,
│                                 # 5,000-individual simulated cohort, metrics & figures
├── scripts/                      # GWAS preprocessing, training & evaluation scripts
├── notebooks/                    # Exploratory Jupyter notebooks
├── tests/
│   └── test_app_and_pipeline.py  # Automated pipeline & UI test suite
└── docs/
    ├── presentation/             # DBT & mentor briefing materials
    └── guides/                   # Original technical & methodology documentation
```

---

## Key Presentation Documents

- **Mentor Explanation Guide:** [docs/presentation/MENTOR_BRIEFING.md](docs/presentation/MENTOR_BRIEFING.md)
- **Full DBT Presentation Reference:** [docs/presentation/PRESENTATION_GUIDE.md](docs/presentation/PRESENTATION_GUIDE.md)
- **5-Minute Live Demo Script:** [docs/presentation/DEMO_SCRIPT.md](docs/presentation/DEMO_SCRIPT.md)

---

## Scientific Scope & Provenance

- **Real Population Genetics:** DIAMANTE South Asian T2D GWAS summary statistics (`10,401,621` raw SNPs → `1,113` QC-passed variants → `200`-SNP PRS panel).
- **Real Clinical Data & Model:** UCI PIMA Indians Diabetes dataset (`n = 768`, `5-Fold CV AUC = 0.839`, `Holdout Test AUC = 0.826`).
- **Simulated Hybrid Prototype:** `5,000` synthetic South Asian profiles generated under Hardy-Weinberg Equilibrium for methodology proof-of-concept only (not clinically validated for diagnosis).

---

## Run Tests

```powershell
python -m pytest -v tests/test_app_and_pipeline.py
```
