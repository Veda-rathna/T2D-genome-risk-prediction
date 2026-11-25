# Presentation Guide: T2D Genome Risk Prediction (DBT Demonstration)

## 1. Research Problem

Type 2 Diabetes (T2D) represents a major public health burden in South Asia, where individuals develop T2D at younger ages and lower Body Mass Index (BMI) thresholds compared to European populations. Conventional screening tools such as the Indian Diabetes Risk Score (IDRS) rely exclusively on clinical and lifestyle factors (age, waist circumference, family history, physical activity) and achieve modest discrimination (typically AUC-ROC ~0.66–0.75), missing individuals who carry high inherited genetic predisposition despite normal BMI or younger age.

Conversely, genome-wide association studies (GWAS) produce population-level summary statistics (allele frequencies, effect sizes/betas, p-values) rather than individual-level patient predictions, and Eurocentric polygenic risk scores generalize poorly to South Asian ancestries. There is a clear research need for an interpretable, ancestry-aligned framework that integrates South Asian GWAS-derived Polygenic Risk Scores (PRS) with clinical risk parameters.

---

## 2. Aim and Objectives

**Aim:** To develop and demonstrate an AI-assisted research framework integrating South Asian population-level genetic risk (via Polygenic Risk Scores) with clinical and lifestyle risk factors for Type 2 Diabetes risk stratification.

**Objectives:**
1. Preprocess and quality-control DIAMANTE South Asian T2D GWAS summary statistics to derive an unambiguous, ancestry-relevant PRS weight panel.
2. Construct an individual-level genotype simulation and PRS calculation pipeline anchored to real GWAS effect sizes (`beta`) and allele frequencies (`eaf`) under Hardy-Weinberg Equilibrium (HWE).
3. Integrate PRS with five clinical/lifestyle parameters (Age, BMI, Waist Circumference, Family History, Physical Activity) and evaluate multiple machine learning classifiers (Logistic Regression, Random Forest, Gradient Boosting, XGBoost).
4. Mitigate simulation overfitting through realistic noise injection, prevalence balancing (12%), regularization, and Platt probability calibration.
5. Provide patient-level explainability via TreeSHAP and per-variant PRS dosage attribution in an interactive offline-ready demonstration dashboard.

---

## 3. Dataset Details & Provenance

A core scientific principle of this project is **strict separation of real vs. simulated data**:

| Data Layer | Source | Scale / Content | Classification |
| :--- | :--- | :--- | :--- |
| **GWAS Summary Statistics** | DIAMANTE South Asian T2D GWAS (`cleaned_gwas_summary_stats.csv`, `prs_weights.txt`) | 10,401,621 raw variants → **1,113** QC-passed SNPs (`p < 1e-5`, `MAF > 0.01`, strand-unambiguous) | **Real population-level GWAS data** |
| **PRS Variant Panel** | Top 200 SNPs ranked by GWAS p-value (including `rs7903146` *TCF7L2*) | 200 SNPs with `effect_allele`, `other_allele`, `eaf`, `beta`, `se`, `pval` | **Real GWAS weights** |
| **Individual Genotypes** | `scripts/simulate_and_train.py` (`simulated_t2d_dataset.csv`) | 5,000 synthetic individuals × 200 SNPs sampled under HWE | **Simulated genetic data** |
| **Clinical Features & Labels** | `scripts/simulate_and_train.py` & `train_realistic_model.py` | Age, BMI, waist_cm, family_history, physical_activity + synthetic `T2D_status` | **Simulated clinical & label data** |
| **Real Clinical Patient Cohort** | UCI PIMA Indians Diabetes Dataset (`data/pima_real_clinical.csv` & `models/pima_real_clinical_model.pkl`) | 768 real patients; 8 clinical measurements + verified outcomes; **5-Fold CV AUC = 0.839, Test AUC = 0.826** | **Real clinical data & model** |

*(Note: `scripts/add_genetics_to_pima.py` generates synthetic genotypes correlated with PIMA outcomes for exploratory workflow testing only and is not treated as real genetic validation.)*

---

## 4. Methodology

### Stage 1: GWAS Preprocessing (`gwas_preprocessing.py`)
- **Initial load:** 10,401,621 SNPs → 10,401,548 after removing 73 rows with missing critical fields.
- **Strand ambiguity removal:** Excluded 1,548,378 palindromic `A/T` and `C/G` SNPs (remaining: 8,853,170).
- **MAF filtering (`eaf > 0.01`):** Excluded 1,475,724 rare variants (remaining: 7,377,446).
- **Significance & QC filtering (`p < 1e-5`, `|beta| < 5`, deduplication):** Retained **1,113** high-confidence SNPs (`processed_data/cleaned_gwas_summary_stats.csv` and `processed_data/prs_weights.txt`).

### Stage 2: PRS Formulation
For each individual, genotype dosage $G_i \in \{0, 1, 2\}$ counts copies of the GWAS `effect_allele` across the top 200 SNPs:
$$\text{PRS}_{\text{raw}} = \sum_{i=1}^{200} G_i \cdot \beta_i$$
In the 5,000-individual simulated cohort, raw PRS values span `-12.95` to `-3.12` (mean `-8.66`, SD `1.31`). Because many common DIAMANTE effect alleles have negative `beta` weights, raw PRS is negative on average; higher (less negative) values correspond to higher polygenic risk.

### Stage 3: Clinical Integration & Machine Learning
- **Feature schema (exact order):** `['age', 'bmi', 'waist_cm', 'family_history', 'physical_activity', 'PRS']` standardized via `StandardScaler`.
- **Baseline model (`models/best_model_ml.pkl`):** Uncalibrated `XGBClassifier` (`n_estimators=200`) trained on 5,000 simulated individuals (25% T2D prevalence).
- **Calibrated Realistic model (`models/realistic_model.pkl`, default in app):** `CalibratedClassifierCV` (5-fold sigmoid Platt calibration) wrapping a regularized `XGBClassifier` (`max_depth=3`, `learning_rate=0.05`, `reg_alpha=1.0`, `reg_lambda=2.0`, `min_child_weight=5`) trained after adding 10% Gaussian feature noise and balancing to 12% prevalence.

### Stage 4: Explainability
- **Global & Local TreeSHAP:** Computes additive log-odds contributions across the 5 calibrated XGBoost base estimators, showing how each clinical and genetic feature shifts risk relative to the cohort baseline.
- **Variant Attribution:** Breaks down $\text{PRS}_{\text{raw}}$ into per-SNP contributions $G_i \cdot \beta_i$.

---

## 5. System Architecture

- **`src/t2d_pipeline.py`**: Core Python prediction and bioinformatics engine (artifact verification, schema checks, clinical bounds validation, allele-aligned PRS calculation, HWE genotype simulation, calibrated inference, TreeSHAP attribution, and test-split evaluation).
- **`app.py`**: Four-page Streamlit research web application (`1. Research Dashboard`, `2. Patient Risk Assessment`, `3. Genetic Risk Explorer`, `4. Model Insights`).
- **`models/`**: Pre-trained serialized artifacts (`realistic_model.pkl`, `realistic_scaler.pkl`, `best_model_ml.pkl`, `scaler.pkl`) loaded once and cached (`@st.cache_resource`).
- **`processed_data/`**: Preprocessed GWAS summary stats, PRS weights, 5,000-individual simulated cohort, ablation evaluation CSV, PIMA evaluation CSVs, and QC/calibration figures.
- **`run_app.bat`**: One-click offline Windows launcher (`python -m streamlit run app.py`).

---

## 6. Existing Implementation & Application Workflow

1. **Research Dashboard:** Presents project aims, the 6-stage pipeline, DIAMANTE QC funnel (10.4M → 1,113 → 200 SNPs), model details, synthetic vs. real results, and data provenance limitations.
2. **Patient Risk Assessment Demonstration:** Allows interactive testing via either **Simulated Genotype Input** (HWE presets + per-SNP `0/1/2` dosage overrides for top variants) or **Sample Synthetic Patient** (5 curated profiles + full 5,000-cohort browser), outputting calibrated demonstration risk score, raw/Z-score/percentile PRS, TreeSHAP explanation, and top variant contributions.
3. **Genetic Risk Explorer:** Queries the 1,113 retained DIAMANTE SNPs and 200-SNP panel, visualizes `-log10(p)` across chromosomes and top `|beta|` variants, and demonstrates `0/1/2` allelic dosage impact.
4. **Model Insights:** Separately reports **Simulated Cohort** ablation/ROC/confusion/calibration/SHAP metrics and **Real PIMA Clinical** evaluation (`AUC-ROC = 0.752`, `PRS = 0.0`).

---

## 7. Demonstration Instructions

1. Double-click `run_app.bat` (or run `python -m streamlit run app.py` from the repository root).
2. Open `http://localhost:8501` in a browser (runs 100% offline).
3. Follow the 5-minute sequence in [DEMO_SCRIPT.md](DEMO_SCRIPT.md).

---

## 8. Experimental Results and Their Limitations

### Simulated Cohort Results (Synthetic Genotypes & Labels)
From `processed_data/model_evaluation_results.csv` (5,000 simulated individuals):
- **XGBoost Ablation:** Clinical Only AUC = `0.9731` | Genetic Only (PRS) AUC = `0.9756` | Combined AUC = `0.9994` (Logistic Regression Combined AUC = `1.0000`).
- **Calibrated Realistic Model (`realistic_model.pkl`) on 20% Holdout Split (n = 1,000):** High rank-ordering (AUC > `0.99`) with smooth, non-extreme probabilities (eliminating the uncalibrated model's 100% certainty spike).
- **Critical Limitation:** Because synthetic T2D labels were generated mathematically from PRS and clinical features, simulated AUC (~0.99+) is artificially inflated. Real-world integrated T2D models typically achieve AUC `0.75–0.85`.

### External Clinical Dataset Results (PIMA Indians Diabetes, n = 757, No Measured Genetics)
From `processed_data/pima_metrics.csv`:
- **AUC-ROC:** `0.752` (demonstrating meaningful clinical rank-ordering transfer).
- **Threshold-0.50 Metrics:** Accuracy = `0.351`, Sensitivity = `1.000`, Specificity = `0.000`.
- **Technical Cause of Low Specificity:** PIMA lacks DNA genotypes (PRS was set to `0.0`), whereas the scaler expects raw PRS centered at `-8.93` (SD `1.18`), placing every patient `+7.6 SD` above the training genetic mean; additionally `waist_cm` was estimated as `3.2 * BMI + 50` (mean `153.9 cm` vs. training mean `69.1 cm`).

---

## 9. Current Research Gaps & Why This Is a Proof-of-Concept

1. **No Individual-Level Indian Genotyped Cohort:** Training relied on HWE simulation rather than measured patient arrays/WGS.
2. **LD Approximation:** Top-N p-value selection (`1,113` / `200` SNPs) was used instead of formal LD clumping (`PLINK --clump` with a 1000 Genomes South Asian reference panel).
3. **Missing Clinical Biomarkers:** Fasting plasma glucose, HbA1c, lipid profile, and blood pressure are not in the current 6-feature model schema.
4. **Population Calibration:** Risk thresholds (`<25%`, `25–50%`, `50–75%`, `≥75%`) are demonstration tiers derived from synthetic output, not prospective clinical incidence cutoffs.

### Data Required for Real-World Genomic Validation
- IRB-approved South Asian / Indian cohort (`N >= 5,000–20,000`) with paired genotyping (array or low-pass WGS) and longitudinal EHR/clinical follow-up (e.g., ICMR-INDIAB or regional biobank cohorts).
- Reference LD panel (1000G SAS / GenomeIndia) for proper PRS clumping or Bayesian PRS (`PRS-CS`, `LDpred2`).
- Prospective calibration and external multi-center validation before any clinical decision-support evaluation.

---

## 10. Future Improvements

1. Replace top-N SNP selection with formal South Asian LD clumping / `PRS-CS` weights.
2. Train and calibrate directly on real genotyped Indian cohorts (e.g., GenomeIndia + clinical registries).
3. Expand clinical inputs to include fasting glucose, HbA1c, triglycerides/HDL, and blood pressure.
4. Implement missing-genotype imputation standards and ancestry principal components (PC1–PC10) adjustment for PRS.

---

## 11. Likely DBT Official Questions & Scientifically Accurate Answers

**Q1. Your simulated AUC is 0.99+. Does that mean this AI predicts diabetes with 99% accuracy in real patients?**  
**A:** No. In `simulate_and_train.py`, synthetic disease labels were generated from a formula combining PRS and clinical factors, so the model recovers that synthetic relationship with AUC ~0.99. We explicitly label this as a methodology validation metric. In real clinical cohorts, published T2D clinical+PRS models typically achieve AUC around `0.75–0.85`, which is still clinically meaningful over clinical-only baselines (~0.66–0.75).

**Q2. Why did you simulate genotypes instead of using real Indian patient genomes?**  
**A:** Publicly accessible DIAMANTE South Asian GWAS provides population summary statistics (`beta`, `eaf`, `pval`), not individual-level genotypes, which require controlled-access biobank/DBT-funded cohorts. Simulating 5,000 individuals under Hardy-Weinberg Equilibrium lets us validate the end-to-end PRS calculation, ML integration, calibration, and SHAP pipeline today, so the software is immediately ready to ingest real VCF/PLINK genotypes once cohort access is approved.

**Q3. Why are the Raw PRS values negative (around -8.7) in your system?**  
**A:** Each SNP contribution is `dosage * beta`, where `dosage` counts copies (`0, 1, 2`) of the GWAS `effect_allele`. In DIAMANTE summary statistics, many common effect alleles carry negative log-odds `beta` values with high allele frequencies (`eaf`), making the raw additive sum negative (`-12.95` to `-3.12`). Importantly, higher (less negative) values still mean higher genetic risk, and we also display the cohort Z-score and percentile for intuitive interpretation.

**Q4. Why did the PIMA external evaluation have AUC 0.752 yet 0% specificity at threshold 0.50?**  
**A:** That is a great example of why external validation matters. AUC `0.752` shows the model correctly ranks higher-risk PIMA patients above lower-risk ones using clinical features alone. However, because PIMA has no genetic data, `PRS` was set to `0.0` while the scaler expected raw PRS around `-8.93`—effectively treating every PIMA patient as `+7.6 SD` genetically high-risk and pushing all probabilities above `0.79`. Fixing that requires neutral imputation at the cohort mean raw PRS (`-8.66`) or retraining without genetics for clinical-only fallback.

**Q5. How does this framework add value specifically for Indian/South Asian populations?**  
**A:** Unlike Eurocentric risk models, our PRS weights come directly from **DIAMANTE South Asian** GWAS summary statistics (`10.4M` variants filtered to `1,113` high-confidence SAS loci), and the clinical inputs align with Indian Diabetes Risk Score (IDRS) parameters (including waist circumference, which captures central adiposity common in South Asians at lower BMI).

**Q6. What would be the next milestone if DBT funds/approves Phase II?**  
**A:** Partnering with an Indian clinical biobank / GenomeIndia dataset to: (1) perform true South Asian LD clumping (`PLINK`/`PRS-CS`), (2) retrain and recalibrate on measured individual genotypes and longitudinal T2D outcomes, and (3) run prospective multi-center validation.
