# Mentor Briefing & Explanation Guide: T2D Genome Risk Prediction

**Purpose:** A complete, honest, and technically rigorous guide to help you explain this project—what is real, what is simulated, what key scientific findings were uncovered, and how to demonstrate it live—to your research mentor and DBT (Department of Biotechnology) evaluators.

---

## 1. The 60-Second Summary (How to Open the Conversation)

> *"We built an interactive, offline-ready Type 2 Diabetes risk prediction prototype with two complementary tracks grounded in real data:*
>
> *1. **Real Population Genetics:** We processed 10.40 million SNPs from the DIAMANTE South Asian T2D GWAS down to 1,113 quality-controlled variants and a 200-SNP Polygenic Risk Score (PRS) panel using real effect sizes (`beta`) and allele frequencies (`eaf`).*
>
> *2. **Real Clinical Modeling:** Because paired individual-level DNA genotypes and clinical records require controlled biobank access (such as GenomeIndia or UK Biobank), we did not fake real patient genetics. Instead, we trained and validated a separate clinical XGBoost pipeline directly on **768 real patients** from the UCI PIMA Indians Diabetes dataset (`5-Fold CV AUC = 0.839`, `Holdout Test AUC = 0.826`, `Specificity = 85.0%`), alongside our **5,000-individual simulated South Asian hybrid prototype** (`realistic_model.pkl`) that demonstrates how clinical factors and 200-SNP PRS combine with TreeSHAP explainability.*
>
> *This gives us a working, scientifically transparent proof-of-concept ready for tomorrow's DBT demonstration and a clear roadmap for Phase II validation on Indian genotyped cohorts."*

---

## 2. The Most Important Scientific Distinction (Be Proactive About This)

Mentors and DBT reviewers will immediately test whether you understand the boundary between **real data** and **simulated data**. State this clearly before they even ask:

| Component | Source in Repository | Is It Real? | Exact Explanation for Your Mentor |
| :--- | :--- | :--- | :--- |
| **GWAS SNP Weights & Allele Frequencies** | `processed_data/cleaned_gwas_summary_stats.csv` & `prs_weights.txt` | **Yes — Real Population Data** | Derived from **DIAMANTE South Asian T2D GWAS** (`10,401,621` raw variants filtered to `1,113` QC-passed SNPs at `p < 1e-5`, `MAF > 0.01`, strand-unambiguous). |
| **Real Clinical Model & Patient Cohort** | `data/pima_real_clinical.csv` & `models/pima_real_clinical_model.pkl` | **Yes — Real Clinical Data** | **768 real clinical patients** with 8 measured clinical parameters (`Glucose`, `BMI`, `Age`, `BloodPressure`, `Insulin`, `SkinThickness`, `Pregnancies`, `DiabetesPedigreeFunction`) and verified diabetes outcomes (`5-Fold CV AUC = 0.839`, `Test AUC = 0.826`). |
| **5,000-Individual Hybrid Cohort & Labels** | `processed_data/simulated_t2d_dataset.csv` & `models/realistic_model.pkl` | **Simulated Proof-of-Concept** | Genotypes for the top 200 SNPs were sampled using Hardy-Weinberg Equilibrium (`EAF` from DIAMANTE), and synthetic T2D labels were generated from a combined genetic + clinical formula to test the multi-modal ML pipeline. |

**Why this framing works with your mentor:**
- You are **not** overselling synthetic accuracy (`0.99+` AUC) as clinical accuracy.
- You **do** have a genuinely trained, real-patient clinical model (`0.839` CV AUC) and real South Asian GWAS genetics (`1,113` SNPs).
- The remaining gap—paired individual-level South Asian DNA genotypes + clinical outcomes—is **exactly why DBT / GenomeIndia-scale biobank collaboration is the next research milestone**.

---

## 3. Three Technical Insights You Should Highlight (Shows Deep Engineering Rigor)

Your mentor will be especially impressed if you mention these three concrete findings discovered during validation:

### Insight A: Catching & Explaining the Raw PRS vs. Zero-Imputation Bug
- **What happened earlier:** In the repository's original `simulated_t2d_dataset.csv`, the `PRS` column is the **raw** additive sum:
  $$\text{PRS}_{\text{raw}} = \sum_{i=1}^{200} \text{dosage}_i \times \beta_i$$
  which has a cohort mean of **`-8.66`** (`SD = 1.31`, range `-12.95` to `-3.12`), and `realistic_scaler.pkl` expects raw PRS centered around **`-8.93`** (`SD = 1.18`).
- **Why the earlier PIMA script (`test_on_pima.py`) had `0.0%` specificity:** Because PIMA has no DNA genotypes, the old script passed `PRS = 0.0`. To `realistic_scaler.pkl`, `0.0` looks **`+7.6` standard deviations above the genetic mean**, forcing predicted probabilities above `79%–95%` for every patient (`AUC = 0.752`, `Accuracy = 0.351`, `Specificity = 0.000` at threshold `0.50`).
- **How we fixed & verified it:**
  1. Our Streamlit backend computes exact raw PRS directly from the 200-SNP dosage vector (`< 1e-12` difference from `simulated_t2d_dataset.csv`) while also showing the intuitive **Cohort Z-score** and **Percentile**.
  2. In Page 4 (**Cross-Cohort Transfer** tab), we show side-by-side that imputing the **cohort-neutral raw PRS (`-8.66`)** instead of `0.0` restores **Specificity from `0.0%` to `99.2%`** (`Accuracy` rises from `35.1%` to `67.1%`).

### Insight B: Fixing Overconfident 100% Predictions via Calibration
- The original uncalibrated XGBoost (`best_model_ml.pkl`) frequently predicted `99.9%–100.0%` certainty.
- The calibrated model (`realistic_model.pkl`) applies **10% feature noise**, **12% prevalence balancing**, **L1/L2 regularization** (`max_depth=3`, `reg_alpha=1.0`, `reg_lambda=2.0`), and **5-fold sigmoid Platt calibration** (`CalibratedClassifierCV`), producing smooth risk probabilities across all four demonstration tiers (`0.1%` Low → `35.8%` Moderate → `60.9%` High → `95.1%` Very High).

### Insight C: Proper Handling of Missing Clinical Zeros in Real Data
- In the UCI PIMA dataset, physiologically impossible `0` values (e.g., `Glucose = 0`, `BMI = 0`, `Insulin = 0`, `BloodPressure = 0`, `SkinThickness = 0`) encode missing measurements.
- In `scripts/train_pima_real_model.py`, those `0` values are converted to `NaN` and imputed inside a `scikit-learn` `Pipeline` (`SimpleImputer(strategy='median')` → `StandardScaler` → regularized `XGBClassifier`) strictly inside cross-validation/train splits to prevent data leakage.

---

## 4. Screen-by-Screen Walkthrough for Your Mentor

Launch the app with `run_app.bat` (or open `http://localhost:8501`) and walk through the 4 pages in order:

### Page 1: Research Dashboard
- **Point to:**
  - The **4 KPI cards** (`10.40M` raw DIAMANTE SNPs → `1,113` QC SNPs → `768` real PIMA patients → `5,000` simulated hybrid cohort).
  - The **6-stage End-to-End Research Workflow** (`GWAS Summary Stats → SNP Selection → PRS Calculation → Clinical Integration → ML Prediction → Risk Interpretation`).
  - The **GWAS Preprocessing Funnel** (`10,401,621` → `10,401,548` non-missing → `8,853,170` after removing `1,548,378` strand-ambiguous `A/T` & `C/G` SNPs → `7,377,446` after `MAF > 0.01` → `1,113` at `p < 1e-5` → `200`-SNP panel).
  - The **Dual-Track Architecture** & **Data Provenance** table.
- **Say:** *"This page establishes our pipeline and clearly separates real GWAS + real clinical data from our simulated multi-modal proof-of-concept."*

### Page 2: Patient Risk Assessment (Two Tabs)
- **Tab 1 — Real Clinical Patient Assessment (PIMA, n = 768):**
  - Pick **Real Patient #1 — Healthy Young Adult (Outcome: No T2D)** → show low predicted probability, verified diagnosis, and protective (teal) TreeSHAP bars.
  - Switch to **Real Patient #4 — High-Risk Hyperglycemic Case (Outcome: T2D)** → show high predicted probability driven by `Plasma Glucose`, `BMI`, and `Age`.
  - Mention that users can also enter **Custom Clinical Input** with physiological range validation.
  - **Say:** *"This tab runs on a model trained exclusively on 768 real clinical patients with verified outcomes."*
- **Tab 2 — Hybrid GWAS-PRS + Clinical Prototype (200 SNPs):**
  - Show **5 Tiered Profiles** (`Low Risk — Young & Active` at `0.1%` vs. `Moderate Risk` at `35.8%` vs. `High Genetic Risk` at `60.9%` vs. `Very High Combined Risk` at `95.1%`).
  - Switch to **Simulated Genotype Input**, expand **Override Top 10 Variant Dosages**, and change a lead SNP (e.g., `rs7903146` in *TCF7L2*) from `0` to `2` copies of the effect allele to show Raw PRS, PRS Percentile, Risk Score, and TreeSHAP updating live.
  - **Say:** *"This tab shows how individual SNP dosages (`0, 1, 2`) sum into a Polygenic Risk Score and combine with clinical risk factors once patient genotyping data is available."*

### Page 3: Genetic Risk Explorer
- **Point to:**
  - The searchable table of all **`1,113` real DIAMANTE South Asian variants** (filterable by `200-SNP PRS Model Panel`, chromosome, and `rsID`, e.g. `rs7903146`).
  - The **Single-Variant Dosage Impact** tool showing `0 / 1 / 2` genotype frequencies under Hardy-Weinberg Equilibrium and exact `dosage × beta` contributions.
  - The **Top 20 Panel SNPs by |Beta|** and **Chromosomal Significance (`-log10 P-value`)** plots.
- **Say:** *"Every variant, allele, frequency, and effect size here is real South Asian GWAS data; strand-ambiguous A/T and C/G SNPs were filtered out so allele orientation is unambiguous."*

### Page 4: Model Insights (Three Tabs)
- **Tab 1 — Real Clinical Model (PIMA, n = 768):** Show `5-Fold CV AUC = 0.839`, `Holdout Test AUC = 0.826`, `Accuracy = 75.3%`, `Specificity = 85.0%`, ROC curve, Confusion Matrix, and Global Mean `|SHAP|`.
- **Tab 2 — Hybrid GWAS-PRS Prototype (Simulated, n = 5,000):** Show the 4-algorithm ablation study (`Logistic Regression`, `Random Forest`, `Gradient Boosting`, `XGBoost` across `Clinical Only`, `Genetic Only`, `Combined`), explicitly stating that `AUC ~0.99+` reflects simulated data generation.
- **Tab 3 — Cross-Cohort Transfer & Saved Figures:** Show the `PRS = 0.0` vs. neutral `PRS = -8.66` comparison table and explain the scaler centering discovery.

---

## 5. Likely Questions From Your Mentor (With Direct Answers)

**Q1: "Is the model actually trained on real genetic data, or is it simulated?"**  
**A:** *"The genetic weights (`beta`, `eaf`, `p-values` for 1,113 SNPs) are 100% real from the DIAMANTE South Asian GWAS, and our clinical baseline model (`CV AUC 0.839`) is trained on 100% real clinical patients from the UCI PIMA cohort. What is simulated is the paired individual-level genotype matrix (`5,000 × 200` SNPs), because individual patient DNA genotypes linked to diabetes status require controlled-access biobank cohorts like GenomeIndia or UK Biobank."*

**Q2: "Why did you use the PIMA Indians dataset if our GWAS is South Asian?"**  
**A:** *"PIMA is a widely benchmarked, open-access clinical diabetes dataset with verified outcomes, allowing us to validate a real-patient clinical ML pipeline (`CV AUC 0.839`) and test cross-cohort behavior offline. We explicitly document the ancestry mismatch (`Native American` vs. `South Asian`) and the absence of measured DNA genotypes in PIMA, which is why we keep it separated from the South Asian PRS prototype."*

**Q3: "Why is the simulated AUC 0.999? Won't reviewers think that's data leakage or overfitting?"**  
**A:** *"Yes—if presented as real accuracy, 0.999 would immediately be rejected. We explicitly label it as a synthetic methodology benchmark because the simulated labels were generated from a mathematical combination of PRS and clinical features. In real-world clinical + PRS cohorts, published literature reports combined AUCs around `0.75–0.85`. Our real-data clinical model already hits `0.839` CV AUC, and our regularized calibrated hybrid model (`realistic_model.pkl`) removes 100% overconfident predictions."*

**Q4: "Why are Raw PRS values negative (around `-8.66`)?"**  
**A:** *"Raw PRS is `sum(dosage_i * beta_i)` where `dosage_i` counts copies (`0, 1, 2`) of the GWAS `effect_allele`. Many common DIAMANTE effect alleles have negative log-odds `beta` values relative to the other allele, so the raw sum averages `-8.66`. Higher (less negative) values still mean higher genetic risk, and the UI displays both Raw PRS and the intuitive Cohort Z-score / Percentile."*

**Q5: "What happens when someone enters invalid or missing patient data?"**  
**A:** *"The prediction backend enforces strict bounds checking (`validate_clinical_inputs` and `validate_pima_clinical_inputs`) for physiological ranges, rejects non-finite values, checks exact feature ordering against the saved `StandardScaler`, and requires explicit imputation flags for missing SNPs rather than silently inventing genotypes."*

**Q6: "What is our concrete next step after this DBT prototype?"**  
**A:** *"Three clear steps: (1) obtain IRB-approved access to an Indian/South Asian genotyped cohort (e.g., GenomeIndia or institutional clinical biobank), (2) replace top-N SNP selection with formal South Asian LD clumping (`PLINK` / `PRS-CS` using 1000G SAS reference), and (3) retrain and prospectively calibrate the integrated clinical + PRS model on measured genotypes and clinical biomarkers (HbA1c, fasting glucose, lipids)."*

---

## 6. Key Numbers Cheat Sheet (Verified From Repository & Test Runs)

- **Raw DIAMANTE SAS GWAS:** `10,401,621` SNPs → **`1,113`** QC-passed SNPs (`p < 1e-5`, `MAF > 0.01`, strand-unambiguous) → **`200`**-SNP PRS model panel.
- **Real Clinical Model (`pima_real_clinical_model.pkl`, `n = 768`):**
  - `5-Fold CV AUC`: **`0.8386 ± 0.0244`**
  - `Holdout Test AUC (n = 154)`: **`0.8261`**
  - `Accuracy`: **`75.3%`** | `Sensitivity`: **`57.4%`** | `Specificity`: **`85.0%`** | `F1`: **`0.620`**
- **Hybrid Simulated Cohort (`n = 5,000`):**
  - Raw PRS mean: **`-8.66`** (`SD = 1.31`, range `-12.95` to `-3.12`)
  - Demo profiles (`realistic_model.pkl`): Low Risk **`0.1%`**, Moderate Risk **`35.8%`**, High Genetic Risk **`60.9%`**, Very High Combined Risk **`95.1%`**.
- **Automated Verification:** `pytest` suite (`tests/test_app_and_pipeline.py`) passes **10/10** checks, including full UI rendering across all 4 pages.
