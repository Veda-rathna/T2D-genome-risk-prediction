# 5-Minute DBT Demonstration Script (`DEMO_SCRIPT.md`)

**Target Audience:** DBT (Department of Biotechnology) Officials & Scientific Evaluators  
**Total Duration:** ~5 minutes  
**Pre-launch Check (15 minutes before meeting):**
1. Double-click `run_app.bat` (or run `python -m streamlit run app.py`).
2. Confirm `http://localhost:8501` loads cleanly and all 5 sidebar **Artifact Health Check** indicators show `✅`.
3. Confirm the sidebar **Trained Model Artifact** is set to **Calibrated XGBoost (Realistic)**.

---

## Minute 0:00 – 1:00 | Page 1: Research Dashboard (Problem & Pipeline)
- **Navigate to:** `1. Research Dashboard`
- **What to point at:**
  1. The header & **6-step Research Workflow** (`GWAS Summary Statistics → SNP Selection → PRS Calculation → Clinical Feature Integration → Machine Learning Prediction → Risk Interpretation`).
  2. The **KPI row**: `10,401,621` raw DIAMANTE South Asian SNPs → `1,113` QC-passed SNPs → `200`-SNP PRS panel → `5,000` simulated cohort.
  3. The **GWAS Variant Filtering Pipeline** chart & **Data Provenance** table.
- **Key talking points:**
  > "This prototype demonstrates how South Asian population GWAS knowledge can be combined with IDRS-aligned clinical parameters for Type 2 Diabetes risk stratification. Notice our strict scientific transparency: GWAS weights and allele frequencies are real DIAMANTE South Asian statistics, while individual genotypes and labels are simulated as a proof-of-concept until measured Indian cohort data is integrated."

---

## Minute 1:00 – 2:30 | Page 2: Patient Risk Assessment (Real Clinical + Hybrid GWAS-PRS)
- **Navigate to:** `2. Patient Risk Assessment`
- **Step A (1:00 – 1:45) — Track A: Real Clinical Patient Assessment (100% Real Data):**
  1. Keep **Real Clinical Patient Assessment (Real UCI PIMA Cohort & Model)** selected.
  2. Pick **Real Patient #1 — Healthy Young Adult (Outcome: No T2D)** → show low predicted risk and negative protective TreeSHAP bars.
  3. Switch to **Real Patient #4 — High-Risk Hyperglycemic Case (Outcome: T2D)** → show high predicted probability matching verified diagnosis, driven by Glucose, BMI, and Age.
- **Step B (1:45 – 2:30) — Track B: Hybrid GWAS-PRS + Clinical Assessment:**
  1. Switch the top Assessment Track to **Hybrid GWAS-PRS + Clinical Assessment (DIAMANTE 200-SNP PRS Prototype)**.
  2. Show how **Sample Synthetic Patient** spans all four tiers (`Low` ~0.1%, `Moderate` ~35.8%, `High` ~60.9%, `Very High` ~95.1%).
  3. Switch to **Simulated Genotype Input** and adjust a lead variant (e.g., `rs7903146` *TCF7L2*) from `0` to `2` effect alleles to show live Raw PRS, Z-score, Percentile, and TreeSHAP updates.

---

## Minute 2:30 – 3:30 | Page 3: Genetic Risk Explorer (Real GWAS Data)
- **Navigate to:** `3. Genetic Risk Explorer`
- **What to point at:**
  1. Search for `rs7903146` (or filter by Chromosome / `200-SNP PRS Model Panel`).
  2. Show the **Interactive Variant Dosage-to-PRS Contribution Demo** showing genotype `0 / 1 / 2` frequencies under Hardy-Weinberg Equilibrium and exact `dosage × beta` contributions.
  3. Scroll briefly to the **Top 20 Panel SNPs by |Beta|** and **Significance Across Chromosomes (-log10 P-value)** plots.
- **Key talking points:**
  > "Every SNP, effect allele, allele frequency, and beta coefficient here comes directly from the cleaned DIAMANTE South Asian GWAS summary statistics. Strand-ambiguous A/T and C/G SNPs were removed during QC to guarantee unambiguous allele orientation."

---

## Minute 3:30 – 4:45 | Page 4: Model Insights (Real Clinical vs. Hybrid & Cross-Cohort Analysis)
- **Navigate to:** `4. Model Insights`
- **What to point at:**
  1. **Tab 1 — Real Clinical Model (PIMA, n=768):** Highlight `5-Fold CV AUC = 0.839`, `Holdout Test AUC = 0.826`, `Accuracy = 75.3%`, `Specificity = 85.0%`, and real clinical TreeSHAP ranking (`Glucose`, `BMI`, `Age`).
  2. **Tab 2 — Hybrid GWAS-PRS Prototype (Simulated, n=5,000):** Show the 4-algorithm ablation chart (`Clinical Only` vs `Genetic Only` vs `Combined`) and note transparently that AUC ~0.99+ reflects synthetic simulation.
  3. **Tab 3 — Cross-Cohort Transfer & Saved Figures:** Show why `PRS = 0.0` caused `0.0%` specificity in the earlier script vs. `99.2%` specificity under cohort-neutral PRS imputation (`-8.66`).

---

## Minute 4:45 – 5:00 | Closing & Next Steps
- **Key closing statement:**
  > "In summary, the end-to-end pipeline—from 10.4 million South Asian GWAS variants to calibrated, SHAP-explainable multi-modal risk assessment—is fully operational offline today. With access to measured Indian individual-level genotype cohorts (e.g., GenomeIndia / ICMR clinical biobanks), this exact framework can be retrained and prospectively validated for translational healthcare research."
