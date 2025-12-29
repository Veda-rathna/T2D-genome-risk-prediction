# T2D Risk Prediction: Simulation-Based Approach

## Overview

This project demonstrates a **simulation-based methodology** for integrating population-specific genetic markers from GWAS with clinical risk factors to enhance Type 2 Diabetes (T2D) risk prediction.

## ⚠️ Important Caveats

### Why Simulation?

**You cannot train machine learning models directly on GWAS summary statistics.** GWAS data contains:
- Population-level effect sizes (betas)
- Population allele frequencies
- **NOT** individual genotypes

To train predictive models, you need **individual-level data** (genotypes + phenotypes).

### Our Approach

We simulate individual genotypes based on:
1. **Real effect sizes** from DIAMANTE South Asian T2D GWAS
2. **Real allele frequencies** from the same study
3. **Hardy-Weinberg equilibrium** for realistic genotype distributions
4. **Polygenic risk scores (PRS)** calculated from validated GWAS weights

Clinical features are generated with realistic correlations to genetic risk.

### Validity

✅ **Valid for:**
- Methodology demonstration
- Proof-of-concept
- Framework development
- Academic exploration

❌ **Not valid for:**
- Clinical deployment
- Real patient prediction
- Publishing as clinical tool

### Next Steps for Real-World Use

To deploy this model clinically, you would need:
1. **Individual genotype data** from real patients (e.g., genotyping arrays, WGS)
2. **Electronic health records** with clinical measurements
3. **IRB approval** and patient consent
4. **Validation** on independent cohorts

## Files Generated

### Input
- `processed_data/cleaned_gwas_summary_stats.csv` - 1,113 filtered GWAS variants

### Output
- `processed_data/simulated_t2d_dataset.csv` - 5,000 simulated individuals
- `models/best_model_ml.pkl` - Trained XGBoost model
- `models/scaler.pkl` - Feature scaler
- `processed_data/shap_summary.png` - SHAP interpretability plot
- `processed_data/simulation_log.txt` - Processing log

## How to Use

### Run Simulation & Training

```bash
python scripts/simulate_and_train.py
```

This will:
1. Load cleaned GWAS summary statistics
2. Simulate 5,000 individuals with genotypes for top 200 SNPs
3. Calculate PRS for each individual
4. Simulate clinical features (age, BMI, waist, family history, activity)
5. Assign T2D phenotypes based on genetic + clinical risk
6. Train XGBoost classifier on combined features
7. Generate SHAP explanations
8. Save model and outputs

### Expected Performance

- **Target AUC:** ≥ 0.80
- **Clinical-only (IDRS baseline):** ~0.66
- **Genetic + Clinical (this model):** ~0.82-0.88

## Methodology for Your Paper

### Data Simulation Section

```
Due to the unavailability of individual-level genotype data for the Indian 
population, we adopted a simulation-based approach that leverages population-level 
GWAS summary statistics to generate synthetic individual genotypes while preserving 
real-world genetic architecture.

Genotype Simulation:
- Individual genotypes were simulated for 200 high-confidence SNPs (p < 1×10⁻⁵) 
  from the DIAMANTE South Asian T2D GWAS study
- Genotypes were sampled according to Hardy-Weinberg equilibrium using population 
  allele frequencies
- Polygenic risk scores (PRS) were calculated using validated effect sizes 
  (beta coefficients) from GWAS

Clinical Feature Integration:
- Traditional IDRS risk factors were simulated with realistic distributions 
  correlated with genetic risk (ρ ≈ 0.4)
- Features: age, BMI, waist circumference, family history, physical activity

Phenotype Assignment:
- T2D status assigned based on combined genetic (60%) and environmental (40%) risk
- Final dataset: 5,000 individuals (25% prevalence)

This simulation strategy enables methodological validation of our integrative 
approach while maintaining biological plausibility through use of real 
population-level genetic effect sizes.
```

### Limitations Section

```
This study uses simulated individual-level data generated from population-level 
GWAS summary statistics. While effect sizes and allele frequencies reflect real 
genetic associations from the DIAMANTE study, the individual genotypes and 
phenotypes are synthetic. Clinical validation on real patient cohorts with 
measured genotypes and clinical outcomes is required before deployment. The 
framework presented here is ready for integration with real individual-level 
genomic and clinical data.
```

## Key Advantages of This Approach

1. **Biologically grounded:** Uses real GWAS effect sizes
2. **Demonstrates integration:** Shows how to combine genetic + clinical features
3. **Interpretable:** SHAP/LIME explanations show feature importance
4. **Extensible:** Framework ready for real data
5. **Reproducible:** Seed-controlled simulation

## Questions?

This approach is standard for methodology papers when real individual data is unavailable. Many computational genomics papers use simulated data to demonstrate novel methods before validation on real cohorts.

---

**Author:** Generated for T2D-genome-risk-prediction project  
**Date:** December 2025
