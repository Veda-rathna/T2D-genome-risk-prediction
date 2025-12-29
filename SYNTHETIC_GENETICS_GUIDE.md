# 🧬 Synthetic Genetics Generation Guide

## Overview

This guide explains the methodology used to generate **synthetic genetic data** for the PIMA Indians Diabetes Dataset. The original PIMA dataset contains only clinical features (age, BMI, glucose, etc.) but lacks genetic information. Since genetic data (especially Polygenic Risk Scores or PRS) is a critical component of modern T2D risk prediction, we created synthetic genotypes to enable complete model testing.

**⚠️ IMPORTANT: The genetic data is SYNTHETIC (simulated) and NOT real genotypes. It is created for demonstration and educational purposes only.**

---

## Table of Contents

1. [Why Synthetic Genetics?](#why-synthetic-genetics)
2. [Methodology Overview](#methodology-overview)
3. [Step-by-Step Process](#step-by-step-process)
4. [Validation and Quality Checks](#validation-and-quality-checks)
5. [Limitations and Caveats](#limitations-and-caveats)
6. [Usage Instructions](#usage-instructions)
7. [Expected vs Actual Performance](#expected-vs-actual-performance)

---

## Why Synthetic Genetics?

### The Problem

The PIMA Indians Diabetes Dataset is a classic machine learning dataset that includes:
- ✅ **Clinical features**: Age, BMI, glucose, blood pressure, etc.
- ❌ **Genetic features**: No genotypes, no PRS, no SNPs

### Why We Need Genetics

Modern T2D risk prediction models incorporate genetic information because:
1. **Genetic predisposition** accounts for ~30-40% of T2D risk
2. **Polygenic Risk Scores (PRS)** significantly improve prediction accuracy
3. **Specific genetic variants** (SNPs) are strongly associated with T2D
4. Our model requires 6 features: `age, bmi, waist_cm, family_history, physical_activity, PRS`

### Options Considered

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **Real genotyped dataset** (UK Biobank, dbGaP) | Real data, high quality | Requires institutional access, expensive (~£5K), months to obtain | ❌ Not feasible for quick testing |
| **Set PRS = 0 for all patients** | Simple, immediate | Ignores genetic component, incomplete model testing | ⚠️ OK for initial testing |
| **Synthetic genetics** | Immediate, customizable, demonstrates concept | Not real data, limitations must be disclosed | ✅ **CHOSEN** |

---

## Methodology Overview

### High-Level Approach

We generate synthetic genotypes that:
1. ✅ Follow **Hardy-Weinberg equilibrium** (realistic population genetics)
2. ✅ Use **real T2D-associated SNPs** from DIAMANTE South Asian GWAS
3. ✅ Create **correlation with diabetes status** (mimics biological reality)
4. ✅ Calculate **PRS** from weighted sum of genotypes
5. ✅ Maintain **realistic distributions** (mean, variance, range)

### Key Principles

1. **Biological plausibility**: Genotypes follow population genetics principles
2. **Effect correlation**: Higher-risk genotypes are more common in diabetic patients
3. **Transparency**: Clearly labeled as SYNTHETIC in all outputs
4. **Reproducibility**: Seeded random generation for consistent results

---

## Step-by-Step Process

### Step 1: Load Real SNP Data

```python
# Load top 50 T2D-associated SNPs from DIAMANTE study
gwas = pd.read_csv('processed_data/cleaned_gwas_summary_stats.csv')
snps = gwas.nsmallest(50, 'pval')  # Top 50 by p-value
```

**Source**: DIAMANTE (DIAbetes Meta-ANalysis of Trans-Ethnic association studies)
- South Asian population GWAS
- ~120,000 individuals
- Genome-wide significant SNPs (p < 5×10⁻⁸)

### Step 2: Hardy-Weinberg Equilibrium

For each SNP, calculate genotype probabilities using Hardy-Weinberg equilibrium:

```
Given: Effect Allele Frequency (EAF) = p

Genotype probabilities:
- 0 copies (AA): p₀ = (1 - p)²
- 1 copy  (Aa): p₁ = 2 × p × (1 - p)
- 2 copies (aa): p₂ = p²

where a = effect allele, A = other allele
```

**Example**:
- SNP: rs7903146 (TCF7L2 gene)
- EAF = 0.30 (30% frequency of risk allele)
- P(0 copies) = 0.49
- P(1 copy) = 0.42
- P(2 copies) = 0.09

### Step 3: Effect-Correlated Adjustment

To create correlation with diabetes status, we adjust allele frequencies based on:
1. Patient's diabetes status (0 = healthy, 1 = diabetic)
2. SNP's effect size (beta coefficient)

```python
adjustment = 0.0
if diabetes_status == 1 and beta > 0:  # Diabetic + risk allele
    adjustment = +min(0.2, abs(beta) * 2)  # Increase risk allele frequency
elif diabetes_status == 0 and beta > 0:  # Healthy + risk allele
    adjustment = -min(0.2, abs(beta) * 2)  # Decrease risk allele frequency

adjusted_eaf = np.clip(eaf + adjustment, 0.1, 0.9)
```

**Rationale**:
- Diabetic patients → higher frequency of risk alleles
- Healthy patients → lower frequency of risk alleles
- Maximum adjustment ±20% to maintain realism
- Effect size determines magnitude of adjustment

### Step 4: Genotype Sampling

For each patient and SNP, sample genotype from adjusted probabilities:

```python
# Recalculate HWE probabilities with adjusted EAF
adj_p_0 = (1 - adjusted_eaf) ** 2
adj_p_1 = 2 * adjusted_eaf * (1 - adjusted_eaf)
adj_p_2 = adjusted_eaf ** 2

# Normalize to sum to 1
total = adj_p_0 + adj_p_1 + adj_p_2
adj_p_0, adj_p_1, adj_p_2 = adj_p_0/total, adj_p_1/total, adj_p_2/total

# Sample genotype
genotype = np.random.choice([0, 1, 2], p=[adj_p_0, adj_p_1, adj_p_2])
```

### Step 5: Calculate PRS

Polygenic Risk Score (PRS) is calculated as weighted sum of genotypes:

```python
PRS = Σ(genotype × beta)

where:
- genotype ∈ {0, 1, 2} for each SNP
- beta = effect size from GWAS (log odds ratio)
```

**Standardization**:
```python
PRS_standardized = (PRS - mean(PRS)) / std(PRS)
```

This ensures:
- Mean PRS = 0
- Standard deviation = 1
- Typical range: -3 to +3

### Step 6: Quality Control

Verify the generated data:
1. **Genotype distribution**: Mean ~0.9-1.1, variance ~0.4-0.5
2. **PRS distribution**: Mean ≈ 0, std ≈ 1
3. **PRS-diabetes correlation**: 0.85-0.95 (strong, as designed)
4. **Mean PRS difference**: Diabetic > Healthy by 1.5-2.0 SD

---

## Validation and Quality Checks

### Generated Dataset Statistics

**From actual run:**
```
✅ 757 patients (cleaned from 768 original)
✅ 50 T2D-associated SNPs from DIAMANTE study
✅ PRS range: -1.86 to +2.39
✅ PRS (Diabetic):     +1.22 ± 0.49
✅ PRS (Non-diabetic): -0.66 ± 0.40
✅ PRS difference:     1.88 (Cohen's d = very large effect)
✅ PRS correlation:    0.90 (very strong)
```

### Quality Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| **Genotype mean** | 0.8-1.2 | 0.94 | ✅ |
| **Genotype variance** | 0.4-0.6 | 0.45 | ✅ |
| **PRS mean** | ~0.0 | 0.00 | ✅ |
| **PRS std** | ~1.0 | 1.00 | ✅ |
| **PRS correlation** | 0.8-0.95 | 0.90 | ✅ |
| **Mean PRS difference** | 1.5-2.5 | 1.88 | ✅ |

### Feature Correlations with T2D

From the enhanced dataset:
```
PRS:                    +0.904  ⭐ (Strongest - by design)
Glucose:                +0.468
BMI:                    +0.306
Age:                    +0.238
DiabetesPedigreeFunction: +0.173
rs7903146 (top SNP):    +0.850
```

**Interpretation**:
- PRS is the **strongest predictor** (as designed)
- Clinical features still contribute (realistic)
- Individual SNPs show moderate-to-strong correlations
- Pattern mimics real-world genetics

---

## Limitations and Caveats

### ⚠️ Critical Limitations

1. **NOT REAL GENOTYPES**
   - Synthetic data generated algorithmically
   - Cannot replace real patient genotyping
   - For demonstration and education ONLY

2. **Correlation by Design**
   - PRS-diabetes correlation (0.90) is **artificially high**
   - Real-world PRS correlations: 0.2-0.4 typically
   - Creates overly optimistic performance estimates

3. **Population Mismatch**
   - SNPs from South Asian GWAS (DIAMANTE)
   - PIMA patients are Native American (Pima Indians)
   - Real effect sizes would differ by ancestry

4. **Simplified Genetics**
   - Assumes additive genetic effects (0, 1, 2 copies)
   - Ignores dominance, epistasis, gene-environment interactions
   - No linkage disequilibrium structure

5. **Selection Bias**
   - Used "top 50 SNPs" by p-value
   - Real PRS use hundreds to millions of variants
   - Modern PRS methods (PRS-CS, LDpred) are more sophisticated

### ✅ What It's Good For

1. **Demonstration**: Shows how genetic data improves prediction
2. **Education**: Teaches PRS concepts and methodology
3. **Development**: Allows testing of full 6-feature model
4. **Comparison**: Clinical-only vs Clinical+Genetic performance
5. **Workflow**: Validates data processing pipeline

### ❌ What It's NOT Good For

1. **Clinical decisions**: Do NOT use for patient care
2. **Publication**: Not suitable for peer-reviewed research
3. **Policy**: Cannot inform healthcare policy
4. **Ancestry inference**: Not real ancestral genetics
5. **Causality**: Does not prove genetic causation

---

## Usage Instructions

### Running the Script

```bash
# Generate synthetic genetics for PIMA dataset
python scripts/add_genetics_to_pima.py
```

**Output files** (saved to `Dataset/` folder):

1. **`pima_with_genetics.csv`** (757 × 21 columns)
   - Clinical features (Age, BMI, Glucose, etc.)
   - PRS (standardized polygenic risk score)
   - Top 10 SNPs (rs identifiers)
   - Outcome (diabetes status: 0/1)

2. **`pima_genotypes.csv`** (757 × 50 columns)
   - Full genotype matrix
   - All 50 SNPs
   - Values: 0, 1, 2 (copies of effect allele)

3. **`pima_snp_info.csv`** (50 × 6 columns)
   - SNP metadata
   - rsID, alleles, EAF, beta, p-value

### Using in Jupyter Notebook

```python
# Load enhanced PIMA dataset
import pandas as pd

# Option 1: Clinical + PRS + Top 10 SNPs
pima = pd.read_csv('Dataset/pima_with_genetics.csv')

# Option 2: Full genotype matrix
genotypes = pd.read_csv('Dataset/pima_genotypes.csv')
snp_info = pd.read_csv('Dataset/pima_snp_info.csv')

# Extract features for model
X = pima[['Age', 'BMI', 'waist_cm', 'family_history', 
          'physical_activity', 'PRS']]
y = pima['Outcome']
```

### Testing the Model

Run the enhanced PIMA testing notebook:

```bash
# Open notebook
jupyter notebook PIMA_Dataset_Testing.ipynb

# Or in VS Code: Open PIMA_Dataset_Testing.ipynb
```

The notebook will:
1. Load enhanced dataset with genetics
2. Compare Clinical-only vs Clinical+Genetic performance
3. Analyze individual SNP contributions
4. Generate comprehensive visualizations

---

## Expected vs Actual Performance

### Expected Performance

**With synthetic genetics** (as generated):
- **AUC-ROC**: 0.85-0.95 (very high due to strong PRS correlation)
- **Accuracy**: 80-90%
- **PRS contribution**: Major predictor (correlation 0.90)
- **Improvement over clinical-only**: +15-25% AUC

**Interpretation**:
- Performance will be **INFLATED** compared to real genetics
- Real-world PRS typically adds +5-10% to AUC
- Useful for demonstration, not realistic estimates

### Real-World Performance (for comparison)

**With REAL genetics** (literature estimates):
- **AUC-ROC**: 0.70-0.80
- **PRS contribution**: Moderate (correlation 0.2-0.4)
- **Improvement over clinical-only**: +5-10% AUC
- **Example**: Diabetes UK study (500K patients, real PRS, AUC = 0.75)

### Model Performance Comparison

| Scenario | AUC | Interpretation |
|----------|-----|----------------|
| **Clinical Only** | 0.70-0.75 | Baseline with age, BMI, etc. |
| **Clinical + Synthetic PRS** | 0.85-0.95 | Inflated (strong artificial correlation) |
| **Clinical + Real PRS** | 0.75-0.80 | Realistic improvement |
| **Perfect Model** | 1.00 | Impossible (too many unknowns) |

---

## Best Practices

### When Using Synthetic Genetics

1. **✅ DO**:
   - Clearly label as "SYNTHETIC" in all outputs
   - Mention methodology in any presentation/report
   - Use for demonstration and education
   - Compare with clinical-only baseline
   - Acknowledge limitations explicitly

2. **❌ DON'T**:
   - Claim these are real genotypes
   - Use for clinical decision-making
   - Publish without extensive validation
   - Ignore the inflated performance
   - Extrapolate to other populations

### Moving to Real Genetics

To test your model on **real genetic data**:

1. **UK Biobank** (recommended)
   - 500,000+ participants with genotypes
   - Application process: https://www.ukbiobank.ac.uk
   - Cost: £2,500-5,000
   - Timeline: 3-6 months

2. **dbGaP** (NIH database)
   - Free for approved researchers
   - Multiple T2D studies with genotypes
   - Application: https://dbgap.ncbi.nlm.nih.gov
   - Timeline: 2-3 months

3. **All of Us** (U.S. cohort)
   - 400,000+ with whole genome sequencing
   - Free registration: https://www.researchallofus.org
   - Timeline: 1-2 months

---

## Technical Details

### Script Structure

**`scripts/add_genetics_to_pima.py`**:

```python
def load_pima_dataset():
    """Load and clean PIMA dataset"""
    
def load_t2d_snps(gwas_file, n_snps=50):
    """Load top T2D-associated SNPs from GWAS"""
    
def simulate_genotypes(n_samples, snps, diabetes_status, seed=42):
    """
    Generate synthetic genotypes with correlation to phenotype
    
    Uses Hardy-Weinberg equilibrium with effect-correlated adjustment
    """
    
def calculate_prs(genotypes, effect_sizes):
    """Calculate Polygenic Risk Score (weighted sum)"""
    
def main():
    """
    Main workflow:
    1. Load PIMA dataset
    2. Load T2D SNPs
    3. Generate synthetic genotypes
    4. Calculate PRS
    5. Save 3 output files
    """
```

### Random Seed

For **reproducibility**, the script uses a fixed seed:
```python
np.random.seed(42)
```

This ensures:
- Same genotypes generated each run
- Consistent PRS values
- Reproducible results for testing

To generate **different** synthetic genetics:
```python
np.random.seed(None)  # Use system time as seed
```

---

## Citation and References

### This Methodology

If using this synthetic genetics approach, cite:
```
Synthetic genetics generated using Hardy-Weinberg equilibrium 
with effect-correlated adjustment, based on top 50 T2D-associated 
SNPs from DIAMANTE South Asian GWAS (Mahajan et al. 2022).

⚠️ Note: Genotypes are SYNTHETIC for demonstration purposes.
```

### Related Publications

1. **DIAMANTE Consortium**:
   - Mahajan et al. (2022) "Multi-ancestry genetic study of type 2 diabetes highlights the power of diverse populations for discovery and translation"
   - Nature Genetics, 54, 560–572

2. **PRS Methodology**:
   - Khera et al. (2018) "Genome-wide polygenic scores for common diseases identify individuals with risk equivalent to monogenic mutations"
   - Nature Genetics, 50, 1219–1224

3. **Hardy-Weinberg Equilibrium**:
   - Hardy, G. H. (1908). "Mendelian proportions in a mixed population"
   - Science, 28 (706): 49–50

4. **Original PIMA Dataset**:
   - Smith et al. (1988) "Using the ADAP learning algorithm to forecast the onset of diabetes mellitus"
   - Proceedings of the Annual Symposium on Computer Application in Medical Care

---

## FAQs

### Q: Why not use real genetic data?

**A**: Real genotyped datasets with clinical phenotypes require:
- Institutional data access agreements
- Significant costs (£2,500-5,000 for UK Biobank)
- Months to obtain approval
- Ethical approvals for human subjects research

Synthetic genetics allow immediate testing while we pursue real data access.

### Q: How realistic are these genotypes?

**A**: The synthetic genotypes follow realistic population genetics principles (Hardy-Weinberg equilibrium) but have **artificially strong** correlation with diabetes (0.90 vs typical 0.2-0.4). This makes them useful for demonstration but not for accurate performance estimates.

### Q: Can I use this for my research paper?

**A**: **Not recommended** as primary evidence. Synthetic genetics should only be used for:
- Methods development
- Proof-of-concept demonstrations
- Educational purposes

For publication, you need **real genetic data**.

### Q: Will my model perform this well on real data?

**A**: **No**. The synthetic genetics create inflated performance (AUC ~0.90+). Real-world PRS typically improve AUC by +5-10%, not +20-30%.

### Q: Can I adjust the correlation strength?

**A**: Yes! In `scripts/add_genetics_to_pima.py`, modify the adjustment parameter:

```python
# Current (strong correlation):
adjustment = min(0.2, abs(beta) * 2)  # ±20% max

# Weaker correlation (more realistic):
adjustment = min(0.05, abs(beta) * 0.5)  # ±5% max
```

Lower adjustment → weaker correlation → more realistic performance.

---

## Conclusion

Synthetic genetics generation is a **powerful tool for demonstration and education** but has **significant limitations** for research and clinical applications. Always:

1. ✅ Label as SYNTHETIC
2. ✅ Acknowledge limitations
3. ✅ Use for learning and development
4. ❌ Don't use for clinical decisions
5. ❌ Don't publish without real data validation

For serious research and clinical applications, pursue access to **real genotyped datasets** (UK Biobank, dbGaP, All of Us).

---

## Contact and Support

For questions about this methodology:
1. Review this guide thoroughly
2. Check the script comments: `scripts/add_genetics_to_pima.py`
3. Examine the outputs: `Dataset/pima_with_genetics.csv`
4. Run the notebook: `PIMA_Dataset_Testing.ipynb`

**Remember**: This is a **demonstration tool**, not a replacement for real genetics!

---

**Last Updated**: 2024
**Version**: 1.0
**Status**: For educational and demonstration use only
