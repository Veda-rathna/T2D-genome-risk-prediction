# 🗃️ Real Datasets for T2D Risk Prediction Testing

## 📋 Table of Contents
1. [Quick Answer](#quick-answer)
2. [Public Datasets with Individual-Level Data](#public-datasets-with-individual-level-data)
3. [GWAS Summary Statistics (What You Have)](#gwas-summary-statistics-what-you-have)
4. [Clinical Datasets](#clinical-datasets)
5. [Synthetic but Realistic Datasets](#synthetic-but-realistic-datasets)
6. [How to Adapt Your Model](#how-to-adapt-your-model)
7. [Important Considerations](#important-considerations)

---

## 🎯 Quick Answer

**Unfortunately, there's NO publicly available dataset with BOTH:**
- ✅ Individual genotype data (1,113 SNPs)
- ✅ Clinical features (age, BMI, waist, etc.)
- ✅ T2D diagnosis labels

**Why?** Privacy laws (HIPAA, GDPR) prevent sharing individual genetic + health data publicly.

**However, you have several options:**

| Option | Data Available | Difficulty | Best For |
|--------|---------------|------------|----------|
| **1. UK Biobank** | ✅ Full data | 🔴 Hard (requires application) | Research validation |
| **2. PIMA Indians** | ✅ Clinical only | 🟢 Easy (public) | Clinical model testing |
| **3. NHANES** | ✅ Clinical only | 🟡 Medium (public survey) | Clinical features |
| **4. dbGaP** | ✅ Full data | 🔴 Hard (controlled access) | Research |
| **5. All of Us** | ✅ Full data | 🟡 Medium (registration) | Research |

---

## 🧬 Public Datasets with Individual-Level Data

### 1. UK Biobank ⭐ (Best Option)

**What it has:**
- ✅ 500,000+ participants with genotype data
- ✅ Clinical measurements (age, BMI, waist, blood tests)
- ✅ T2D diagnosis labels
- ✅ Longitudinal follow-up

**How to access:**
1. Visit: https://www.ukbiobank.ac.uk/
2. Register as a researcher (need institution affiliation)
3. Submit application describing your project
4. Wait 2-4 weeks for approval
5. Pay access fee (~£2,500-5,000 for researchers)

**Pros:**
- Gold standard for genetic epidemiology
- Exact data you need (genotypes + clinical + T2D)
- Well-documented, quality-controlled
- Many SNPs overlap with your DIAMANTE study

**Cons:**
- ❌ Not free (requires funding)
- ❌ Requires institutional affiliation
- ❌ Application process takes time
- ❌ Data use restrictions

**Code to extract your SNPs:**
```bash
# After getting UK Biobank data
plink2 \
  --bfile ukb_genotypes \
  --extract snp_list.txt \
  --make-bed \
  --out ukb_t2d_snps

# snp_list.txt contains your 1,113 SNPs from DIAMANTE
```

---

### 2. dbGaP (Database of Genotypes and Phenotypes)

**What it has:**
- ✅ Multiple T2D studies with genotypes + phenotypes
- ✅ Diverse populations (EUR, AFR, HIS, EAS, SAS)
- ✅ Free for researchers

**Key studies:**
- **ARIC** (Atherosclerosis Risk in Communities)
- **FHS** (Framingham Heart Study)
- **MESA** (Multi-Ethnic Study of Atherosclerosis)
- **WHI** (Women's Health Initiative)

**How to access:**
1. Visit: https://www.ncbi.nlm.nih.gov/gap/
2. Register for a dbGaP account
3. Complete data access training (1 hour)
4. Submit Data Access Request (DAR) for specific study
5. Wait 2-6 weeks for approval
6. Download data

**Search for T2D datasets:**
```
URL: https://www.ncbi.nlm.nih.gov/gap/advanced_search/
Search: "type 2 diabetes" + "genotype"
Filter: "Study with Individual Level Data"
```

**Pros:**
- ✅ Free for approved researchers
- ✅ High-quality data from major studies
- ✅ Multiple cohorts available

**Cons:**
- ❌ Requires institutional email and IRB approval
- ❌ Application process (2-6 weeks)
- ❌ Data use agreements and restrictions

---

### 3. All of Us Research Program

**What it has:**
- ✅ 400,000+ diverse participants
- ✅ Whole genome sequencing data
- ✅ Electronic health records (includes T2D diagnosis)
- ✅ Survey data (lifestyle, family history)
- ✅ Physical measurements

**How to access:**
1. Visit: https://www.researchallofus.org/
2. Register and complete training modules
3. Access data through Researcher Workbench (cloud-based)
4. Free for researchers

**Pros:**
- ✅ Free access for researchers
- ✅ Diverse population (underrepresented groups)
- ✅ Comprehensive data (genetics + EHR + surveys)
- ✅ Cloud-based analysis (no download needed)

**Cons:**
- ❌ Requires registration and training
- ❌ Data must stay in cloud environment
- ❌ Still growing (not all participants have genetic data yet)

**More info:**
- Website: https://www.researchallofus.org/
- Registration: https://workbench.researchallofus.org/

---

## 🏥 Clinical Datasets (No Genetics, but Good for Testing)

### 1. PIMA Indians Diabetes Dataset ⭐ (Easiest!)

**What it has:**
- ✅ 768 female patients (Pima Indian heritage)
- ✅ 8 clinical features
- ✅ Diabetes diagnosis (0/1)
- ✅ Publicly available (NO registration needed!)

**Features:**
- Pregnancies
- Glucose concentration
- Blood pressure
- Skin thickness
- Insulin
- BMI ✅ (You have this!)
- Diabetes pedigree function (similar to family history)
- Age ✅ (You have this!)

**How to access:**
```python
# Method 1: Direct download
import pandas as pd
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

# Method 2: UCI repository
# https://archive.ics.uci.edu/ml/datasets/diabetes
```

**How to use with your model:**
```python
# Map PIMA features to your model features
pima_data = pd.DataFrame({
    'age': df['Age'],
    'bmi': df['BMI'],
    'waist_cm': df['BMI'] * 3.2 + 50,  # Estimate from BMI
    'family_history': (df['DiabetesPedigreeFunction'] > 0.5).astype(int),
    'physical_activity': 0,  # Not available, assume sedentary
    'PRS': 0  # Not available, use neutral score
})

# Make predictions
X_scaled = scaler.transform(pima_data)
predictions = model.predict_proba(X_scaled)[:, 1]
```

**Pros:**
- ✅ Immediate access (no registration!)
- ✅ Easy to use
- ✅ Real patient data
- ✅ Has BMI and Age (2 of your features)

**Cons:**
- ❌ No genetic data (can't use your PRS)
- ❌ Small sample size (768 patients)
- ❌ Single ethnic group (limited generalization)
- ❌ Older dataset (1988)

---

### 2. NHANES (National Health and Nutrition Examination Survey)

**What it has:**
- ✅ 50,000+ participants across multiple cycles
- ✅ Clinical measurements (BMI, waist, blood pressure, etc.)
- ✅ Laboratory data (glucose, HbA1c, insulin)
- ✅ Diabetes diagnosis
- ✅ Publicly available!

**How to access:**
1. Visit: https://wwwn.cdc.gov/nchs/nhanes/Default.aspx
2. Select survey cycle (e.g., 2017-2020)
3. Download relevant datasets:
   - Demographics (age, sex, race)
   - Body Measures (BMI, waist circumference)
   - Diabetes (diagnosis, glucose, HbA1c)
   - Physical Activity

**Example download:**
```python
import pandas as pd

# Demographics 2017-2018
demo = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT')

# Body Measures
bmx = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BMX_J.XPT')

# Diabetes
diq = pd.read_sas('https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DIQ_J.XPT')

# Merge on participant ID
merged = demo.merge(bmx, on='SEQN').merge(diq, on='SEQN')
```

**Map to your features:**
```python
nhanes_data = pd.DataFrame({
    'age': merged['RIDAGEYR'],  # Age in years
    'bmi': merged['BMXBMI'],    # BMI
    'waist_cm': merged['BMXWAIST'],  # Waist circumference
    'family_history': (merged['DIQ175A'] == 1).astype(int),  # Family history
    'physical_activity': (merged['PAQ605'] >= 5).astype(int),  # Physical activity
    'PRS': 0  # Not available
})
```

**Pros:**
- ✅ Free and public
- ✅ Large sample size
- ✅ High-quality measurements
- ✅ Has BMI, waist, age (4 of your 6 features!)
- ✅ Recent data (updated regularly)

**Cons:**
- ❌ No genetic data
- ❌ Complex survey design (needs weighting)
- ❌ Multiple files to merge

**Tutorial:**
- https://wwwn.cdc.gov/nchs/nhanes/tutorials/default.aspx

---

### 3. Kaggle Diabetes Datasets

**Several options on Kaggle:**

#### A. Diabetes Health Indicators Dataset
- **Link:** https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
- **Size:** 250,000+ participants from BRFSS
- **Features:** BMI, age, physical activity, etc.
- **Download:** Direct (no application)

#### B. Hospital Diabetes Dataset
- **Link:** https://www.kaggle.com/datasets/brandao/diabetes
- **Size:** 100,000+ hospital records
- **Features:** Age, medications, procedures
- **Download:** Direct

**How to use:**
```python
# After downloading from Kaggle
df = pd.read_csv('diabetes.csv')

# Map features and test your model
# (Similar to PIMA example above)
```

**Pros:**
- ✅ Immediate download
- ✅ Various sizes and populations
- ✅ Easy to explore

**Cons:**
- ❌ No genetic data
- ❌ Variable data quality
- ❌ Often pre-cleaned (less realistic)

---

## 🧪 Synthetic but Realistic Datasets

### 1. Synthea (Synthetic Patient Generator)

**What it is:**
- Generates synthetic but realistic patient records
- Can create T2D patients with clinical features
- No privacy concerns (completely synthetic)

**How to generate:**
```bash
# Install Synthea
git clone https://github.com/synthetichealth/synthea.git
cd synthea

# Generate 1000 patients
./run_synthea -p 1000

# Output: CSV files with demographics, conditions, medications, etc.
```

**Extract T2D patients:**
```python
import pandas as pd

# Load conditions
conditions = pd.read_csv('output/csv/conditions.csv')

# Filter T2D patients
t2d_patients = conditions[conditions['DESCRIPTION'].str.contains('Diabetes mellitus type 2')]

# Load patient demographics
patients = pd.read_csv('output/csv/patients.csv')
observations = pd.read_csv('output/csv/observations.csv')

# Merge and extract features
# (BMI, age, etc.)
```

**Pros:**
- ✅ No privacy concerns
- ✅ Unlimited data generation
- ✅ Realistic correlations between features
- ✅ Good for testing pipelines

**Cons:**
- ❌ Still synthetic (not real patients)
- ❌ No real genetic data
- ❌ Limited validation value

**Website:** https://synthetichealth.github.io/synthea/

---

## 🔧 How to Adapt Your Model

### Scenario 1: Dataset WITHOUT Genetic Data (PIMA, NHANES)

**Option A: Use Clinical Features Only**
```python
# Train model without PRS
feature_cols_clinical = ['age', 'bmi', 'waist_cm', 'family_history', 'physical_activity']

X_clinical = df[feature_cols_clinical]
y = df['T2D_status']

# Retrain
scaler_clinical = StandardScaler()
X_scaled = scaler_clinical.fit_transform(X_clinical)

model_clinical = XGBClassifier(...)
model_clinical.fit(X_scaled, y)
```

**Option B: Set PRS to Neutral (0)**
```python
# Add PRS column with neutral value
df['PRS'] = 0.0

# Use existing model (but PRS won't contribute)
X_scaled = scaler.transform(df[all_features])
predictions = model.predict_proba(X_scaled)[:, 1]
```

**Option C: Impute PRS from Ethnicity**
```python
# Use population-specific average PRS
prs_by_ethnicity = {
    'European': 0.1,
    'African': -0.2,
    'Asian': 0.05,
    'Hispanic': 0.0,
    'South Asian': 0.15  # DIAMANTE population
}

df['PRS'] = df['ethnicity'].map(prs_by_ethnicity)
```

---

### Scenario 2: Dataset WITH Genetic Data but Different SNPs

**Step 1: Find Overlapping SNPs**
```python
# Your SNPs from DIAMANTE
diamante_snps = pd.read_csv('processed_data/cleaned_gwas_summary_stats.csv')
your_snps = set(diamante_snps['SNP'])

# New dataset SNPs
new_data_snps = set(new_data.columns)

# Find overlap
overlap = your_snps.intersection(new_data_snps)
print(f"Overlapping SNPs: {len(overlap)} / {len(your_snps)}")
```

**Step 2: Recalculate PRS with Available SNPs**
```python
# Load weights
weights = pd.read_csv('processed_data/prs_weights.txt')
weights_overlap = weights[weights['SNP'].isin(overlap)]

# Calculate PRS
def compute_prs_partial(genotypes, weights_df):
    prs = 0
    for _, row in weights_df.iterrows():
        snp = row['SNP']
        if snp in genotypes.columns:
            # Dosage: 0, 1, or 2 copies of effect allele
            prs += genotypes[snp] * row['Beta']
    return prs

new_data['PRS'] = new_data.apply(lambda row: compute_prs_partial(row, weights_overlap), axis=1)

# Standardize
new_data['PRS'] = (new_data['PRS'] - new_data['PRS'].mean()) / new_data['PRS'].std()
```

**Step 3: Validate**
```python
# Check PRS distribution
print(f"PRS range: {new_data['PRS'].min():.2f} to {new_data['PRS'].max():.2f}")
print(f"PRS mean: {new_data['PRS'].mean():.2f} (should be ~0)")
print(f"PRS std: {new_data['PRS'].std():.2f} (should be ~1)")
```

---

## ⚠️ Important Considerations

### 1. **Population Ancestry**

Your model was trained on **South Asian** data (DIAMANTE).

**Effect sizes differ by ancestry:**
- SNP effect in South Asians: β = 0.15
- Same SNP in Europeans: β = 0.12
- Same SNP in Africans: β = 0.08

**Solutions:**
- ✅ Test on similar ancestry (South Asian cohorts)
- ✅ Retrain on new population
- ✅ Use trans-ethnic PRS (weighted by ancestry)
- ❌ Don't directly apply to very different populations

---

### 2. **Feature Availability**

Your model needs 6 features:

| Feature | Critical? | Alternatives |
|---------|-----------|-------------|
| **Age** | ✅ Critical | Use median if missing |
| **BMI** | ✅ Critical | Calculate from height/weight |
| **Waist** | 🟡 Important | Estimate: waist ≈ BMI × 3.2 + 50 |
| **Family History** | 🟡 Important | Set to 0 if unknown |
| **Physical Activity** | 🟢 Optional | Set to 0 (sedentary) if unknown |
| **PRS** | ⭐ Very Important | See scenarios above |

**Minimum viable features:** Age + BMI + PRS

---

### 3. **Model Retraining vs Transfer Learning**

#### Option A: Full Retraining (Recommended)
```python
# Use new real data to train from scratch
X_new = real_data[feature_cols]
y_new = real_data['T2D_status']

model_real = XGBClassifier(...)  # Same hyperparameters
model_real.fit(X_new, y_new)

# Evaluate
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model_real, X_new, y_new, cv=5, scoring='roc_auc')
print(f"Real data AUC: {scores.mean():.3f} ± {scores.std():.3f}")
```

#### Option B: Fine-tuning (If new data is small)
```python
# Start with simulated model weights
# Fine-tune on real data

# Load simulated model
base_model = joblib.load('models/realistic_model.pkl')

# Fine-tune on real data
# (XGBoost doesn't directly support this, so retrain with more regularization)
model_finetuned = XGBClassifier(
    **base_model.get_params(),
    learning_rate=0.01,  # Slower learning
    n_estimators=50      # Fewer additional trees
)
model_finetuned.fit(X_real, y_real, 
                    xgb_model=base_model.get_booster())  # Start from existing
```

---

### 4. **Evaluation Metrics**

On real data, expect **more realistic performance:**

| Metric | Simulated Data | Expected on Real Data |
|--------|---------------|----------------------|
| **AUC** | 0.995 | 0.70-0.85 |
| **Accuracy** | 97.5% | 75-85% |
| **Sensitivity** | 92.2% | 70-80% |
| **Specificity** | 99.9% | 75-85% |

**Why lower?** Real data has:
- Measurement errors
- Missing confounders
- Biological complexity
- Environmental factors

**This is normal and expected!**

---

### 5. **Ethical and Legal Requirements**

#### Before using real patient data:

- [ ] **IRB Approval** - Institutional Review Board approval for human subjects research
- [ ] **Data Use Agreement** - Sign agreements for controlled-access databases
- [ ] **HIPAA Compliance** - Ensure data is de-identified (if US data)
- [ ] **GDPR Compliance** - Follow regulations (if European data)
- [ ] **Informed Consent** - Verify patients consented to research use
- [ ] **Secure Storage** - Encrypted storage, limited access
- [ ] **No Re-identification** - Don't attempt to identify individuals

**Public datasets usually handle this**, but verify!

---

## 📚 Recommended Approach

### For Learning/Development (Now):
1. ✅ Continue with simulated data (what you have)
2. ✅ Test on PIMA Indians dataset (easy, public, no genetics)
3. ✅ Test on NHANES (more features, but complex)

### For Research/Publication:
1. Apply to UK Biobank (best option, gold standard)
2. Or apply to dbGaP studies (free, multiple cohorts)
3. Or use All of Us (growing, diverse)

### For Clinical Deployment:
1. Partner with hospital/clinic
2. Collect prospective data with consent
3. Validate on local population
4. Get regulatory approval (FDA for US, CE mark for EU)

---

## 🚀 Quick Start: Test on PIMA Dataset

Here's a complete script to test your model on PIMA data:

```python
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

# 1. Load PIMA dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
pima = pd.read_csv(url, names=columns)

print(f"Loaded {len(pima)} patients from PIMA dataset")
print(f"T2D prevalence: {pima['Outcome'].mean():.1%}")

# 2. Map to your model's features
pima_mapped = pd.DataFrame({
    'age': pima['Age'],
    'bmi': pima['BMI'],
    'waist_cm': pima['BMI'] * 3.2 + 50,  # Rough estimate from BMI
    'family_history': (pima['DiabetesPedigreeFunction'] > 0.5).astype(int),
    'physical_activity': 0,  # Not available, assume sedentary baseline
    'PRS': 0.0  # Not available, use neutral
})

# 3. Clean data (handle zeros that represent missing)
pima_mapped = pima_mapped[pima_mapped['bmi'] > 0]  # Remove invalid BMI
pima_mapped['age'] = pima_mapped['age'].clip(18, 100)  # Realistic age range

# 4. Load your model
model = joblib.load('models/realistic_model.pkl')
scaler = joblib.load('models/realistic_scaler.pkl')

# 5. Make predictions
X_scaled = scaler.transform(pima_mapped)
y_pred_proba = model.predict_proba(X_scaled)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

# 6. Evaluate (align indices after cleaning)
y_true = pima.loc[pima_mapped.index, 'Outcome'].values

auc = roc_auc_score(y_true, y_pred_proba)
print(f"\n{'='*60}")
print(f"PIMA Dataset Evaluation Results")
print(f"{'='*60}")
print(f"AUC-ROC: {auc:.3f}")
print(f"\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['No Diabetes', 'Diabetes']))

# 7. Show example predictions
print(f"\n{'='*60}")
print("Sample Predictions:")
print(f"{'='*60}")
sample = pima_mapped.head(10).copy()
sample['Predicted_Risk'] = y_pred_proba[:10]
sample['Actual'] = y_true[:10]
print(sample[['age', 'bmi', 'Predicted_Risk', 'Actual']])

print(f"\n✅ Testing complete!")
print(f"Note: Performance may be lower than simulated data (expected!)")
print(f"Reason: Missing genetic data (PRS=0) and estimated features")
```

**Save this as:** `scripts/test_on_pima.py`

**Run:**
```bash
python scripts/test_on_pima.py
```

---

## 📖 Additional Resources

### Databases:
- **UK Biobank:** https://www.ukbiobank.ac.uk/
- **dbGaP:** https://www.ncbi.nlm.nih.gov/gap/
- **All of Us:** https://www.researchallofus.org/
- **NHANES:** https://www.cdc.gov/nchs/nhanes/
- **PIMA:** https://archive.ics.uci.edu/ml/datasets/diabetes

### Tutorials:
- **UK Biobank Access:** https://biobank.ndph.ox.ac.uk/showcase/
- **dbGaP Tutorial:** https://www.youtube.com/watch?v=Z9V8vQsJc-o
- **NHANES Tutorial:** https://wwwn.cdc.gov/nchs/nhanes/tutorials/

### Ethics:
- **IRB Guidelines:** https://www.hhs.gov/ohrp/
- **HIPAA:** https://www.hhs.gov/hipaa/
- **GDPR:** https://gdpr.eu/

---

## ✅ Summary

**Immediate Options (No Application):**
1. ✅ **PIMA Indians Dataset** - Download and test now! (see script above)
2. ✅ **NHANES** - Public access, comprehensive
3. ✅ **Kaggle Diabetes Datasets** - Various options

**Best Options (Requires Application):**
1. ⭐ **UK Biobank** - Gold standard (2-4 weeks, costs money)
2. ⭐ **dbGaP** - Free, multiple studies (2-6 weeks)
3. ⭐ **All of Us** - Free, diverse (1-2 weeks)

**Recommended Next Step:**
- Start with PIMA dataset (test the script above)
- Meanwhile, apply to UK Biobank or dbGaP for proper validation
- Use real data findings to improve your model

---

**Questions?**
- See `REALISTIC_MODEL_QUICKSTART.md` for model details
- See `OVERFITTING_FIX_GUIDE.md` for training details
- Check dataset websites for specific access instructions
