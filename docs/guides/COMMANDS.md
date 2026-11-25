# 🚀 Command Reference: Complete Project Workflow

## Quick Commands

### 1️⃣ Preprocess GWAS Data (One-time)
```bash
python example_usage.py
```
**Output:** `processed_data/cleaned_gwas_summary_stats.csv` (1,113 SNPs)

---

### 2️⃣ Generate Simulated Dataset & Train Model
```bash
python scripts/simulate_and_train.py
```
**Output:**
- `processed_data/simulated_t2d_dataset.csv` (5,000 individuals)
- `models/best_model_ml.pkl` (trained XGBoost)
- `models/scaler.pkl` (feature scaler)
- `processed_data/shap_summary.png` (interpretability)

**Runtime:** ~30 seconds

---

### 3️⃣ Evaluate Model Performance (Comprehensive)
```bash
python scripts/evaluate_and_improve.py
```
**Output:**
- `processed_data/model_evaluation_results.csv` (all metrics)
- `processed_data/model_comparison.png` (visual comparison)
- `processed_data/roc_curves.png` (ROC curves)
- `processed_data/confusion_matrices.png` (classification results)
- `processed_data/evaluation_report.txt` (detailed report)

**Runtime:** ~2-3 minutes

---

## View Results

### Open Evaluation Report
```bash
# Windows
notepad processed_data/evaluation_report.txt

# Or in VS Code
code processed_data/evaluation_report.txt
```

### View Visualizations
```bash
# Windows
start processed_data/model_comparison.png
start processed_data/roc_curves.png
start processed_data/confusion_matrices.png
start processed_data/shap_summary.png

# Or open in VS Code
code processed_data/model_comparison.png
```

### View Metrics Table
```bash
python -c "import pandas as pd; print(pd.read_csv('processed_data/model_evaluation_results.csv'))"
```

---

## Use Trained Model

### Load and Predict (Python)
```python
import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load('models/best_model_ml.pkl')
scaler = joblib.load('models/scaler.pkl')

# New patient data
patient = pd.DataFrame({
    'age': [52],
    'bmi': [31.5],
    'waist_cm': [102],
    'family_history': [1],
    'physical_activity': [0],
    'PRS': [-1.2]
})

# Predict
X_scaled = scaler.transform(patient)
risk_prob = model.predict_proba(X_scaled)[0, 1]
print(f"T2D Risk: {risk_prob:.1%}")
```

### Batch Predictions
```bash
python -c "
import joblib, pandas as pd
model = joblib.load('models/best_model_ml.pkl')
scaler = joblib.load('models/scaler.pkl')
data = pd.read_csv('processed_data/simulated_t2d_dataset.csv')
X = data[['age', 'bmi', 'waist_cm', 'family_history', 'physical_activity', 'PRS']].head(10)
probs = model.predict_proba(scaler.transform(X))[:, 1]
print('Risk scores:', probs)
"
```

---

## Project Documentation

### Read Guides
```bash
# Comprehensive evaluation guide
code EVALUATION_GUIDE.md

# Quick summary
code QUICK_SUMMARY.md

# Simulation methodology
code README_SIMULATION.md

# GWAS preprocessing
code PREPROCESSING_README.md
```

---

## File Organization

```
T2D-genome-risk-prediction/
├── Dataset/
│   └── DIAMANTE-SAS.sumstat.csv       # Raw GWAS data
│
├── processed_data/
│   ├── cleaned_gwas_summary_stats.csv  # Filtered GWAS (1,113 SNPs)
│   ├── prs_weights.txt                 # PRS weight file
│   ├── simulated_t2d_dataset.csv       # 5,000 simulated individuals
│   ├── model_evaluation_results.csv    # Full metrics table
│   ├── evaluation_report.txt           # Detailed analysis
│   ├── model_comparison.png            # Model comparison plot
│   ├── roc_curves.png                  # ROC curves
│   ├── confusion_matrices.png          # Confusion matrices
│   └── shap_summary.png                # Feature importance
│
├── models/
│   ├── best_model_ml.pkl               # Trained XGBoost
│   └── scaler.pkl                      # Feature scaler
│
├── scripts/
│   ├── simulate_and_train.py           # Main simulation & training
│   └── evaluate_and_improve.py         # Comprehensive evaluation
│
└── Documentation/
    ├── EVALUATION_GUIDE.md             # How to evaluate models
    ├── QUICK_SUMMARY.md                # Quick reference
    ├── README_SIMULATION.md            # Simulation methodology
    └── COMMANDS.md                     # This file
```

---

## Troubleshooting

### Error: Module not found
```bash
pip install -r requirements.txt
```

### Error: File not found
```bash
# Make sure you're in project root
cd D:\Code_wid_pablo\Genome_T2D\T2D-genome-risk-prediction

# Check current directory
pwd

# List files
ls
```

### Error: GWAS data not processed
```bash
# Run preprocessing first
python example_usage.py
```

### Error: No simulated data
```bash
# Generate dataset first
python scripts/simulate_and_train.py
```

---

## Full Pipeline (Start to Finish)

```bash
# Navigate to project
cd D:\Code_wid_pablo\Genome_T2D\T2D-genome-risk-prediction

# Install dependencies (one-time)
pip install -r requirements.txt

# Step 1: Preprocess GWAS data
python example_usage.py

# Step 2: Generate simulated data and train model
python scripts/simulate_and_train.py

# Step 3: Comprehensive evaluation
python scripts/evaluate_and_improve.py

# Step 4: View results
start processed_data/model_comparison.png
start processed_data/roc_curves.png
notepad processed_data/evaluation_report.txt
```

**Total runtime:** ~3-4 minutes

---

## For Presentation/Demo

```bash
# 1. Show preprocessing results
python -c "import pandas as pd; df=pd.read_csv('processed_data/cleaned_gwas_summary_stats.csv'); print(f'Filtered SNPs: {len(df)}'); print(f'Chromosomes: {df.chr.nunique()}'); print(f'Mean p-value: {df.pval.mean():.2e}')"

# 2. Show simulated dataset stats
python -c "import pandas as pd; df=pd.read_csv('processed_data/simulated_t2d_dataset.csv'); print(f'Individuals: {len(df)}'); print(f'T2D cases: {df.T2D_status.sum()} ({df.T2D_status.mean():.1%})'); print(f'Mean PRS: {df.PRS.mean():.2f}')"

# 3. Show model performance
python -c "import pandas as pd; df=pd.read_csv('processed_data/model_evaluation_results.csv'); best=df.nlargest(1, 'AUC-ROC').iloc[0]; print(f'Best Model: {best.Model}'); print(f'Feature Set: {best[\"Feature Set\"]}'); print(f'AUC-ROC: {best[\"AUC-ROC\"]:.4f}'); print(f'Accuracy: {best.Accuracy:.1%}')"

# 4. Open all visualizations
start processed_data/model_comparison.png
start processed_data/roc_curves.png
start processed_data/confusion_matrices.png
start processed_data/shap_summary.png
```

---

## Modify Parameters

### Change Sample Size
Edit `scripts/simulate_and_train.py` line ~166:
```python
# Current: 5,000 individuals
gen_df, top_snps = simulate_genotypes(gwas_df, n_individuals=5000, top_n_snps=200)

# Change to 10,000:
gen_df, top_snps = simulate_genotypes(gwas_df, n_individuals=10000, top_n_snps=200)
```

### Change Number of SNPs
```python
# Current: Top 200 SNPs
gen_df, top_snps = simulate_genotypes(gwas_df, n_individuals=5000, top_n_snps=200)

# Change to 500:
gen_df, top_snps = simulate_genotypes(gwas_df, n_individuals=5000, top_n_snps=500)
```

### Change T2D Prevalence
Edit line ~170:
```python
# Current: 25%
phenotype, probs = assign_phenotype(prs, clinical, prevalence=0.25)

# Change to 15%:
phenotype, probs = assign_phenotype(prs, clinical, prevalence=0.15)
```

---

## Export for Paper

### Generate LaTeX Table
```bash
python -c "import pandas as pd; df=pd.read_csv('processed_data/model_evaluation_results.csv'); df_xgb=df[df['Model']=='XGBoost'][['Feature Set','AUC-ROC','Accuracy','Precision','Recall']]; print(df_xgb.to_latex(index=False))"
```

### Generate CSV for Excel
```bash
python -c "import pandas as pd; df=pd.read_csv('processed_data/model_evaluation_results.csv'); df[['Model','Feature Set','AUC-ROC','Accuracy','Precision','Recall','F1-Score']].to_csv('paper_results.csv', index=False)"
```

---

## Clean Up / Reset

### Remove Generated Files (Start Fresh)
```bash
# Remove processed data (keeps raw data)
Remove-Item processed_data/* -Exclude cleaned_gwas_summary_stats.csv,prs_weights.txt

# Remove models
Remove-Item models/*
```

### Full Reset (WARNING: Deletes everything)
```bash
# This removes all processed data and models
Remove-Item processed_data/*
Remove-Item models/*

# Then re-run pipeline
python example_usage.py
python scripts/simulate_and_train.py
python scripts/evaluate_and_improve.py
```

---

## Help & Documentation

```bash
# View all README files
ls *.md

# Open main documentation
code README_SIMULATION.md
code EVALUATION_GUIDE.md
code QUICK_SUMMARY.md

# Check Python script help
python scripts/simulate_and_train.py --help
python scripts/evaluate_and_improve.py --help
```

---

## Quick Tests

### Test Model Loading
```bash
python -c "import joblib; m=joblib.load('models/best_model_ml.pkl'); print('✓ Model loaded successfully')"
```

### Test Data Loading
```bash
python -c "import pandas as pd; df=pd.read_csv('processed_data/simulated_t2d_dataset.csv'); print(f'✓ Data loaded: {df.shape}')"
```

### Test Dependencies
```bash
python -c "import pandas, numpy, sklearn, xgboost, shap, matplotlib; print('✓ All packages installed')"
```

---

**Need help?** Check:
1. `EVALUATION_GUIDE.md` - Detailed evaluation guide
2. `QUICK_SUMMARY.md` - Quick reference
3. `processed_data/evaluation_report.txt` - Automated analysis

**Questions?** Review the generated evaluation report for specific recommendations!
