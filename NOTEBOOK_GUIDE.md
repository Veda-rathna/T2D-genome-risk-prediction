# 📓 Jupyter Notebook Demo Guide

## 🎯 Interactive Demo Notebook Created!

**File:** `T2D_Risk_Prediction_Demo.ipynb`

---

## 🚀 How to Use the Notebook

### Option 1: Open in VS Code (Recommended)

1. **Open the notebook:**
   ```bash
   code T2D_Risk_Prediction_Demo.ipynb
   ```

2. **Select Python kernel** (if prompted)
   - Click "Select Kernel" in top-right
   - Choose your Python 3.11 environment

3. **Run cells:**
   - Click "Run All" button, OR
   - Run cells one by one with Shift+Enter

---

### Option 2: Open in Jupyter Lab/Notebook

```bash
# Install Jupyter (if not installed)
pip install jupyter

# Launch Jupyter
jupyter notebook T2D_Risk_Prediction_Demo.ipynb
```

---

## 📋 Notebook Contents

### 🔹 **Section 1: Load Model**
- Imports libraries
- Loads trained XGBoost model
- Displays model information

### 🔹 **Section 2: Single Patient Prediction**
- **INTERACTIVE:** Modify patient values
- Get risk prediction
- Visual risk indicator

**Example:**
```python
patient_data = {
    'age': 52,
    'bmi': 31.5,
    'waist_cm': 102,
    'family_history': 1,      # 1=Yes, 0=No
    'physical_activity': 0,   # 1=Yes, 0=No
    'PRS': -1.2               # Polygenic Risk Score
}
```

### 🔹 **Section 3: Multiple Patient Comparison**
- Compare 4 patients side-by-side
- Visual comparison chart
- Risk categorization

### 🔹 **Section 4: Model Performance**
- Test on 500 samples
- ROC curve visualization
- Confusion matrix
- Performance metrics

### 🔹 **Section 5: Feature Importance**
- Which features matter most?
- PRS vs Clinical factors
- Visual ranking

### 🔹 **Section 6: Interactive Risk Calculator**
- Function to calculate any patient's risk
- Try different scenarios
- Instant feedback

### 🔹 **Section 7: Scenario Comparison**
- See how lifestyle changes affect risk
- Compare baseline vs interventions
- Visual impact analysis

---

## 🎮 Interactive Features

### 1. **Modify Patient Values**

Find cells marked with:
```python
# ========================================
# 👤 PATIENT INPUT - MODIFY THESE VALUES
# ========================================
```

Change the values and re-run the cell to see new predictions!

### 2. **Try Different Scenarios**

```python
# High Risk Patient
calculate_t2d_risk(
    age=60, bmi=35, waist_cm=110,
    family_history=1, physical_activity=0, prs=1.5
)

# Low Risk Patient
calculate_t2d_risk(
    age=30, bmi=22, waist_cm=75,
    family_history=0, physical_activity=1, prs=-1.2
)
```

### 3. **Compare Multiple Patients**

Add your own patients to the comparison table and visualize!

---

## 📊 What You'll See

### Visualizations Included:

1. **Risk Gauge** - Horizontal bar showing risk probability
2. **Multi-Patient Comparison** - Bar chart comparing multiple patients
3. **ROC Curve** - Model performance visualization
4. **Confusion Matrix** - Classification results heatmap
5. **Feature Importance** - Bar chart showing which factors matter most
6. **Scenario Analysis** - Impact of lifestyle changes

---

## 🎯 Perfect for Presentations!

### Demo Flow:

1. **Start:** Show model loading (Section 1)
2. **Single Prediction:** Enter a patient and show risk (Section 2)
3. **Comparison:** Show multiple patients side-by-side (Section 3)
4. **Performance:** Display ROC curve and metrics (Section 4)
5. **Insights:** Show feature importance (Section 5)
6. **Interactive:** Calculate risk live during presentation (Section 6)
7. **Impact:** Show scenario comparisons (Section 7)

---

## 💡 Tips for Showcase

### For Live Demo:

1. **Pre-run the notebook** once before presenting
2. **Clear outputs** before demo: Cell → All Output → Clear
3. **Run cells live** during presentation for impact
4. **Modify patient values** in real-time to show interactivity

### For Screenshots/Report:

1. **Run all cells** to generate all visualizations
2. **Export to PDF:**
   ```bash
   jupyter nbconvert --to pdf T2D_Risk_Prediction_Demo.ipynb
   ```
3. **Export to HTML:**
   ```bash
   jupyter nbconvert --to html T2D_Risk_Prediction_Demo.ipynb
   ```

---

## 🎨 Customization Ideas

### Change Risk Thresholds:

```python
if risk_probability >= 0.75:      # Very High
    category = "Very High Risk"
elif risk_probability >= 0.50:    # High
    category = "High Risk"
elif risk_probability >= 0.25:    # Moderate
    category = "Moderate Risk"
else:                              # Low
    category = "Low Risk"
```

### Add Your Own Patients:

```python
my_patient = {
    'age': 45,
    'bmi': 28.5,
    'waist_cm': 95,
    'family_history': 0,
    'physical_activity': 1,
    'PRS': 0.2
}
```

### Create New Scenarios:

```python
scenarios = pd.DataFrame({
    'Scenario': ['Before Diet', 'After Diet', 'With Exercise'],
    'age': [50, 50, 50],
    'bmi': [32, 28, 28],
    # ... add more
})
```

---

## 🐛 Troubleshooting

### Error: Kernel not found
```bash
# Install ipykernel
pip install ipykernel

# Register kernel
python -m ipykernel install --user --name python3
```

### Error: Module not found
```bash
# Install required packages
pip install jupyter matplotlib seaborn scikit-learn xgboost joblib pandas numpy
```

### Error: Model file not found
```bash
# Make sure you're in project root
cd D:\Code_wid_pablo\Genome_T2D\T2D-genome-risk-prediction

# Check if model exists
ls models/
```

### Cell won't run
- Click "Restart Kernel" button
- Run cells from top to bottom in order
- Make sure previous cells completed successfully

---

## 📝 Sample Presentation Script

### Slide 1: Introduction
*"I've developed a machine learning model that integrates genetic and clinical factors for T2D risk prediction."*

**[Run Section 1 - Load Model]**

### Slide 2: Single Patient Demo
*"Let me show you how it works with a real example. Here's a 52-year-old patient with elevated BMI..."*

**[Modify values in Section 2, Run]**

*"The model predicts a 68% probability of T2D - categorized as HIGH RISK."*

### Slide 3: Comparison
*"We can compare multiple patients simultaneously..."*

**[Run Section 3]**

*"Notice how Patient B has the highest risk due to age, BMI, and positive family history."*

### Slide 4: Performance
*"How good is the model? Let me show you the ROC curve..."*

**[Run Section 4]**

*"AUC of 0.999 indicates excellent discrimination. The confusion matrix shows very few misclassifications."*

### Slide 5: Feature Importance
*"Which factors matter most?"*

**[Run Section 5]**

*"PRS is the strongest predictor, followed by BMI and waist circumference."*

### Slide 6: Interactive Calculator
*"Let me demonstrate the interactive calculator..."*

**[Modify values in Section 6, Run live]**

*"You can adjust any parameter and instantly see the updated risk."*

### Slide 7: Scenario Analysis
*"Finally, let's see how lifestyle changes impact risk..."*

**[Run Section 7]**

*"Adding exercise reduces risk by 15%. Weight loss has even greater impact."*

---

## 🎓 Academic Value

This notebook demonstrates:
- ✅ End-to-end ML pipeline
- ✅ Model evaluation and validation
- ✅ Interactive prediction system
- ✅ Feature importance analysis
- ✅ Scenario-based analysis
- ✅ Publication-quality visualizations

Perfect for:
- Project presentations
- Thesis defense
- Paper supplementary materials
- Portfolio showcase
- Teaching demonstrations

---

## 📤 Export Options

### Save Visualizations:
```python
# Add to any plotting cell
plt.savefig('my_plot.png', dpi=300, bbox_inches='tight')
```

### Export Full Notebook:
```bash
# To PDF (requires LaTeX)
jupyter nbconvert --to pdf T2D_Risk_Prediction_Demo.ipynb

# To HTML (easier)
jupyter nbconvert --to html T2D_Risk_Prediction_Demo.ipynb

# To Python script
jupyter nbconvert --to python T2D_Risk_Prediction_Demo.ipynb
```

---

## 🚀 Quick Start Commands

```bash
# Open in VS Code
code T2D_Risk_Prediction_Demo.ipynb

# Or open in Jupyter
jupyter notebook T2D_Risk_Prediction_Demo.ipynb

# Or open in Jupyter Lab
jupyter lab T2D_Risk_Prediction_Demo.ipynb
```

---

## ✨ Pro Tips

1. **Run "Run All"** first to generate all outputs
2. **Save frequently** (Ctrl+S)
3. **Clear outputs** before sharing to reduce file size
4. **Add markdown cells** for your own notes
5. **Duplicate cells** to try multiple variations
6. **Use comments** to explain custom modifications

---

**Your interactive demo is ready! Open `T2D_Risk_Prediction_Demo.ipynb` and start showcasing your work!** 🎉
