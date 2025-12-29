# 🌈 Diverse Patient Dataset Reference

## Overview

The notebook now includes **12 diverse patients** representing all risk categories and demographic profiles.

---

## 📊 Patient Profiles

### 🟢 **Low Risk Patients (< 25%)**

#### **Patient 1 - Very Low Risk**
- Age: 28, BMI: 21.5, Waist: 75 cm
- Family History: No, Active: Yes
- PRS: -1.5 (very low genetic risk)
- **Profile:** Young, healthy weight, active lifestyle, excellent genetic profile

#### **Patient 2 - Low Risk**
- Age: 35, BMI: 23.0, Waist: 82 cm
- Family History: No, Active: Yes
- PRS: -0.8 (low genetic risk)
- **Profile:** Young adult, normal weight, physically active

#### **Patient 8 - Young Healthy**
- Age: 25, BMI: 22.0, Waist: 78 cm
- Family History: No, Active: Yes
- PRS: -1.2 (low genetic risk)
- **Profile:** Very young athlete, optimal health indicators

---

### 🟡 **Moderate Risk Patients (25-50%)**

#### **Patient 3 - Low-Moderate**
- Age: 42, BMI: 26.5, Waist: 90 cm
- Family History: No, Active: Yes
- PRS: -0.3 (slightly below average)
- **Profile:** Middle-aged, slightly overweight but active

#### **Patient 4 - Moderate Risk**
- Age: 50, BMI: 28.0, Waist: 95 cm
- Family History: Yes, Active: No
- PRS: 0.2 (slightly above average)
- **Profile:** Overweight with family history, sedentary lifestyle

#### **Patient 9 - Older Healthy**
- Age: 65, BMI: 24.5, Waist: 88 cm
- Family History: No, Active: Yes
- PRS: -0.5 (low-moderate genetic risk)
- **Profile:** Senior maintaining healthy lifestyle despite age

---

### 🟠 **High Risk Patients (50-75%)**

#### **Patient 5 - Moderate-High**
- Age: 55, BMI: 30.5, Waist: 102 cm
- Family History: Yes, Active: No
- PRS: 0.6 (moderate genetic risk)
- **Profile:** Obese Class I, family history, sedentary

#### **Patient 6 - High Risk**
- Age: 60, BMI: 32.0, Waist: 108 cm
- Family History: Yes, Active: No
- PRS: 1.0 (high genetic risk)
- **Profile:** Senior, obese, high genetic risk, inactive

#### **Patient 11 - Clinical Risk**
- Age: 58, BMI: 33.0, Waist: 110 cm
- Family History: Yes, Active: No
- PRS: 0.3 (slightly elevated)
- **Profile:** Multiple clinical risk factors dominate

---

### 🔴 **Very High Risk Patients (≥ 75%)**

#### **Patient 7 - Very High Risk**
- Age: 68, BMI: 36.5, Waist: 115 cm
- Family History: Yes, Active: No
- PRS: 1.8 (very high genetic risk)
- **Profile:** Elderly, severely obese, very high genetic risk

#### **Patient 10 - Genetic Risk**
- Age: 40, BMI: 25.0, Waist: 87 cm
- Family History: Yes, Active: Yes
- PRS: 2.5 (extremely high genetic risk - rare)
- **Profile:** Young and active but extremely high genetic predisposition

#### **Patient 12 - Combined Risk**
- Age: 62, BMI: 34.5, Waist: 112 cm
- Family History: Yes, Active: No
- PRS: 1.5 (high genetic risk)
- **Profile:** Worst case scenario - high genetic + clinical risk factors

---

## 🎯 Use Cases for Each Patient Type

### **Educational Demonstrations:**

1. **Low Risk (Patients 1, 2, 8):**
   - Show protective factors: youth, activity, healthy weight
   - Demonstrate low PRS impact
   - Example of optimal health management

2. **Moderate Risk (Patients 3, 4, 9):**
   - Show transition zone - room for intervention
   - Demonstrate impact of lifestyle choices
   - Age vs lifestyle balance

3. **High Risk (Patients 5, 6, 11):**
   - Show accumulation of risk factors
   - Demonstrate need for clinical intervention
   - Impact of obesity + family history

4. **Very High Risk (Patients 7, 10, 12):**
   - Show extreme cases requiring immediate attention
   - Patient 10: genetic risk despite healthy lifestyle
   - Patient 12: combined genetic + clinical factors

---

## 📈 Expected Risk Distribution

Based on the diverse dataset:

- **🟢 Low Risk:** ~25% (3/12 patients)
- **🟡 Moderate Risk:** ~25% (3/12 patients)
- **🟠 High Risk:** ~25% (3/12 patients)
- **🔴 Very High Risk:** ~25% (3/12 patients)

**Balanced representation** across all categories!

---

## 💡 Key Insights to Demonstrate

### 1. **Age Effect**
Compare Patient 8 (25) vs Patient 9 (65) - both healthy lifestyles, different ages

### 2. **Genetic vs Clinical**
Compare Patient 10 (high genetic, low clinical) vs Patient 11 (low genetic, high clinical)

### 3. **Lifestyle Impact**
Compare Patient 6 (inactive) vs Patient 9 (active) - similar ages, different lifestyles

### 4. **Combined Effect**
Show Patient 12 - how genetic + clinical factors compound risk

### 5. **Protective Factors**
Show Patients 1, 2, 8 - how multiple protective factors reduce risk

---

## 🎨 Visualization Features

### New Color-Coded Chart:
- **Green bars** (🟢): Low risk patients
- **Yellow bars** (🟡): Moderate risk patients
- **Orange bars** (🟠): High risk patients
- **Red bars** (🔴): Very high risk patients

### Threshold Lines:
- 25% threshold (green dashed)
- 50% threshold (orange dashed)
- 75% threshold (red dashed)

### Risk Labels:
- Percentage shown on each bar
- Patient descriptions included
- Summary statistics displayed

---

## 📊 Detailed Analysis Output

The updated notebook now shows:

1. **Comprehensive table** with all patient details
2. **Risk distribution** (counts and percentages by category)
3. **Statistical summary** (min, max, mean, median risk)
4. **Visual comparison** with color-coded bars
5. **Patient descriptions** explaining risk profile

---

## 🎓 For Presentations

### Demo Flow:

1. **Show the diverse dataset** - 12 patients, all categories
2. **Run predictions** - calculate risk for all patients
3. **Display detailed analysis** - comprehensive results table
4. **Show visualization** - color-coded risk chart
5. **Highlight specific cases:**
   - Patient 1 vs Patient 7 (best vs worst case)
   - Patient 10 (genetic risk despite health)
   - Patient 9 (healthy aging)

### Key Talking Points:

- "Notice Patient 10 - young and active but highest risk due to genetics"
- "Compare Patient 9 - older but maintains low risk through lifestyle"
- "Patient 12 shows how genetic and clinical factors compound"
- "The model identifies risk across the full spectrum"

---

## 🔧 Customization Options

### Add Your Own Patients:

```python
# Add to the patients DataFrame
new_patient = {
    'Patient_ID': 'Patient 13 - Custom',
    'age': 45,
    'bmi': 27.0,
    'waist_cm': 92,
    'family_history': 0,
    'physical_activity': 1,
    'PRS': 0.1,
    'Description': 'Your custom patient description'
}

patients = patients.append(new_patient, ignore_index=True)
```

### Modify Risk Thresholds:

```python
# Change in the get_risk_category function
def get_risk_category(prob):
    if prob >= 0.80:      # Changed from 0.75
        return 'Very High', '🔴'
    elif prob >= 0.60:    # Changed from 0.50
        return 'High', '🟠'
    # ... etc
```

---

## 📝 Summary

**Dataset Improvements:**
- ✅ **12 patients** (up from 4)
- ✅ **All risk categories** represented equally
- ✅ **Diverse demographics** (age 25-68)
- ✅ **Various risk profiles** (genetic, clinical, combined)
- ✅ **Real-world scenarios** (athlete, senior, obese, genetic)
- ✅ **Educational value** (shows full spectrum)
- ✅ **Presentation-ready** (clear visualizations)

**Perfect for:**
- Academic presentations
- Model demonstrations
- Risk factor education
- Clinical decision-making examples
- Portfolio showcases

---

**Run the updated notebook cells to see the comprehensive analysis!** 🚀
