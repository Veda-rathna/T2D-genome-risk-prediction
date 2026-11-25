"""
Train and serialize a real-data clinical T2D baseline model on the UCI PIMA Indians
Diabetes Dataset (`data/pima_real_clinical.csv`, n=768 real patient records).

No synthetic genotypes or synthetic labels are used in this model.
Outputs:
  - models/pima_real_clinical_model.pkl
"""

from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "pima_real_clinical.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

PIMA_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
ZERO_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df[PIMA_FEATURES].copy()
    for col in ZERO_MISSING_COLS:
        X[col] = X[col].replace(0, np.nan)
    y = df["Outcome"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                XGBClassifier(
                    n_estimators=120,
                    max_depth=3,
                    learning_rate=0.05,
                    min_child_weight=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=1.0,
                    reg_lambda=2.0,
                    random_state=42,
                    eval_metric="logloss",
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")

    out_path = MODELS_DIR / "pima_real_clinical_model.pkl"
    joblib.dump(pipeline, out_path)

    print(f"Saved real clinical model: {out_path}")
    print(f"Test AUC-ROC:  {roc_auc_score(y_test, y_proba):.4f}")
    print(f"5-Fold CV AUC: {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")
    print(f"Accuracy:      {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision:     {precision_score(y_test, y_pred):.4f}")
    print(f"Sensitivity:   {tp / (tp + fn):.4f}")
    print(f"Specificity:   {tn / (tn + fp):.4f}")
    print(f"F1-Score:      {f1_score(y_test, y_pred):.4f}")


if __name__ == "__main__":
    main()
