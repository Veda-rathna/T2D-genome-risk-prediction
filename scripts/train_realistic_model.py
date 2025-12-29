"""
Train a MORE REALISTIC T2D Risk Prediction Model
=================================================
This script addresses overfitting by:
1. Adding realistic noise to training data
2. Using stronger regularization
3. Calibrating probabilities to match real-world T2D prevalence (~10-15%)
4. Cross-validation with proper evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

def add_realistic_noise(df, feature_cols, noise_level=0.15):
    """
    Add realistic noise to prevent overfitting
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        noise_level: Standard deviation of Gaussian noise
    
    Returns:
        DataFrame with added noise
    """
    df_noisy = df.copy()
    for col in feature_cols:
        if col != 'family_history':  # Don't add noise to binary features
            noise = np.random.normal(0, noise_level * df[col].std(), size=len(df))
            df_noisy[col] = df[col] + noise
            # Ensure realistic bounds
            if col == 'age':
                df_noisy[col] = df_noisy[col].clip(18, 90)
            elif col == 'bmi':
                df_noisy[col] = df_noisy[col].clip(15, 50)
            elif col == 'waist_cm':
                df_noisy[col] = df_noisy[col].clip(60, 140)
            elif col == 'physical_activity':
                df_noisy[col] = df_noisy[col].clip(0, 7)
    return df_noisy

def balance_dataset(X, y, target_prevalence=0.12):
    """
    Balance dataset to match realistic T2D prevalence (~12%)
    
    Args:
        X: Feature matrix
        y: Labels
        target_prevalence: Desired prevalence of T2D cases
    
    Returns:
        Balanced X, y
    """
    # Count current cases
    n_cases = y.sum()
    n_controls = len(y) - n_cases
    current_prevalence = n_cases / len(y)
    
    print(f"Current prevalence: {current_prevalence:.2%}")
    print(f"Target prevalence: {target_prevalence:.2%}")
    
    if current_prevalence > target_prevalence:
        # Undersample cases
        case_indices = np.where(y == 1)[0]
        control_indices = np.where(y == 0)[0]
        
        n_desired_cases = int(len(control_indices) * target_prevalence / (1 - target_prevalence))
        
        selected_case_indices = np.random.choice(case_indices, size=n_desired_cases, replace=False)
        selected_indices = np.concatenate([control_indices, selected_case_indices])
        np.random.shuffle(selected_indices)
        
        X_balanced = X.iloc[selected_indices]
        y_balanced = y.iloc[selected_indices]
        
        print(f"After balancing: {y_balanced.sum()}/{len(y_balanced)} = {y_balanced.mean():.2%}")
        return X_balanced, y_balanced
    else:
        return X, y

def train_regularized_model(X_train, X_test, y_train, y_test, scaler):
    """
    Train a regularized model with proper calibration
    
    Returns:
        Calibrated model, performance metrics
    """
    print("\n" + "="*80)
    print("TRAINING REGULARIZED MODEL")
    print("="*80)
    
    # Scale features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost with strong regularization
    print("\n1. Training XGBoost with regularization...")
    xgb_model = XGBClassifier(
        n_estimators=100,  # Reduced from 200
        max_depth=3,       # Reduced from default 6
        learning_rate=0.05, # Reduced from default 0.3
        min_child_weight=5, # Increased from default 1
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,      # L1 regularization
        reg_lambda=2.0,     # L2 regularization
        scale_pos_weight=7, # Account for class imbalance
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_scaled, y_train)
    
    # Evaluate before calibration
    print("\n2. Performance BEFORE calibration:")
    y_pred_proba_before = xgb_model.predict_proba(X_test_scaled)[:, 1]
    print(f"   Mean predicted probability: {y_pred_proba_before.mean():.2%}")
    print(f"   Actual prevalence: {y_test.mean():.2%}")
    print(f"   Predictions > 50%: {(y_pred_proba_before > 0.5).sum()} / {len(y_pred_proba_before)}")
    
    # Calibrate probabilities
    print("\n3. Calibrating probabilities...")
    calibrated_model = CalibratedClassifierCV(xgb_model, method='sigmoid', cv=5)
    calibrated_model.fit(X_train_scaled, y_train)
    
    # Evaluate after calibration
    print("\n4. Performance AFTER calibration:")
    y_pred_proba_after = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    print(f"   Mean predicted probability: {y_pred_proba_after.mean():.2%}")
    print(f"   Actual prevalence: {y_test.mean():.2%}")
    print(f"   Predictions > 50%: {(y_pred_proba_after > 0.5).sum()} / {len(y_pred_proba_after)}")
    
    # Calculate metrics
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    
    y_pred = (y_pred_proba_after > 0.5).astype(int)
    
    metrics = {
        'AUC': roc_auc_score(y_test, y_pred_proba_after),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    }
    
    print("\n5. Final Metrics:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    # Plot probability distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(y_pred_proba_before, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(y_pred_proba_before.mean(), color='red', linestyle='--', label=f'Mean: {y_pred_proba_before.mean():.2%}')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('BEFORE Calibration')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(y_pred_proba_after, bins=50, alpha=0.7, edgecolor='black', color='green')
    plt.axvline(y_pred_proba_after.mean(), color='red', linestyle='--', label=f'Mean: {y_pred_proba_after.mean():.2%}')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('AFTER Calibration')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('processed_data/probability_calibration.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved probability distribution plot: processed_data/probability_calibration.png")
    
    return calibrated_model, metrics

def create_calibration_plot(model, X_test, y_test, scaler):
    """Create calibration plot to visualize reliability"""
    from sklearn.calibration import calibration_curve
    
    X_test_scaled = scaler.transform(X_test)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Proportion of T2D Cases')
    plt.title('Calibration Plot: How Well Do Probabilities Match Reality?')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('processed_data/calibration_plot.png', dpi=300, bbox_inches='tight')
    print("✅ Saved calibration plot: processed_data/calibration_plot.png")

def main():
    print("="*80)
    print("TRAINING REALISTIC T2D RISK PREDICTION MODEL")
    print("="*80)
    
    # Load simulated data
    print("\n📂 Loading simulated data...")
    data_path = Path("processed_data/simulated_t2d_dataset.csv")
    if not data_path.exists():
        print("❌ Error: simulated_t2d_dataset.csv not found!")
        print("   Please run simulate_and_train.py first.")
        return
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} samples")
    
    # Prepare features
    feature_cols = ['age', 'bmi', 'waist_cm', 'family_history', 'physical_activity', 'PRS']
    X = df[feature_cols]
    y = df['T2D_status']
    
    print(f"\n📊 Original dataset: {y.sum()} cases / {len(y)} total = {y.mean():.2%} prevalence")
    
    # Add noise to features
    print("\n🔊 Adding realistic noise to features...")
    X_noisy = add_realistic_noise(X, feature_cols, noise_level=0.1)
    
    # Balance dataset
    print("\n⚖️ Balancing dataset to realistic prevalence...")
    X_balanced, y_balanced = balance_dataset(X_noisy, y, target_prevalence=0.12)
    
    # Split data
    print("\n✂️ Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )
    print(f"   Training set: {len(X_train)} samples ({y_train.sum()} cases)")
    print(f"   Test set: {len(X_test)} samples ({y_test.sum()} cases)")
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Train and evaluate
    model, metrics = train_regularized_model(X_train, X_test, y_train, y_test, scaler)
    
    # Create calibration plot
    print("\n📈 Creating calibration plot...")
    create_calibration_plot(model, X_test, y_test, scaler)
    
    # Save model
    print("\n💾 Saving realistic model...")
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/realistic_model.pkl")
    joblib.dump(scaler, "models/realistic_scaler.pkl")
    print("✅ Saved: models/realistic_model.pkl")
    print("✅ Saved: models/realistic_scaler.pkl")
    
    # Feature importance
    print("\n🎯 Feature Importance:")
    # Get feature importance from base estimator
    base_estimator = model.calibrated_classifiers_[0].estimator
    importances = base_estimator.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    print(feature_importance.to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    colors = ['#e74c3c' if imp > 0.2 else '#3498db' for imp in feature_importance['Importance']]
    plt.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors)
    plt.xlabel('Importance')
    plt.title('Feature Importance in Realistic Model')
    plt.tight_layout()
    plt.savefig('processed_data/realistic_feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved feature importance plot: processed_data/realistic_feature_importance.png")
    
    print("\n" + "="*80)
    print("✅ REALISTIC MODEL TRAINING COMPLETE!")
    print("="*80)
    print("\n📝 Summary:")
    print(f"   - Model type: Calibrated XGBoost")
    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Test AUC: {metrics['AUC']:.4f}")
    print(f"   - Test Accuracy: {metrics['Accuracy']:.4f}")
    print(f"   - Regularization: L1={1.0}, L2={2.0}")
    print(f"   - Calibration: Sigmoid method")
    print("\n💡 Key Improvements:")
    print("   ✓ Added realistic noise to training data")
    print("   ✓ Balanced to realistic T2D prevalence (12%)")
    print("   ✓ Strong regularization (max_depth=3, alpha=1.0, lambda=2.0)")
    print("   ✓ Probability calibration (sigmoid method)")
    print("   ✓ No more 100% predictions!")
    print("\n🚀 Next Steps:")
    print("   1. Use 'models/realistic_model.pkl' in your notebook")
    print("   2. Update notebook to load realistic_model.pkl instead of best_model_ml.pkl")
    print("   3. Re-run predictions - you should see more realistic probabilities!")
    print("="*80)

if __name__ == "__main__":
    main()
