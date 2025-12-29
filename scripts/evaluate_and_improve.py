"""
Comprehensive Model Evaluation and Improvement Pipeline

This script:
1. Evaluates model performance with multiple metrics
2. Compares different model architectures
3. Performs ablation studies (clinical only, genetic only, combined)
4. Generates detailed visualizations
5. Provides recommendations for improvement
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix, accuracy_score, f1_score,
    precision_score, recall_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Setup
ROOT = os.path.dirname(os.path.dirname(__file__))
PROCESSED = os.path.join(ROOT, 'processed_data')
MODELS = os.path.join(ROOT, 'models')
os.makedirs(PROCESSED, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

np.random.seed(42)

def load_data():
    """Load simulated dataset"""
    df = pd.read_csv(os.path.join(PROCESSED, 'simulated_t2d_dataset.csv'))
    return df

def evaluate_model_comprehensive(model, X_train, X_test, y_train, y_test, model_name):
    """Comprehensive model evaluation with multiple metrics"""
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'Model': model_name,
        'AUC-ROC': roc_auc_score(y_test, y_pred_proba),
        'AUC-PR': average_precision_score(y_test, y_pred_proba),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'Specificity': recall_score(y_test, y_pred, pos_label=0),
        'F1-Score': f1_score(y_test, y_pred)
    }
    
    # Cross-validation AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    metrics['CV-AUC (mean)'] = cv_scores.mean()
    metrics['CV-AUC (std)'] = cv_scores.std()
    
    return metrics, y_pred, y_pred_proba

def compare_models(X_train, X_test, y_train, y_test, feature_set_name):
    """Compare multiple ML algorithms"""
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss', use_label_encoder=False)
    }
    
    results = []
    predictions = {}
    
    print(f"\n{'='*70}")
    print(f"EVALUATING MODELS: {feature_set_name}")
    print('='*70)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        metrics, y_pred, y_pred_proba = evaluate_model_comprehensive(
            model, X_train, X_test, y_train, y_test, name
        )
        
        results.append(metrics)
        predictions[name] = {'pred': y_pred, 'proba': y_pred_proba}
        
        print(f"  AUC-ROC: {metrics['AUC-ROC']:.4f}")
        print(f"  Accuracy: {metrics['Accuracy']:.4f}")
        print(f"  F1-Score: {metrics['F1-Score']:.4f}")
    
    results_df = pd.DataFrame(results)
    return results_df, predictions, models

def ablation_study(df):
    """Compare Clinical Only vs Genetic Only vs Combined"""
    
    print("\n" + "="*70)
    print("ABLATION STUDY: Feature Set Comparison")
    print("="*70)
    
    feature_sets = {
        'Clinical Only (IDRS Baseline)': ['age', 'bmi', 'waist_cm', 'family_history', 'physical_activity'],
        'Genetic Only (PRS)': ['PRS'],
        'Combined (Proposed)': ['age', 'bmi', 'waist_cm', 'family_history', 'physical_activity', 'PRS']
    }
    
    all_results = []
    all_predictions = {}
    best_models = {}
    
    for set_name, features in feature_sets.items():
        X = df[features]
        y = df['T2D_status']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results_df, predictions, models = compare_models(
            X_train_scaled, X_test_scaled, y_train, y_test, set_name
        )
        
        results_df['Feature Set'] = set_name
        all_results.append(results_df)
        all_predictions[set_name] = {
            'predictions': predictions,
            'y_test': y_test,
            'X_test': X_test
        }
        best_models[set_name] = models
    
    final_results = pd.concat(all_results, ignore_index=True)
    return final_results, all_predictions, best_models

def plot_model_comparison(results_df, save_path):
    """Visualize model comparison"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. AUC-ROC Comparison
    pivot_auc = results_df.pivot(index='Model', columns='Feature Set', values='AUC-ROC')
    pivot_auc.plot(kind='bar', ax=axes[0, 0], width=0.8)
    axes[0, 0].set_title('AUC-ROC Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('AUC-ROC Score')
    axes[0, 0].set_ylim([0.5, 1.0])
    axes[0, 0].axhline(y=0.80, color='r', linestyle='--', label='Target (0.80)')
    axes[0, 0].axhline(y=0.66, color='orange', linestyle='--', label='IDRS Baseline (0.66)')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Accuracy Comparison
    pivot_acc = results_df.pivot(index='Model', columns='Feature Set', values='Accuracy')
    pivot_acc.plot(kind='bar', ax=axes[0, 1], width=0.8)
    axes[0, 1].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_ylim([0.5, 1.0])
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. F1-Score Comparison
    pivot_f1 = results_df.pivot(index='Model', columns='Feature Set', values='F1-Score')
    pivot_f1.plot(kind='bar', ax=axes[1, 0], width=0.8)
    axes[1, 0].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].set_ylim([0.5, 1.0])
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Precision vs Recall
    axes[1, 1].scatter(results_df['Recall'], results_df['Precision'], 
                       c=results_df['AUC-ROC'], cmap='viridis', s=100, alpha=0.6)
    axes[1, 1].set_xlabel('Recall (Sensitivity)')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].set_title('Precision vs Recall', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label('AUC-ROC')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved model comparison plot: {save_path}")

def plot_roc_curves(all_predictions, save_path):
    """Plot ROC curves for all feature sets"""
    
    plt.figure(figsize=(10, 8))
    
    colors = {'Clinical Only (IDRS Baseline)': 'blue', 
              'Genetic Only (PRS)': 'green', 
              'Combined (Proposed)': 'red'}
    
    for set_name, data in all_predictions.items():
        # Use XGBoost predictions (best model)
        y_test = data['y_test']
        y_proba = data['predictions']['XGBoost']['proba']
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        
        plt.plot(fpr, tpr, color=colors.get(set_name, 'gray'), 
                 lw=2, label=f'{set_name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curves: Feature Set Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved ROC curves: {save_path}")

def plot_confusion_matrices(all_predictions, save_path):
    """Plot confusion matrices for best model across feature sets"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (set_name, data) in enumerate(all_predictions.items()):
        y_test = data['y_test']
        y_pred = data['predictions']['XGBoost']['pred']
        
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['No T2D', 'T2D'],
                    yticklabels=['No T2D', 'T2D'])
        axes[idx].set_title(set_name, fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        
        axes[idx].text(0.5, -0.15, 
                       f'Sensitivity: {sensitivity:.3f}\nSpecificity: {specificity:.3f}',
                       ha='center', transform=axes[idx].transAxes, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved confusion matrices: {save_path}")

def generate_evaluation_report(results_df, save_path):
    """Generate comprehensive evaluation report"""
    
    report = []
    report.append("="*80)
    report.append("COMPREHENSIVE MODEL EVALUATION REPORT")
    report.append("="*80)
    report.append("")
    
    # Overall best model
    best_idx = results_df['AUC-ROC'].idxmax()
    best_model = results_df.loc[best_idx]
    
    report.append("🏆 BEST PERFORMING MODEL")
    report.append("-" * 80)
    report.append(f"Model: {best_model['Model']}")
    report.append(f"Feature Set: {best_model['Feature Set']}")
    report.append(f"AUC-ROC: {best_model['AUC-ROC']:.4f}")
    report.append(f"Accuracy: {best_model['Accuracy']:.4f}")
    report.append(f"F1-Score: {best_model['F1-Score']:.4f}")
    report.append(f"Precision: {best_model['Precision']:.4f}")
    report.append(f"Recall (Sensitivity): {best_model['Recall']:.4f}")
    report.append(f"Specificity: {best_model['Specificity']:.4f}")
    report.append("")
    
    # Feature set comparison
    report.append("📊 FEATURE SET COMPARISON (XGBoost)")
    report.append("-" * 80)
    xgb_results = results_df[results_df['Model'] == 'XGBoost'].sort_values('AUC-ROC', ascending=False)
    for _, row in xgb_results.iterrows():
        report.append(f"\n{row['Feature Set']}:")
        report.append(f"  AUC-ROC: {row['AUC-ROC']:.4f}")
        report.append(f"  Accuracy: {row['Accuracy']:.4f}")
        report.append(f"  F1-Score: {row['F1-Score']:.4f}")
    report.append("")
    
    # Performance assessment
    report.append("✅ PERFORMANCE ASSESSMENT")
    report.append("-" * 80)
    
    target_auc = 0.80
    idrs_baseline = 0.66
    best_auc = best_model['AUC-ROC']
    
    if best_auc >= target_auc:
        report.append(f"✅ Target AUC (≥{target_auc}) ACHIEVED: {best_auc:.4f}")
    else:
        report.append(f"⚠️  Target AUC (≥{target_auc}) NOT MET: {best_auc:.4f}")
    
    improvement = ((best_auc - idrs_baseline) / idrs_baseline) * 100
    report.append(f"📈 Improvement over IDRS baseline: {improvement:.1f}%")
    report.append("")
    
    # Model strengths and weaknesses
    report.append("🔍 MODEL ANALYSIS")
    report.append("-" * 80)
    
    if best_model['Precision'] > 0.85:
        report.append("✅ HIGH PRECISION: Low false positive rate (good for screening)")
    else:
        report.append("⚠️  MODERATE PRECISION: Some false positives expected")
    
    if best_model['Recall'] > 0.85:
        report.append("✅ HIGH RECALL: Catches most T2D cases (good sensitivity)")
    else:
        report.append("⚠️  MODERATE RECALL: May miss some T2D cases")
    
    if best_model['Specificity'] > 0.85:
        report.append("✅ HIGH SPECIFICITY: Good at identifying healthy individuals")
    else:
        report.append("⚠️  MODERATE SPECIFICITY: Some healthy individuals misclassified")
    report.append("")
    
    # Recommendations
    report.append("💡 RECOMMENDATIONS FOR IMPROVEMENT")
    report.append("-" * 80)
    
    recommendations = []
    
    # Check if more realistic simulation needed
    if best_auc > 0.95:
        recommendations.append("1. SIMULATION TOO OPTIMISTIC:")
        recommendations.append("   - Current AUC is unrealistically high (likely due to simulation)")
        recommendations.append("   - Add more noise to clinical features")
        recommendations.append("   - Reduce correlation between PRS and phenotype")
        recommendations.append("   - Expected real-world AUC: 0.75-0.85")
        recommendations.append("")
    
    # Check feature engineering
    clinical_auc = results_df[(results_df['Model'] == 'XGBoost') & 
                              (results_df['Feature Set'] == 'Clinical Only (IDRS Baseline)')]['AUC-ROC'].values[0]
    combined_auc = results_df[(results_df['Model'] == 'XGBoost') & 
                              (results_df['Feature Set'] == 'Combined (Proposed)')]['AUC-ROC'].values[0]
    
    if combined_auc - clinical_auc < 0.05:
        recommendations.append("2. GENETIC FEATURES NOT ADDING MUCH VALUE:")
        recommendations.append("   - Ensure PRS is calculated correctly")
        recommendations.append("   - Consider adding individual SNP features")
        recommendations.append("   - Use more SNPs in PRS calculation")
        recommendations.append("")
    
    # Check model selection
    model_aucs = results_df.groupby('Model')['AUC-ROC'].max()
    auc_range = model_aucs.max() - model_aucs.min()
    if auc_range < 0.02:
        recommendations.append("3. ALL MODELS PERFORM SIMILARLY:")
        recommendations.append("   - Choose simpler model (Logistic Regression) for interpretability")
        recommendations.append("   - Or use ensemble of multiple models")
        recommendations.append("")
    
    # General recommendations
    recommendations.append("4. GENERAL IMPROVEMENTS:")
    recommendations.append("   - Try deep learning models (MLP, CNN)")
    recommendations.append("   - Perform hyperparameter tuning (GridSearchCV)")
    recommendations.append("   - Add interaction features (e.g., Age × BMI)")
    recommendations.append("   - Consider SMOTE for class balancing")
    recommendations.append("   - Use nested cross-validation for robust evaluation")
    recommendations.append("")
    
    recommendations.append("5. FOR REAL-WORLD DEPLOYMENT:")
    recommendations.append("   - Validate on independent cohort")
    recommendations.append("   - Test on real patient data")
    recommendations.append("   - Calibrate probability scores (Platt scaling)")
    recommendations.append("   - Conduct prospective clinical trial")
    
    report.extend(recommendations)
    report.append("")
    report.append("="*80)
    
    report_text = "\n".join(report)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print(f"\n✅ Saved evaluation report: {save_path}")
    
    return report_text

def main():
    """Main execution"""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL EVALUATION AND IMPROVEMENT")
    print("="*70)
    
    # Load data
    print("\n[1/6] Loading data...")
    df = load_data()
    print(f"   Dataset: {df.shape[0]} individuals, {df.shape[1]} features")
    
    # Ablation study
    print("\n[2/6] Running ablation study...")
    results_df, all_predictions, best_models = ablation_study(df)
    
    # Save results
    results_path = os.path.join(PROCESSED, 'model_evaluation_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Saved evaluation results: {results_path}")
    
    # Visualizations
    print("\n[3/6] Generating comparison plots...")
    plot_model_comparison(results_df, os.path.join(PROCESSED, 'model_comparison.png'))
    
    print("\n[4/6] Generating ROC curves...")
    plot_roc_curves(all_predictions, os.path.join(PROCESSED, 'roc_curves.png'))
    
    print("\n[5/6] Generating confusion matrices...")
    plot_confusion_matrices(all_predictions, os.path.join(PROCESSED, 'confusion_matrices.png'))
    
    # Generate report
    print("\n[6/6] Generating evaluation report...")
    report_path = os.path.join(PROCESSED, 'evaluation_report.txt')
    generate_evaluation_report(results_df, report_path)
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print(f"  1. {results_path}")
    print(f"  2. {os.path.join(PROCESSED, 'model_comparison.png')}")
    print(f"  3. {os.path.join(PROCESSED, 'roc_curves.png')}")
    print(f"  4. {os.path.join(PROCESSED, 'confusion_matrices.png')}")
    print(f"  5. {report_path}")

if __name__ == '__main__':
    main()
