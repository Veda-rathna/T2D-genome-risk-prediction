"""
Simulate individual genotypes and clinical features from GWAS summary statistics,
train ML models, and generate SHAP/LIME explanations.

Outputs:
 - processed_data/simulated_t2d_dataset.csv
 - models/best_model_ml.pkl
 - processed_data/shap_summary.png
 - processed_data/lime_explanation.png
 - processed_data/simulation_log.txt

Caveats: Data is simulated for methodology demonstration only. See README_SIMULATION.md
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import logging

# Directories
ROOT = os.path.dirname(os.path.dirname(__file__))
PROCESSED = os.path.join(ROOT, 'processed_data')
MODELS = os.path.join(ROOT, 'models')
os.makedirs(PROCESSED, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

# Logging
logging.basicConfig(filename=os.path.join(PROCESSED, 'simulation_log.txt'), level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

np.random.seed(42)


def load_gwas(path):
    df = pd.read_csv(path)
    return df


def simulate_genotypes(gwas_df, n_individuals=5000, top_n_snps=200):
    """Simulate genotypes under HWE for top_n_snps ordered by pval."""
    gdf = gwas_df.sort_values('pval').head(top_n_snps).copy()
    gdf = gdf.reset_index(drop=True)
    n_snps = gdf.shape[0]

    genotypes = np.zeros((n_individuals, n_snps), dtype=np.int8)

    for i, row in gdf.iterrows():
        eaf = float(row['eaf'])
        p0 = (1 - eaf) ** 2
        p1 = 2 * eaf * (1 - eaf)
        p2 = eaf ** 2
        genotypes[:, i] = np.random.choice([0, 1, 2], size=n_individuals, p=[p0, p1, p2])

    gen_df = pd.DataFrame(genotypes, columns=gdf['rsID'].values)
    return gen_df, gdf


def compute_prs(gen_df, gdf):
    # Align weights
    weights = gdf.set_index('rsID')['beta']
    common = [s for s in gen_df.columns if s in weights.index]
    W = weights.loc[common].values
    G = gen_df[common].values
    prs = G.dot(W)
    return prs


def simulate_clinical(prs, n_individuals):
    # Normalize PRS
    prs_norm = (prs - prs.mean()) / (prs.std() + 1e-8)

    age = np.random.randint(25, 75, size=n_individuals)
    bmi = np.clip(18 + 6 + 3 * prs_norm + np.random.normal(0, 3, n_individuals), 15, 45)
    waist_cm = np.clip(70 + 10 * prs_norm + np.random.normal(0, 6, n_individuals), 50, 150)
    family_history = (np.random.rand(n_individuals) < (0.2 + 0.15 * (prs_norm - prs_norm.min()) / (prs_norm.ptp() + 1e-8))).astype(int)
    physical_activity = (np.random.rand(n_individuals) > 0.4 - 0.1 * prs_norm).astype(int)

    df = pd.DataFrame({
        'age': age,
        'bmi': bmi,
        'waist_cm': waist_cm,
        'family_history': family_history,
        'physical_activity': physical_activity
    })
    return df


def assign_phenotype(prs, clinical_df, prevalence=0.2):
    # Logistic model combining prs and clinical signal
    prs_z = (prs - prs.mean()) / (prs.std() + 1e-8)
    clin_score = (0.03 * (clinical_df['age'] - 50) +
                  0.08 * (clinical_df['bmi'] - 25) +
                  0.06 * (clinical_df['waist_cm'] - 90) +
                  0.5 * clinical_df['family_history'] -
                  0.2 * clinical_df['physical_activity'])

    linear = 0.8 * prs_z + 0.6 * (clin_score / (np.std(clin_score) + 1e-8))
    probs = 1 / (1 + np.exp(-linear))

    # scale to desired prevalence
    cutoff = np.quantile(probs, 1 - prevalence)
    phenotype = (probs >= cutoff).astype(int)
    return phenotype, probs


def train_and_evaluate(df, features, label='T2D_status'):
    X = df[features]
    y = df[label]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # XGBoost
    model = xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train_scaled, y_train)

    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    logging.info(f"Trained XGBoost model AUC: {auc:.4f}")

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
    plt.tight_layout()
    shap_path = os.path.join(PROCESSED, 'shap_summary.png')
    plt.savefig(shap_path, dpi=200)
    plt.close()

    # Save model and scaler
    model_path = os.path.join(MODELS, 'best_model_ml.pkl')
    scaler_path = os.path.join(MODELS, 'scaler.pkl')
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    return model, scaler, auc, shap_path


def main():
    gwas_path = os.path.join(PROCESSED, 'cleaned_gwas_summary_stats.csv')
    gwas_df = load_gwas(gwas_path)

    logging.info('Loaded GWAS summary statistics with %d variants' % gwas_df.shape[0])

    gen_df, top_snps = simulate_genotypes(gwas_df, n_individuals=5000, top_n_snps=200)
    logging.info('Simulated genotypes shape: %s' % (str(gen_df.shape)))

    prs = compute_prs(gen_df, top_snps)
    clinical = simulate_clinical(prs, n_individuals=gen_df.shape[0])

    phenotype, probs = assign_phenotype(prs, clinical, prevalence=0.25)

    out = pd.concat([clinical.reset_index(drop=True), pd.DataFrame({'PRS': prs, 'T2D_status': phenotype})], axis=1)
    out = pd.concat([out, gen_df.reset_index(drop=True)], axis=1)

    simulated_path = os.path.join(PROCESSED, 'simulated_t2d_dataset.csv')
    out.to_csv(simulated_path, index=False)
    logging.info('Saved simulated dataset to %s' % simulated_path)

    # Train on combined features
    features = ['age', 'bmi', 'waist_cm', 'family_history', 'physical_activity', 'PRS']
    model, scaler, auc, shap_path = train_and_evaluate(out, features, label='T2D_status')

    logging.info('Model AUC: %.4f' % auc)
    print(f"Trained model AUC: {auc:.4f}")
    print('Saved SHAP summary to', shap_path)


if __name__ == '__main__':
    main()
