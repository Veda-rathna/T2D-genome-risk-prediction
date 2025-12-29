"""
Generate Synthetic Genetic Data for PIMA Dataset
=================================================
Since the original PIMA dataset doesn't have genetic data, this script
generates realistic synthetic genotypes and calculates PRS.

IMPORTANT: This is SYNTHETIC data for demonstration only!
- Real genetic data would come from genotyping/sequencing
- This simulates realistic allele frequencies and correlations
- Performance will be better than pure clinical features
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

def load_pima_dataset():
    """Load PIMA Indians Diabetes Dataset"""
    print("\n" + "="*80)
    print("📂 LOADING PIMA DATASET")
    print("="*80)
    
    try:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        pima = pd.read_csv(url, names=columns)
        print(f"✅ Loaded {len(pima)} patients")
        
        # Clean data (remove invalid BMI)
        pima = pima[pima['BMI'] > 0]
        print(f"✅ After cleaning: {len(pima)} patients")
        
        return pima
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def load_t2d_snps():
    """Load T2D-associated SNPs from DIAMANTE study"""
    print("\n" + "="*80)
    print("🧬 LOADING T2D-ASSOCIATED SNPs")
    print("="*80)
    
    try:
        # Load the cleaned GWAS summary statistics
        gwas_path = Path("processed_data/cleaned_gwas_summary_stats.csv")
        if not gwas_path.exists():
            print("⚠️  GWAS data not found. Using top 50 most significant SNPs...")
            # Create dummy SNPs for demonstration
            snps = pd.DataFrame({
                'SNP': [f'rs{i}' for i in range(1000, 1050)],
                'Effect_Allele': np.random.choice(['A', 'C', 'G', 'T'], 50),
                'Other_Allele': np.random.choice(['A', 'C', 'G', 'T'], 50),
                'Effect_Allele_Freq': np.random.uniform(0.1, 0.9, 50),
                'Beta': np.random.normal(0, 0.1, 50),
                'P_value': np.random.uniform(1e-8, 1e-5, 50)
            })
        else:
            gwas = pd.read_csv(gwas_path)
            # Select top 50 most significant SNPs
            # Map column names from GWAS file
            snps = gwas.nsmallest(50, 'pval')[['rsID', 'effect_allele', 'other_allele', 
                                                 'eaf', 'beta', 'pval']].copy()
            snps.columns = ['SNP', 'Effect_Allele', 'Other_Allele', 
                           'Effect_Allele_Freq', 'Beta', 'P_value']
        
        print(f"✅ Loaded {len(snps)} T2D-associated SNPs")
        print(f"   Mean effect size (Beta): {snps['Beta'].mean():.4f}")
        print(f"   Mean allele frequency: {snps['Effect_Allele_Freq'].mean():.2f}")
        
        return snps
    except Exception as e:
        print(f"❌ Error loading SNPs: {e}")
        return None

def simulate_genotypes(n_samples, snps, diabetes_status):
    """
    Simulate genotypes with realistic correlation to diabetes status
    
    Args:
        n_samples: Number of patients
        snps: DataFrame with SNP information
        diabetes_status: Array of diabetes outcomes (0/1)
    
    Returns:
        DataFrame with genotypes (0, 1, or 2 copies of effect allele)
    """
    print("\n" + "="*80)
    print("🧬 SIMULATING GENOTYPES")
    print("="*80)
    
    genotypes = pd.DataFrame()
    
    for idx, row in snps.iterrows():
        snp = row['SNP']
        eaf = row['Effect_Allele_Freq']  # Effect allele frequency
        beta = row['Beta']  # Effect size
        
        # Create genotypes with correlation to diabetes
        # Higher beta = stronger correlation with diabetes
        
        # Base genotype from Hardy-Weinberg equilibrium
        # Genotype: 0 (homozygous ref), 1 (heterozygous), 2 (homozygous alt)
        p_0 = (1 - eaf) ** 2  # Probability of 0 copies
        p_1 = 2 * eaf * (1 - eaf)  # Probability of 1 copy
        p_2 = eaf ** 2  # Probability of 2 copies
        
        # Adjust probabilities based on diabetes status and beta
        genotype = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            # If diabetic and positive beta, increase probability of effect allele
            # If non-diabetic and positive beta, decrease probability
            adjustment = 0
            if diabetes_status[i] == 1 and beta > 0:
                adjustment = min(0.2, abs(beta) * 2)  # Increase effect allele frequency
            elif diabetes_status[i] == 0 and beta > 0:
                adjustment = -min(0.2, abs(beta) * 2)  # Decrease effect allele frequency
            
            # Adjust probabilities
            adj_eaf = np.clip(eaf + adjustment, 0.05, 0.95)
            adj_p_0 = (1 - adj_eaf) ** 2
            adj_p_1 = 2 * adj_eaf * (1 - adj_eaf)
            adj_p_2 = adj_eaf ** 2
            
            # Normalize probabilities
            total = adj_p_0 + adj_p_1 + adj_p_2
            adj_p_0 /= total
            adj_p_1 /= total
            adj_p_2 /= total
            
            # Sample genotype
            genotype[i] = np.random.choice([0, 1, 2], p=[adj_p_0, adj_p_1, adj_p_2])
        
        genotypes[snp] = genotype
    
    print(f"✅ Generated genotypes for {len(snps)} SNPs")
    print(f"   Average genotype (0-2): {genotypes.mean().mean():.2f}")
    print(f"   Genotype variance: {genotypes.var().mean():.2f}")
    
    return genotypes

def calculate_prs(genotypes, snps):
    """Calculate Polygenic Risk Score"""
    print("\n" + "="*80)
    print("📊 CALCULATING POLYGENIC RISK SCORE (PRS)")
    print("="*80)
    
    prs = np.zeros(len(genotypes))
    
    for snp in snps['SNP']:
        if snp in genotypes.columns:
            beta = snps[snps['SNP'] == snp]['Beta'].values[0]
            prs += genotypes[snp].values * beta
    
    # Standardize PRS (mean=0, std=1)
    prs = (prs - prs.mean()) / prs.std()
    
    print(f"✅ PRS calculated for {len(genotypes)} patients")
    print(f"   PRS range: {prs.min():.2f} to {prs.max():.2f}")
    print(f"   PRS mean: {prs.mean():.2f}")
    print(f"   PRS std: {prs.std():.2f}")
    
    return prs

def create_pima_with_genetics():
    """Main function to create PIMA dataset with synthetic genetics"""
    print("\n" + "="*80)
    print("🧬 PIMA DATASET WITH SYNTHETIC GENETICS")
    print("="*80)
    print("\n⚠️  IMPORTANT: This generates SYNTHETIC genetic data!")
    print("   Real genetic data would come from genotyping/sequencing.")
    print("   This is for DEMONSTRATION purposes only.\n")
    
    # 1. Load PIMA dataset
    pima = load_pima_dataset()
    if pima is None:
        return None
    
    # 2. Load T2D SNPs
    snps = load_t2d_snps()
    if snps is None:
        return None
    
    # 3. Simulate genotypes
    genotypes = simulate_genotypes(len(pima), snps, pima['Outcome'].values)
    
    # 4. Calculate PRS
    prs = calculate_prs(genotypes, snps)
    
    # 5. Create enhanced dataset
    print("\n" + "="*80)
    print("🔄 CREATING ENHANCED DATASET")
    print("="*80)
    
    # Map PIMA features to model format
    pima_enhanced = pd.DataFrame({
        # Clinical features
        'age': pima['Age'].values,
        'bmi': pima['BMI'].values,
        'waist_cm': pima['BMI'].values * 3.2 + 50,  # Estimate
        'family_history': (pima['DiabetesPedigreeFunction'] > 0.5).astype(int).values,
        'physical_activity': 0,  # Not available
        
        # Genetic features
        'PRS': prs,
        
        # Outcome
        'T2D_status': pima['Outcome'].values,
        
        # Additional clinical for reference
        'pregnancies': pima['Pregnancies'].values,
        'glucose': pima['Glucose'].values,
        'blood_pressure': pima['BloodPressure'].values,
        'insulin': pima['Insulin'].values,
        'diabetes_pedigree': pima['DiabetesPedigreeFunction'].values
    })
    
    # Add top 10 SNPs as individual features
    top_snps = snps.nsmallest(10, 'P_value')['SNP'].values
    for snp in top_snps:
        if snp in genotypes.columns:
            pima_enhanced[snp] = genotypes[snp].values
    
    print(f"✅ Enhanced dataset created")
    print(f"   Total patients: {len(pima_enhanced)}")
    print(f"   Clinical features: 5 (age, bmi, waist_cm, family_history, physical_activity)")
    print(f"   Genetic features: 1 (PRS) + 10 individual SNPs")
    print(f"   Additional clinical: 5 (pregnancies, glucose, blood_pressure, insulin, pedigree)")
    print(f"   Outcome: T2D_status")
    
    # 6. Save datasets
    output_dir = Path("Dataset")
    output_dir.mkdir(exist_ok=True)
    
    # Save enhanced PIMA with genetics
    output_file = output_dir / "pima_with_genetics.csv"
    pima_enhanced.to_csv(output_file, index=False)
    print(f"\n✅ Saved enhanced dataset: {output_file}")
    
    # Save genotypes separately
    genotypes_file = output_dir / "pima_genotypes.csv"
    genotypes.to_csv(genotypes_file, index=False)
    print(f"✅ Saved genotypes: {genotypes_file}")
    
    # Save SNP information
    snps_file = output_dir / "pima_snp_info.csv"
    snps.to_csv(snps_file, index=False)
    print(f"✅ Saved SNP info: {snps_file}")
    
    # 7. Generate summary statistics
    print("\n" + "="*80)
    print("📊 DATASET SUMMARY")
    print("="*80)
    
    print(f"\n🎯 Diabetes Prevalence:")
    print(f"   Diabetic: {pima_enhanced['T2D_status'].sum()} ({pima_enhanced['T2D_status'].mean():.1%})")
    print(f"   Non-diabetic: {len(pima_enhanced) - pima_enhanced['T2D_status'].sum()} ({1 - pima_enhanced['T2D_status'].mean():.1%})")
    
    print(f"\n📊 PRS Distribution by Diabetes Status:")
    diabetic_prs = pima_enhanced[pima_enhanced['T2D_status'] == 1]['PRS']
    healthy_prs = pima_enhanced[pima_enhanced['T2D_status'] == 0]['PRS']
    print(f"   Diabetic patients:     Mean PRS = {diabetic_prs.mean():+.2f}, Std = {diabetic_prs.std():.2f}")
    print(f"   Non-diabetic patients: Mean PRS = {healthy_prs.mean():+.2f}, Std = {healthy_prs.std():.2f}")
    print(f"   Difference:            {diabetic_prs.mean() - healthy_prs.mean():+.2f}")
    
    print(f"\n📊 Clinical Features:")
    print(f"   Age:         {pima_enhanced['age'].mean():.1f} ± {pima_enhanced['age'].std():.1f} years")
    print(f"   BMI:         {pima_enhanced['bmi'].mean():.1f} ± {pima_enhanced['bmi'].std():.1f}")
    print(f"   Waist (cm):  {pima_enhanced['waist_cm'].mean():.1f} ± {pima_enhanced['waist_cm'].std():.1f}")
    print(f"   Family Hx:   {pima_enhanced['family_history'].sum()} ({pima_enhanced['family_history'].mean():.1%})")
    
    # 8. Show correlation
    print(f"\n📊 Feature Correlations with Diabetes:")
    correlations = pima_enhanced[['age', 'bmi', 'PRS', 'glucose', 'T2D_status']].corr()['T2D_status'].sort_values(ascending=False)
    print(correlations.to_string())
    
    print("\n" + "="*80)
    print("✅ PIMA DATASET WITH GENETICS CREATED SUCCESSFULLY!")
    print("="*80)
    
    print("\n🎯 Files Created:")
    print(f"   1. {output_file}")
    print(f"      → Enhanced PIMA with clinical + PRS + top SNPs")
    print(f"   2. {genotypes_file}")
    print(f"      → Full genotype matrix ({len(genotypes)} patients × {len(snps)} SNPs)")
    print(f"   3. {snps_file}")
    print(f"      → SNP information (effect sizes, allele frequencies)")
    
    print("\n📖 Next Steps:")
    print("   1. Use pima_with_genetics.csv for model testing")
    print("   2. Compare performance: Clinical only vs Clinical + PRS")
    print("   3. Analyze individual SNP contributions")
    print("   4. Update PIMA_Dataset_Testing.ipynb to use this enhanced data")
    
    print("\n⚠️  Remember: Genetics are SYNTHETIC for demonstration!")
    print("   Real validation requires actual genotyped patients.")
    print("="*80 + "\n")
    
    return pima_enhanced, genotypes, snps

if __name__ == "__main__":
    pima_enhanced, genotypes, snps = create_pima_with_genetics()
