"""
GWAS Summary Statistics Preprocessing Pipeline
================================================
Dataset: DIAMANTE South Asian T2D GWAS Summary Statistics
Author: Type 2 Diabetes Genome Risk Prediction Project
Date: December 2025

This script performs comprehensive preprocessing of GWAS summary statistics
to prepare them for polygenic risk score (PRS) construction and downstream
machine learning analysis.

Preprocessing Steps:
1. Load and basic sanity checks
2. Column renaming for consistency
3. Remove ambiguous strand SNPs
4. Minor allele frequency (MAF) filtering
5. Statistical significance filtering
6. Effect size quality control
7. Deduplication
8. LD-aware SNP selection
9. Create PRS weight file
10. Generate quality control report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class GWASPreprocessor:
    """
    A comprehensive preprocessor for GWAS summary statistics.
    """
    
    def __init__(self, input_file, output_dir="processed_data"):
        """
        Initialize the preprocessor.
        
        Parameters:
        -----------
        input_file : str
            Path to the GWAS summary statistics CSV file
        output_dir : str
            Directory to save processed outputs
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.df = None
        self.df_clean = None
        self.log = []
        
        # Create output directory if it doesn't exist
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def _log_step(self, message, count=None):
        """Log a preprocessing step with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if count is not None:
            log_msg = f"[{timestamp}] {message} | SNPs remaining: {count:,}"
        else:
            log_msg = f"[{timestamp}] {message}"
        self.log.append(log_msg)
        print(log_msg)
        
    def step1_load_data(self):
        """
        Step 1: Load and perform basic sanity checks.
        """
        print("\n" + "="*70)
        print("STEP 1: LOAD AND BASIC SANITY CHECKS")
        print("="*70)
        
        self._log_step("Loading GWAS summary statistics...")
        self.df = pd.read_csv(self.input_file)
        
        initial_count = len(self.df)
        self._log_step(f"Initial dataset loaded", initial_count)
        
        # Display basic information
        print("\n📊 Dataset Shape:", self.df.shape)
        print("\n📋 Column Names:")
        for i, col in enumerate(self.df.columns, 1):
            print(f"   {i}. {col}")
        
        # Check for required columns
        required_cols = [
            "chromosome(b37)",
            "position(b37)",
            "rsID",
            "effect_allele",
            "other_allele",
            "effect_allele_frequency",
            "Fixed-effects_beta",
            "Fixed-effects_p-value"
        ]
        
        print("\n✅ Checking for mandatory columns...")
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        else:
            print("   All mandatory columns present!")
        
        # Check missing values
        print("\n🔍 Missing values per column:")
        missing = self.df[required_cols].isnull().sum()
        for col, count in missing.items():
            pct = (count / len(self.df)) * 100
            print(f"   {col}: {count:,} ({pct:.2f}%)")
        
        # Remove rows with missing critical values
        critical_cols = ["rsID", "effect_allele", "other_allele", 
                        "Fixed-effects_beta", "Fixed-effects_p-value"]
        
        self._log_step("Removing rows with missing critical values...")
        self.df = self.df.dropna(subset=critical_cols)
        self._log_step("Removed missing value rows", len(self.df))
        
        return self
    
    def step2_rename_columns(self):
        """
        Step 2: Rename columns to be ML-friendly.
        """
        print("\n" + "="*70)
        print("STEP 2: COLUMN RENAMING")
        print("="*70)
        
        self._log_step("Renaming columns for consistency...")
        
        self.df = self.df.rename(columns={
            "chromosome(b37)": "chr",
            "position(b37)": "pos",
            "Fixed-effects_beta": "beta",
            "Fixed-effects_SE": "se",
            "Fixed-effects_p-value": "pval",
            "effect_allele_frequency": "eaf"
        })
        
        print("✅ Renamed columns:")
        print(f"   chromosome(b37) → chr")
        print(f"   position(b37) → pos")
        print(f"   Fixed-effects_beta → beta")
        print(f"   Fixed-effects_SE → se")
        print(f"   Fixed-effects_p-value → pval")
        print(f"   effect_allele_frequency → eaf")
        
        return self
    
    def step3_remove_ambiguous_snps(self):
        """
        Step 3: Remove ambiguous strand SNPs (A/T, T/A, C/G, G/C).
        This is CRITICAL for avoiding alignment issues.
        """
        print("\n" + "="*70)
        print("STEP 3: ALLELE SANITY CHECKS - REMOVE AMBIGUOUS SNPs")
        print("="*70)
        
        before_count = len(self.df)
        
        # Convert alleles to uppercase for consistent comparison
        self.df['effect_allele'] = self.df['effect_allele'].str.upper()
        self.df['other_allele'] = self.df['other_allele'].str.upper()
        
        # Identify ambiguous SNPs
        ambiguous = (
            ((self.df["effect_allele"] == "A") & (self.df["other_allele"] == "T")) |
            ((self.df["effect_allele"] == "T") & (self.df["other_allele"] == "A")) |
            ((self.df["effect_allele"] == "C") & (self.df["other_allele"] == "G")) |
            ((self.df["effect_allele"] == "G") & (self.df["other_allele"] == "C"))
        )
        
        ambiguous_count = ambiguous.sum()
        print(f"\n⚠️  Found {ambiguous_count:,} ambiguous SNPs:")
        print(f"   A/T or T/A pairs")
        print(f"   C/G or G/C pairs")
        
        self.df = self.df[~ambiguous]
        
        removed_count = before_count - len(self.df)
        self._log_step(f"Removed {removed_count:,} ambiguous SNPs", len(self.df))
        
        print(f"\n📌 This is standard GWAS practice to avoid strand ambiguity.")
        
        return self
    
    def step4_maf_filtering(self, maf_threshold=0.01):
        """
        Step 4: Filter by Minor Allele Frequency (MAF).
        Remove rare variants that are unstable.
        """
        print("\n" + "="*70)
        print(f"STEP 4: MINOR ALLELE FREQUENCY (MAF) FILTERING")
        print("="*70)
        
        before_count = len(self.df)
        
        # Handle missing EAF values
        self.df = self.df.dropna(subset=['eaf'])
        
        print(f"\n📊 EAF distribution before filtering:")
        print(f"   Min: {self.df['eaf'].min():.4f}")
        print(f"   Max: {self.df['eaf'].max():.4f}")
        print(f"   Mean: {self.df['eaf'].mean():.4f}")
        
        # Apply MAF filter
        self.df = self.df[self.df["eaf"] > maf_threshold]
        
        removed_count = before_count - len(self.df)
        self._log_step(f"Removed {removed_count:,} rare variants (MAF < {maf_threshold})", 
                      len(self.df))
        
        print(f"\n💡 Reason: Rare SNPs inflate noise and reduce generalization")
        
        return self
    
    def step5_significance_filtering(self, pval_threshold=1e-5, strategy="PRS"):
        """
        Step 5: Filter by statistical significance.
        
        Parameters:
        -----------
        pval_threshold : float
            P-value threshold (default: 1e-5 for PRS-optimized)
        strategy : str
            'GWAS' for strict genome-wide significance (5e-8)
            'PRS' for PRS-optimized (1e-5) - RECOMMENDED
        """
        print("\n" + "="*70)
        print(f"STEP 5: STATISTICAL SIGNIFICANCE FILTERING")
        print("="*70)
        
        before_count = len(self.df)
        
        if strategy == "GWAS":
            pval_threshold = 5e-8
            print(f"\n🔹 Strategy A: Strict GWAS markers")
            print(f"   Threshold: p < 5e-8 (genome-wide significance)")
            print(f"   Use case: Small, strong SNP set for interpretation")
        else:
            print(f"\n🔹 Strategy B: PRS-optimized (RECOMMENDED)")
            print(f"   Threshold: p < {pval_threshold}")
            print(f"   Use case: Captures many small effects, improves AUC")
        
        self.df_clean = self.df[self.df["pval"] < pval_threshold].copy()
        
        removed_count = before_count - len(self.df_clean)
        self._log_step(f"Retained {len(self.df_clean):,} significant SNPs", 
                      len(self.df_clean))
        
        # Show p-value distribution
        print(f"\n📊 P-value statistics of retained SNPs:")
        print(f"   Min: {self.df_clean['pval'].min():.2e}")
        print(f"   Max: {self.df_clean['pval'].max():.2e}")
        print(f"   Median: {self.df_clean['pval'].median():.2e}")
        
        return self
    
    def step6_effect_size_qc(self, beta_threshold=5):
        """
        Step 6: Quality control on effect sizes.
        Remove SNPs with unrealistic effect sizes.
        """
        print("\n" + "="*70)
        print(f"STEP 6: EFFECT SIZE QUALITY CONTROL")
        print("="*70)
        
        before_count = len(self.df_clean)
        
        print(f"\n📊 Beta distribution before filtering:")
        print(f"   Min: {self.df_clean['beta'].min():.4f}")
        print(f"   Max: {self.df_clean['beta'].max():.4f}")
        print(f"   Mean: {self.df_clean['beta'].mean():.4f}")
        
        # Filter extreme betas
        self.df_clean = self.df_clean[self.df_clean["beta"].abs() < beta_threshold]
        
        removed_count = before_count - len(self.df_clean)
        self._log_step(f"Removed {removed_count} SNPs with |beta| > {beta_threshold}", 
                      len(self.df_clean))
        
        print(f"\n💡 Reason: Extremely large betas usually indicate artifacts")
        
        return self
    
    def step7_deduplication(self):
        """
        Step 7: Remove duplicate SNPs.
        """
        print("\n" + "="*70)
        print(f"STEP 7: DEDUPLICATION")
        print("="*70)
        
        before_count = len(self.df_clean)
        
        # Check for duplicates
        duplicates = self.df_clean['rsID'].duplicated().sum()
        print(f"\n🔍 Found {duplicates:,} duplicate rsIDs")
        
        # Keep first occurrence
        self.df_clean = self.df_clean.drop_duplicates(subset=["rsID"])
        
        removed_count = before_count - len(self.df_clean)
        self._log_step(f"Removed {removed_count} duplicate entries", 
                      len(self.df_clean))
        
        return self
    
    def step8_ld_aware_selection(self, top_n=5000):
        """
        Step 8: LD-aware SNP selection.
        
        Note: True LD pruning requires a reference panel and PLINK.
        This method selects top N SNPs by p-value as an approximation.
        """
        print("\n" + "="*70)
        print(f"STEP 8: LD-AWARE SNP SELECTION")
        print("="*70)
        
        before_count = len(self.df_clean)
        
        print(f"\n⚠️  Important Limitation:")
        print(f"   True LD pruning requires:")
        print(f"   • Reference genotype panel (e.g., 1000 Genomes)")
        print(f"   • PLINK software")
        print(f"\n   For this academic project, we use:")
        print(f"   • Top-N SNP selection by p-value")
        print(f"   • This approximates LD pruning")
        
        if before_count > top_n:
            self.df_clean = self.df_clean.sort_values("pval").head(top_n)
            self._log_step(f"Selected top {top_n:,} SNPs by p-value", 
                          len(self.df_clean))
        else:
            print(f"\n   Dataset has {before_count:,} SNPs (< {top_n:,})")
            print(f"   Keeping all SNPs without further filtering")
            self._log_step(f"Keeping all {before_count:,} SNPs", len(self.df_clean))
        
        return self
    
    def step9_create_prs_weights(self):
        """
        Step 9: Create PRS weight file.
        """
        print("\n" + "="*70)
        print(f"STEP 9: CREATE PRS WEIGHT FILE")
        print("="*70)
        
        # Create PRS weights table
        prs_weights = self.df_clean[["rsID", "effect_allele", "beta"]].copy()
        prs_weights.columns = ["SNP", "Effect_Allele", "Beta"]
        
        # Save to file
        output_file = f"{self.output_dir}/prs_weights.txt"
        prs_weights.to_csv(output_file, sep="\t", index=False)
        
        self._log_step(f"Created PRS weight file: {output_file}")
        
        print(f"\n✅ Output format:")
        print(f"   Columns: SNP | Effect_Allele | Beta")
        print(f"   Total SNPs: {len(prs_weights):,}")
        print(f"\n   First 5 entries:")
        print(prs_weights.head().to_string(index=False))
        
        print(f"\n📌 This file can be used:")
        print(f"   • With PLINK for PRS calculation")
        print(f"   • As weights in ML feature engineering")
        print(f"   • For conceptual PRS integration")
        
        return self
    
    def step10_generate_report(self):
        """
        Step 10: Generate comprehensive quality control report.
        """
        print("\n" + "="*70)
        print(f"STEP 10: QUALITY CONTROL REPORT")
        print("="*70)
        
        # Save cleaned dataset
        output_file = f"{self.output_dir}/cleaned_gwas_summary_stats.csv"
        self.df_clean.to_csv(output_file, index=False)
        self._log_step(f"Saved cleaned dataset: {output_file}")
        
        # Save processing log
        log_file = f"{self.output_dir}/preprocessing_log.txt"
        with open(log_file, 'w') as f:
            f.write("GWAS PREPROCESSING LOG\n")
            f.write("=" * 70 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {self.input_file}\n")
            f.write("\n" + "\n".join(self.log))
        
        self._log_step(f"Saved processing log: {log_file}")
        
        # Generate summary statistics
        self._generate_summary_plots()
        
        # Print final summary
        print("\n" + "="*70)
        print("📊 PREPROCESSING SUMMARY")
        print("="*70)
        
        print(f"\n✅ Files Generated:")
        print(f"   1. {self.output_dir}/prs_weights.txt")
        print(f"   2. {self.output_dir}/cleaned_gwas_summary_stats.csv")
        print(f"   3. {self.output_dir}/preprocessing_log.txt")
        print(f"   4. {self.output_dir}/qc_plots.png")
        
        print(f"\n📈 Quality Metrics:")
        print(f"   Total SNPs processed: {len(self.df_clean):,}")
        print(f"   Chromosomes covered: {self.df_clean['chr'].nunique()}")
        print(f"   Mean |beta|: {self.df_clean['beta'].abs().mean():.4f}")
        print(f"   Mean p-value: {self.df_clean['pval'].mean():.2e}")
        
        return self
    
    def _generate_summary_plots(self):
        """
        Generate visualization plots for QC.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('GWAS Preprocessing Quality Control', fontsize=16, fontweight='bold')
        
        # Plot 1: Manhattan-like p-value distribution
        ax1 = axes[0, 0]
        self.df_clean['neg_log10_pval'] = -np.log10(self.df_clean['pval'])
        for chrom in sorted(self.df_clean['chr'].unique()):
            chrom_data = self.df_clean[self.df_clean['chr'] == chrom]
            ax1.scatter(chrom_data['pos'], chrom_data['neg_log10_pval'], 
                       s=5, alpha=0.6, label=f'Chr{chrom}' if chrom <= 3 else '')
        ax1.set_xlabel('Position (bp)', fontsize=10)
        ax1.set_ylabel('-log10(p-value)', fontsize=10)
        ax1.set_title('P-value Distribution Across Genome', fontsize=11)
        ax1.axhline(y=-np.log10(5e-8), color='r', linestyle='--', 
                   label='Genome-wide sig', linewidth=1)
        
        # Plot 2: Effect size distribution
        ax2 = axes[0, 1]
        ax2.hist(self.df_clean['beta'], bins=50, color='steelblue', 
                edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Effect Size (Beta)', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.set_title('Effect Size Distribution', fontsize=11)
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=1)
        
        # Plot 3: Allele frequency distribution
        ax3 = axes[1, 0]
        ax3.hist(self.df_clean['eaf'], bins=50, color='forestgreen', 
                edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Effect Allele Frequency', fontsize=10)
        ax3.set_ylabel('Frequency', fontsize=10)
        ax3.set_title('Allele Frequency Distribution', fontsize=11)
        
        # Plot 4: SNPs per chromosome
        ax4 = axes[1, 1]
        chr_counts = self.df_clean['chr'].value_counts().sort_index()
        ax4.bar(chr_counts.index.astype(str), chr_counts.values, 
               color='coral', edgecolor='black', alpha=0.7)
        ax4.set_xlabel('Chromosome', fontsize=10)
        ax4.set_ylabel('Number of SNPs', fontsize=10)
        ax4.set_title('SNP Distribution Across Chromosomes', fontsize=11)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_file = f"{self.output_dir}/qc_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\n✅ Generated QC plots: {plot_file}")
        plt.close()
    
    def run_full_pipeline(self, maf_threshold=0.01, pval_threshold=1e-5, 
                         strategy="PRS", top_n=5000):
        """
        Run the complete preprocessing pipeline.
        
        Parameters:
        -----------
        maf_threshold : float
            Minor allele frequency threshold (default: 0.01)
        pval_threshold : float
            P-value threshold (default: 1e-5)
        strategy : str
            'GWAS' or 'PRS' (default: 'PRS')
        top_n : int
            Maximum number of SNPs to retain (default: 5000)
        """
        print("\n" + "="*70)
        print("🧬 GWAS SUMMARY STATISTICS PREPROCESSING PIPELINE")
        print("="*70)
        print(f"Dataset: DIAMANTE South Asian T2D GWAS")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        try:
            (self
             .step1_load_data()
             .step2_rename_columns()
             .step3_remove_ambiguous_snps()
             .step4_maf_filtering(maf_threshold=maf_threshold)
             .step5_significance_filtering(pval_threshold=pval_threshold, 
                                          strategy=strategy)
             .step6_effect_size_qc()
             .step7_deduplication()
             .step8_ld_aware_selection(top_n=top_n)
             .step9_create_prs_weights()
             .step10_generate_report())
            
            print("\n" + "="*70)
            print("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            raise
        
        return self


def main():
    """
    Main execution function.
    """
    # Configuration
    INPUT_FILE = "Dataset/DIAMANTE-SAS.sumstat.csv"
    OUTPUT_DIR = "processed_data"
    
    # Parameters (can be adjusted)
    MAF_THRESHOLD = 0.01      # Remove variants with MAF < 1%
    PVAL_THRESHOLD = 1e-5     # PRS-optimized significance threshold
    STRATEGY = "PRS"          # "PRS" or "GWAS"
    TOP_N_SNPS = 5000         # Maximum number of SNPs to retain
    
    # Initialize and run preprocessor
    preprocessor = GWASPreprocessor(
        input_file=INPUT_FILE,
        output_dir=OUTPUT_DIR
    )
    
    preprocessor.run_full_pipeline(
        maf_threshold=MAF_THRESHOLD,
        pval_threshold=PVAL_THRESHOLD,
        strategy=STRATEGY,
        top_n=TOP_N_SNPS
    )
    
    # Methodology text for paper
    print("\n" + "="*70)
    print("📝 FOR YOUR METHODOLOGY SECTION (COPY THIS)")
    print("="*70)
    print("""
GWAS summary statistics were preprocessed by removing ambiguous and rare 
variants, filtering SNPs based on minor allele frequency (MAF > 0.01) and 
statistical significance (p < 1×10⁻⁵), and retaining independent, 
high-confidence variants. Effect sizes from the filtered variants were used 
to construct polygenic risk representations for downstream machine learning 
analysis. Quality control included removal of strand-ambiguous SNPs (A/T, C/G), 
exclusion of extreme effect sizes, and deduplication. The final set comprised 
the top 5,000 SNPs ranked by p-value, approximating LD-pruned markers for 
PRS construction.
    """)
    
    print("\n" + "="*70)
    print("🎯 NEXT STEPS")
    print("="*70)
    print("""
1. ✅ You now have clean, filtered SNP weights
2. 🔄 Next: Integrate with clinical data
3. 🤖 Then: Build ML/DL models with PRS features
4. 📊 Finally: Evaluate predictive performance

Key files to use:
• processed_data/prs_weights.txt → For PRS calculation
• processed_data/cleaned_gwas_summary_stats.csv → Full cleaned data
    """)


if __name__ == "__main__":
    main()
