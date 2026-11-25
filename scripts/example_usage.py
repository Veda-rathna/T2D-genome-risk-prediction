"""
Example Usage: GWAS Preprocessing Pipeline
===========================================

This script demonstrates how to use the GWASPreprocessor class
with different configurations.
"""

from gwas_preprocessing import GWASPreprocessor

# ============================================================================
# EXAMPLE 1: Basic Usage with Default Parameters
# ============================================================================

def example_basic():
    """Run preprocessing with default PRS-optimized settings."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage (PRS-Optimized)")
    print("="*70)
    
    preprocessor = GWASPreprocessor(
        input_file="Dataset/DIAMANTE-SAS.sumstat.csv",
        output_dir="processed_data"
    )
    
    # Run with defaults: MAF>0.01, p<1e-5, top 5000 SNPs
    preprocessor.run_full_pipeline()


# ============================================================================
# EXAMPLE 2: Strict GWAS Significance
# ============================================================================

def example_strict_gwas():
    """Run preprocessing with genome-wide significance threshold."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Strict GWAS Significance")
    print("="*70)
    
    preprocessor = GWASPreprocessor(
        input_file="Dataset/DIAMANTE-SAS.sumstat.csv",
        output_dir="processed_data_strict"
    )
    
    # Strict genome-wide significance
    preprocessor.run_full_pipeline(
        maf_threshold=0.01,
        pval_threshold=5e-8,  # Genome-wide significance
        strategy="GWAS",
        top_n=10000
    )


# ============================================================================
# EXAMPLE 3: Liberal Filtering for Discovery
# ============================================================================

def example_liberal():
    """Run preprocessing with more liberal thresholds for discovery."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Liberal Filtering")
    print("="*70)
    
    preprocessor = GWASPreprocessor(
        input_file="Dataset/DIAMANTE-SAS.sumstat.csv",
        output_dir="processed_data_liberal"
    )
    
    # More lenient thresholds
    preprocessor.run_full_pipeline(
        maf_threshold=0.005,   # Include rarer variants
        pval_threshold=1e-4,   # More liberal p-value
        strategy="PRS",
        top_n=10000           # Keep more SNPs
    )


# ============================================================================
# EXAMPLE 4: Step-by-Step with Custom Logic
# ============================================================================

def example_stepwise():
    """Run preprocessing step-by-step with custom filtering between steps."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Step-by-Step Custom Processing")
    print("="*70)
    
    preprocessor = GWASPreprocessor(
        input_file="Dataset/DIAMANTE-SAS.sumstat.csv",
        output_dir="processed_data_custom"
    )
    
    # Run steps individually
    preprocessor.step1_load_data()
    preprocessor.step2_rename_columns()
    preprocessor.step3_remove_ambiguous_snps()
    preprocessor.step4_maf_filtering(maf_threshold=0.01)
    
    # Custom filtering: Keep only chromosomes 1-10
    print("\n🔧 Custom filter: Keeping only chromosomes 1-10...")
    preprocessor.df = preprocessor.df[preprocessor.df['chr'] <= 10]
    print(f"   SNPs remaining: {len(preprocessor.df):,}")
    
    # Continue with remaining steps
    preprocessor.step5_significance_filtering(pval_threshold=1e-5, strategy="PRS")
    preprocessor.step6_effect_size_qc()
    preprocessor.step7_deduplication()
    preprocessor.step8_ld_aware_selection(top_n=3000)
    preprocessor.step9_create_prs_weights()
    preprocessor.step10_generate_report()


# ============================================================================
# EXAMPLE 5: Conservative for Clinical Use
# ============================================================================

def example_conservative():
    """Run preprocessing with conservative settings for clinical applications."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Conservative for Clinical Use")
    print("="*70)
    
    preprocessor = GWASPreprocessor(
        input_file="Dataset/DIAMANTE-SAS.sumstat.csv",
        output_dir="processed_data_clinical"
    )
    
    # Conservative thresholds
    preprocessor.run_full_pipeline(
        maf_threshold=0.05,    # Common variants only
        pval_threshold=5e-8,   # Genome-wide significance
        strategy="GWAS",
        top_n=1000            # Small, robust set
    )


# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    """Main menu for selecting example to run."""
    print("\n" + "="*70)
    print("GWAS PREPROCESSING - EXAMPLE USAGE")
    print("="*70)
    print("\nSelect an example to run:\n")
    print("1. Basic Usage (PRS-Optimized) [RECOMMENDED]")
    print("2. Strict GWAS Significance")
    print("3. Liberal Filtering for Discovery")
    print("4. Step-by-Step Custom Processing")
    print("5. Conservative for Clinical Use")
    print("0. Exit")
    
    choice = input("\nEnter your choice (0-5): ").strip()
    
    examples = {
        '1': example_basic,
        '2': example_strict_gwas,
        '3': example_liberal,
        '4': example_stepwise,
        '5': example_conservative
    }
    
    if choice == '0':
        print("\nExiting...")
        return
    
    if choice in examples:
        examples[choice]()
    else:
        print("\n❌ Invalid choice. Please select 0-5.")
        main()


if __name__ == "__main__":
    # For quick start, just run the basic example
    # Uncomment the line below to use the interactive menu instead
    
    example_basic()  # Default: Run basic example
    # main()         # Alternative: Show interactive menu
