# GWAS Preprocessing Pipeline

## 📋 Overview

This pipeline preprocesses GWAS summary statistics from the DIAMANTE South Asian T2D study to prepare them for polygenic risk score (PRS) construction and machine learning analysis.

## 🎯 What This Script Does

### The 10-Step Preprocessing Pipeline:

1. **Load & Sanity Checks** - Validates file structure and removes broken rows
2. **Column Renaming** - Creates ML-friendly column names
3. **Remove Ambiguous SNPs** - Filters out A/T and C/G pairs (strand ambiguity)
4. **MAF Filtering** - Removes rare variants (default: MAF < 0.01)
5. **Significance Filtering** - Selects informative SNPs (default: p < 1e-5)
6. **Effect Size QC** - Removes extreme beta values (artifacts)
7. **Deduplication** - Ensures each SNP appears once
8. **LD-Aware Selection** - Selects top N SNPs by p-value (approximates LD pruning)
9. **Create PRS Weights** - Generates weight file for PRS calculation
10. **Generate Report** - Creates logs and QC visualizations

## 🚀 Quick Start

### Prerequisites

```powershell
# Install required packages
pip install pandas numpy matplotlib seaborn
```

### Run the Pipeline

```powershell
# Navigate to project directory
cd d:\Code_wid_pablo\Genome_T2D\T2D-genome-risk-prediction

# Run the preprocessing script
python gwas_preprocessing.py
```

## 📊 Output Files

After running, you'll find these files in the `processed_data/` directory:

1. **`prs_weights.txt`** - SNP weights for PRS calculation
   - Format: SNP | Effect_Allele | Beta
   - Use with PLINK or in ML models

2. **`cleaned_gwas_summary_stats.csv`** - Full cleaned dataset
   - All QC-passed SNPs with complete information

3. **`preprocessing_log.txt`** - Detailed processing log
   - Timestamped steps with SNP counts

4. **`qc_plots.png`** - Quality control visualizations
   - P-value distribution (Manhattan-like)
   - Effect size distribution
   - Allele frequency distribution
   - SNPs per chromosome

## ⚙️ Configuration

You can adjust parameters in the script's `main()` function:

```python
# In gwas_preprocessing.py, modify these values:

MAF_THRESHOLD = 0.01      # Minor allele frequency threshold
PVAL_THRESHOLD = 1e-5     # P-value threshold (1e-5 for PRS, 5e-8 for GWAS)
STRATEGY = "PRS"          # "PRS" or "GWAS"
TOP_N_SNPS = 5000         # Maximum SNPs to retain
```

### Strategy Options:

- **`"PRS"` (Recommended)**: p < 1e-5
  - Captures many small effects
  - Better for prediction models
  - Improves AUC performance

- **`"GWAS"` (Strict)**: p < 5e-8
  - Genome-wide significance
  - Smaller, stronger SNP set
  - Better for interpretation

## 📈 Expected Results

### Input Dataset:
- ~10-20 million SNPs (typical GWAS)

### After Preprocessing:
- ~5,000-10,000 high-quality SNPs
- Ready for ML integration

### Processing Time:
- ~5-15 minutes (depending on file size and system)

## 🔬 What You Get

After preprocessing, you have:

✅ **Clean SNP list** - QC-passed variants  
✅ **Reliable effect sizes** - Artifact-free betas  
✅ **PRS-ready weights** - Formatted for scoring  
✅ **Population-scale genetic knowledge** - DIAMANTE consortium data  

## 🎓 For Your Paper

### Methodology Section (Copy This):

```
GWAS summary statistics were preprocessed by removing ambiguous and rare 
variants, filtering SNPs based on minor allele frequency (MAF > 0.01) and 
statistical significance (p < 1×10⁻⁵), and retaining independent, 
high-confidence variants. Effect sizes from the filtered variants were used 
to construct polygenic risk representations for downstream machine learning 
analysis. Quality control included removal of strand-ambiguous SNPs (A/T, C/G), 
exclusion of extreme effect sizes, and deduplication. The final set comprised 
the top 5,000 SNPs ranked by p-value, approximating LD-pruned markers for 
PRS construction.
```

## 🔄 Next Steps in Your Pipeline

```
GWAS summary stats
   ↓ (THIS SCRIPT)
Filtered SNP weights
   ↓
PRS feature engineering
   ↓
Clinical dataset integration
   ↓
ML / DL model training
   ↓
Prediction & Evaluation
```

## ⚠️ Important Notes

### What This Script DOES:
- ✅ Filters and cleans GWAS summary statistics
- ✅ Creates PRS weight files
- ✅ Performs quality control
- ✅ Generates reports and visualizations

### What This Script DOES NOT DO:
- ❌ Calculate individual PRS scores (requires genotype data)
- ❌ Perform true LD pruning (requires reference panel + PLINK)
- ❌ Train ML models (that's the next stage)

### LD Pruning Limitation:
This script approximates LD pruning by selecting top SNPs by p-value. For true LD pruning:
- Need: 1000 Genomes reference panel (South Asian population)
- Tool: PLINK with `--clump` or `--indep-pairwise`
- Acceptable for academic projects to use top-N approximation

## 🐛 Troubleshooting

### Error: "Files above 50MB cannot be synchronized"
- This is normal for GWAS files
- The script handles large files correctly
- Processing happens in chunks

### Memory Error
```powershell
# If you run out of memory, reduce TOP_N_SNPS:
TOP_N_SNPS = 3000  # Instead of 5000
```

### Missing Columns Error
- Verify your CSV has all required columns
- Check column names match DIAMANTE format

## 📚 References

### DIAMANTE Consortium:
- Mahajan, A., et al. (2022). "Multi-ancestry genetic study of type 2 diabetes highlights the power of diverse populations for discovery and translation."
- Nature Genetics, 54, 560-572.

### Methods:
- Choi, S.W., et al. (2020). "Tutorial: a guide to performing polygenic risk score analyses."
- Nature Protocols, 15, 2759-2772.

## 💡 Tips

1. **First Run**: Use default parameters
2. **Check QC Plots**: Verify distributions look reasonable
3. **Adjust Parameters**: Based on your specific needs
4. **Document Changes**: Keep track of parameter choices for methods section

## 📞 Support

For issues specific to:
- **This script**: Check code comments and error messages
- **DIAMANTE data**: See consortium documentation
- **GWAS methods**: Consult GWAS textbooks/tutorials

## ✅ Validation Checklist

Before proceeding to ML stage:

- [ ] Script ran without errors
- [ ] Output files generated in `processed_data/`
- [ ] QC plots look reasonable
- [ ] Log file shows expected SNP counts
- [ ] PRS weights file has 3 columns (SNP, Effect_Allele, Beta)
- [ ] Cleaned dataset saved successfully

---

**Version**: 1.0  
**Last Updated**: December 2025  
**Author**: T2D Genome Risk Prediction Project
