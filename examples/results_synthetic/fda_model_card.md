
# FDA Model Card: Virtual Cell AI Precision Oncology
    
## 1. INTENDED USE
Predict cancer progression/response using mechanistic multi-pathway modeling.

## 2. PERFORMANCE (TCGA BRCA n=75)  
| Endpoint | AUC | Sensitivity | Specificity |
|----------|-----|-------------|-------------|
| Progression | 0.942 | TBD | TBD |
| Response | 0.957 | TBD | TBD |

## 3. KEY BIOMARKERS (Prospective Validation Targets)
| type       | feature    | clinical_note             |
|:-----------|:-----------|:--------------------------|
| predictive | cycE_max   | 66.7% response difference |
| predictive | cycE_final | 66.7% response difference |
| predictive | p53_mean   | 65.3% response difference |
| predictive | p53_auc    | 65.3% response difference |
| predictive | cycE_mean  | 65.3% response difference |

## 4. PROSPECTIVE STUDY DESIGN
- Design: Multi-center, observational
- N=500 patients (2 hospitals)  
- Primary: Progression-free survival (C-index ≥0.75)
- Timeline: Q1-Q3 2026

## 5. HEALTH ECONOMICS
Cost/test: $500 | Wrong treatment avoided: $50K
Expected savings: $12K/patient | Cost/QALY: $25K (<$50K threshold)

## 6. LIMITATIONS
Synthetic data training | External validation pending
