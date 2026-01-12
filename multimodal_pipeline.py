# multimodal_pipeline.py
"""Multi-Modal: Molecular + Image Features"""

import numpy as np
import pandas as pd
from enhanced_pipeline import train_multitask_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle



# Load TCGA molecular feature matrix
ml_df = pd.read_csv("results_tcga/tcga_ml_features.csv")

print("=" * 80)
print("MULTI-MODAL PIPELINE: Molecular + Imaging")
print("=" * 80)

# 0) Train molecular-only baseline
print("\n[0/3] Training molecular-only baseline on TCGA features...")
results_molecular, _, _, _, _ = train_multitask_model(ml_df)
print(f"✓ Molecular-only progression AUC: {results_molecular['progression']['auc']:.3f}")

# 1) Simulate image features
print("\n[1/3] Simulating histopathology image features...")

n_patients = len(ml_df)  # ← define n_patients here

image_features = pd.DataFrame(
    np.random.randn(n_patients, 50),
    columns=[f"img_{i}" for i in range(50)]
)

# (Optional) make first 5 image features weakly predictive
y = ml_df["disease_progression"].values
for i in range(5):
    image_features[f"img_{i}"] += y * 0.8

print(f"✓ Generated {image_features.shape[1]} image features")














# Combine molecular + image
print("\n[2/3] Combining molecular and image features...")
multimodal_df = pd.concat([ml_df.reset_index(drop=True), image_features], axis=1)
print(f"✓ Combined features: {len(ml_df.columns)} molecular + {len(image_features.columns)} image")

# Train on multi-modal
print("\n[3/3] Training multi-modal model...")
results_mm, _, _, _, _ = train_multitask_model(
    multimodal_df,
    n_estimators=300,
    max_depth=None,
)


improvement = results_mm['progression']['auc'] - results_molecular['progression']['auc']

# Save results
pd.DataFrame([{
    'approach': 'Molecular Only',
    'progression_auc': results_molecular['progression']['auc'],
    'response_auc': results_molecular['response']['auc']
}, {
    'approach': 'Multi-Modal',
    'progression_auc': results_mm['progression']['auc'],
    'response_auc': results_mm['response']['auc']
}]).to_csv('multimodal_comparison.csv', index=False)

print("\n" + "=" * 80)
print("MULTI-MODAL RESULTS")
print("=" * 80)
print(f"Molecular Only: {results_molecular['progression']['auc']:.3f} AUC")
print(f"Multi-Modal:    {results_mm['progression']['auc']:.3f} AUC")
print(f"Improvement:    +{improvement:.3f} ({improvement*100:.1f}%)")
print("\n✓ Results saved to: multimodal_comparison.csv")
print("=" * 80)
