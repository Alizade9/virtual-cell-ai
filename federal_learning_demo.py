"""
Federated Learning Simulation
privacy-preserving multi-site approach
"""

import numpy as np
import pandas as pd
from enhanced_pipeline import train_multitask_model, generate_enhanced_patient_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

print("=" * 80)
print("FEDERATED LEARNING SIMULATION")
print("Privacy-Preserving Multi-Site Training")
print("=" * 80)

# Simulate 5 hospital sites
print("\n[1/4] Simulating 5 hospital sites...")
hospitals = [
    {'name': 'Hospital A (Boston)', 'patients': 80},
    {'name': 'Hospital B (Paris)', 'patients': 60},
    {'name': 'Hospital C (London)', 'patients': 70},
    {'name': 'Hospital D (Tokyo)', 'patients': 50},
    {'name': 'Hospital E (Sydney)', 'patients': 40}
]

total_patients = sum(h['patients'] for h in hospitals)
print(f"✓ Total patients across sites: {total_patients}")
for h in hospitals:
    print(f"  • {h['name']}: {h['patients']} patients")

# Generate data per site
print("\n[2/4] Training local models (data never leaves hospitals)...")
local_models = []
local_aucs = []

for i, hospital in enumerate(hospitals):
    # Each hospital has its own data
    site_data = generate_enhanced_patient_data(n_patients=hospital['patients'])
    
    # Simulate patients
    from tcga_validation import all_features, simulate_patient_with_drug, extract_integrated_features
    site_features = []
    for _, patient in site_data.iterrows():
        sim = simulate_patient_with_drug(patient, simulation_time=20)
        features = extract_integrated_features(sim)
        combined = {'patient_id': patient['patient_id'], **features, 
                   **{col: patient[col] for col in patient.index if col != 'patient_id'}}
        site_features.append(combined)
    
    ml_df_site = pd.DataFrame(site_features)
    
    # Train local model
    results, feature_cols, _, _, _ = train_multitask_model(ml_df_site)
    local_model = results['progression']['model']
    local_auc = results['progression']['auc']
    
    local_models.append(local_model)
    local_aucs.append(local_auc)
    
    print(f"  ✓ {hospital['name']}: Local AUC = {local_auc:.3f}")

# Federated averaging (Owkin's approach)
print("\n[3/4] Aggregating models (Federated Averaging)...")

# Average model parameters across sites
class FederatedModel:
    def __init__(self, local_models, weights=None):
        self.local_models = local_models
        self.weights = weights if weights else [1/len(local_models)] * len(local_models)
    
    def predict_proba(self, X):
        # Average predictions from all local models
        predictions = np.array([model.predict_proba(X) for model in self.local_models])
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        return weighted_pred

# Weight by site size
weights = [h['patients']/total_patients for h in hospitals]
federated_model = FederatedModel(local_models, weights)

# Test on held-out data
test_data = generate_enhanced_patient_data(n_patients=100)
test_features = []
for _, patient in test_data.iterrows():
    sim = simulate_patient_with_drug(patient, simulation_time=20)
    features = extract_integrated_features(sim)
    combined = {'patient_id': patient['patient_id'], **features,
               **{col: patient[col] for col in patient.index if col != 'patient_id'}}
    test_features.append(combined)

ml_df_test = pd.DataFrame(test_features)
X_test = ml_df_test[[col for col in ml_df_test.columns 
                     if col not in ['patient_id', 'disease_progression', 
                                   'treatment_response', 'survival_months', 'censored']]]
y_test = ml_df_test['disease_progression']

federated_pred = federated_model.predict_proba(X_test)[:, 1]
federated_auc = roc_auc_score(y_test, federated_pred)

print(f"✓ Federated Model AUC: {federated_auc:.3f}")

# Compare to centralized
print("\n[4/4] Comparing federated vs. centralized...")
centralized_data = generate_enhanced_patient_data(n_patients=total_patients)
centralized_features = []
for _, patient in centralized_data.iterrows():
    sim = simulate_patient_with_drug(patient, simulation_time=20)
    features = extract_integrated_features(sim)
    combined = {'patient_id': patient['patient_id'], **features,
               **{col: patient[col] for col in patient.index if col != 'patient_id'}}
    centralized_features.append(combined)

ml_df_central = pd.DataFrame(centralized_features)
results_central, _, _, _, _ = train_multitask_model(ml_df_central)
centralized_auc = results_central['progression']['auc']

print(f"✓ Centralized Model AUC: {centralized_auc:.3f}")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Site performance
ax1.bar(range(len(hospitals)), local_aucs, color='steelblue', alpha=0.7)
ax1.axhline(federated_auc, color='green', linestyle='--', linewidth=2, label=f'Federated: {federated_auc:.3f}')
ax1.axhline(centralized_auc, color='red', linestyle='--', linewidth=2, label=f'Centralized: {centralized_auc:.3f}')
ax1.set_xticks(range(len(hospitals)))
ax1.set_xticklabels([h['name'].split('(')[0] for h in hospitals], rotation=45, ha='right')
ax1.set_ylabel('AUC')
ax1.set_title('Performance by Hospital Site', fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3, axis='y')

# Comparison
approaches = ['Local\n(avg)', 'Federated', 'Centralized']
aucs = [np.mean(local_aucs), federated_auc, centralized_auc]
colors = ['steelblue', 'green', 'red']
ax2.bar(approaches, aucs, color=colors, alpha=0.7)
ax2.set_ylabel('AUC')
ax2.set_ylim([0.7, 1.0])
ax2.set_title('Federated vs. Centralized', fontweight='bold')
ax2.grid(alpha=0.3, axis='y')

for i, v in enumerate(aucs):
    ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('federated_learning_results.png', dpi=150, bbox_inches='tight')
plt.close()

# Save results
pd.DataFrame([
    {'approach': 'Local (average)', 'auc': np.mean(local_aucs), 'n_sites': len(hospitals)},
    {'approach': 'Federated', 'auc': federated_auc, 'n_sites': len(hospitals)},
    {'approach': 'Centralized', 'auc': centralized_auc, 'n_sites': 1}
]).to_csv('federated_learning_results.csv', index=False)

print("\n" + "=" * 80)
print("FEDERATED LEARNING SUMMARY")
print("=" * 80)
print(f"\n✓ Privacy-Preserving: No patient data shared between sites")
print(f"✓ Sites: {len(hospitals)} hospitals across 5 locations")
print(f"✓ Total patients: {total_patients}")
print(f"\nPerformance:")
print(f"  • Local models (avg): {np.mean(local_aucs):.3f} AUC")
print(f"  • Federated model:    {federated_auc:.3f} AUC")
print(f"  • Centralized model:  {centralized_auc:.3f} AUC")
print(f"  • Performance gap:    {abs(federated_auc - centralized_auc):.3f} ({abs(federated_auc - centralized_auc)*100:.1f}%)")
print(f"\n✓ Results saved: federated_learning_results.csv")
print(f"✓ Visualization: federated_learning_results.png")
print("=" * 80)
