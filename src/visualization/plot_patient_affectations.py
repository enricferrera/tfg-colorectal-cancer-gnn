import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
current_dir = Path(__file__).resolve().parent
if str(current_dir.parent) not in sys.path:
    sys.path.append(str(current_dir.parent))

from dataset.load_cls import load_cls_metadata, patient_dict_builder

def plot_affectations():
    print("--- GENERATING AFFECTATION SUMMARY ---")
    npz_path = current_dir.parent.parent / "data" / "NEW_DATASET_cls_2048"
    
    if not npz_path.exists():
        print(f"ERROR: Data folder not found at {npz_path}")
        return

    # Load data
    results = load_cls_metadata(npz_path)
    patient_dict = patient_dict_builder(results[2], results[3], results[4], results[5], results[6], results[7], results[0], results[1])

    summary_data = []
    
    # Create results directory
    output_dir = current_dir / "results" / "affectation_study"
    output_dir.mkdir(parents=True, exist_ok=True)

    for pid, data in patient_dict.items():
        affs = np.array([mp['affectation'].item() if isinstance(mp['affectation'], torch.Tensor) else mp['affectation'] for mp in data['megapatches']])
        
        summary_data.append({
            'Patient_ID': pid,
            'Label': 'N1' if data['label'] == 1 else 'N0',
            'Total_Patches': len(affs),
            'Infiltrated_Patches': np.sum(affs > 0),
            'Max_Aff': affs.max(),
            'Mean_Aff': affs.mean()
        })

    df = pd.DataFrame(summary_data)
    
    # Save CSV
    df = df.sort_values(by=['Label', 'Infiltrated_Patches'], ascending=[False, False])
    df.to_csv(output_dir / "patient_affectation_summary.csv", index=False)
    
    print("\n--- CSV SAMPLE (First 20 rows) ---")
    print(df.head(20).to_string(index=False))
    
    print(f"\nFull CSV saved to: {output_dir / 'patient_affectation_summary.csv'}")
    
    # Also generate a plot for quick reference
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Max_Aff', hue='Label', bins=30, kde=True)
    plt.title('Distribution of Max Affectation per Patient')
    plt.xlabel('Max Affectation Value')
    plt.ylabel('Patient Count')
    plt.savefig(output_dir / "max_affectation_distribution.png")

if __name__ == "__main__":
    plot_affectations()
