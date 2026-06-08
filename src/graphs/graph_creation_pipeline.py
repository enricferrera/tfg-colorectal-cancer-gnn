"""
Pipeline for graph creation.
@author: Enric Ferrera González
"""
# Standard library imports
from pathlib import Path
import torch
import os

# Local application imports
from utils.seed import fix_seeds
from dataset.load_cls import load_cls_metadata, patient_dict_builder
from graphs import (
    radius_graph_creation_cpu, 
    euclidean_knn_graph_creation_cpu, 
    cosine_knn_graph_creation_cpu, 
    fully_connected_graph_creation_cpu
)

# ------ GLOBAL CONFIGURATION -------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
NUM_WORKERS = 10  # Optimized for 12-core CPU

# Keep in case we add non-deterministic graph creation
r_seed = 123
fix_seeds(r_seed=r_seed)

# Base paths
npz_path = Path(__file__).resolve().parents[2] / "data" / "NEW_DATASET_cls_2048"

def main():
    # Load metadata
    print(f"Loading metadata from {npz_path}...")
    pat_nx_dict, pat_histodata_dict, features, affectation, hospitals, patients, slides, coords = load_cls_metadata(npz_path)

    print('------ filtering out patients without diagnosis...-------')
    patient_dict = patient_dict_builder(features, affectation, hospitals, patients, slides, coords, pat_nx_dict, pat_histodata_dict)

    print('------ pre-stacking and saving features for speed and space...-------')
    features_dir = Path(__file__).resolve().parent / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    for pat_id in patient_dict:
        stacked = torch.stack([m['features'] for m in patient_dict[pat_id]['megapatches']])
        patient_dict[pat_id]['features_stacked'] = stacked
        # Save features once here
        torch.save(stacked, features_dir / f"{pat_id}.pt")
    
    print(f"Saved global features to {features_dir}")

    # --- Graph creation ---

    # 1. Radius Graphs (60 to 80, step 4)
    for r in range(60, 81, 4):
        try:
            print(f"\n[r={r}] Creating Radius Graphs...")
            radius_graph_creation_cpu(patient_dict, r=r, num_workers=NUM_WORKERS)
        except Exception as e:
            print(f"\nFAILED Radius r={r}: {e}")

    # 2. KNN Graphs (Euclidean & Cosine)
    # k from 8 to 20 (step 2), plus 50 and 100
    k_values = list(range(8, 21, 2)) + [50, 100]
    for k in k_values:
        # Euclidean
        try:
            print(f"\n[k={k}] Creating Euclidean KNN Graphs...")
            euclidean_knn_graph_creation_cpu(patient_dict, k=k, num_workers=NUM_WORKERS)
        except Exception as e:
            print(f"\nFAILED Euclidean k={k}: {e}")

        # Cosine
        try:
            print(f"\n[k={k}] Creating Cosine KNN Graphs...")
            cosine_knn_graph_creation_cpu(patient_dict, k=k, num_workers=NUM_WORKERS)
        except Exception as e:
            print(f"\nFAILED Cosine k={k}: {e}")

    # 3. Fully Connected Graphs
    try:
        print("\nCreating Fully Connected Graphs...")
        fully_connected_graph_creation_cpu(patient_dict, num_workers=NUM_WORKERS)
    except Exception as e:
        print(f"\nFAILED Fully Connected: {e}")

if __name__ == '__main__':
    main()
