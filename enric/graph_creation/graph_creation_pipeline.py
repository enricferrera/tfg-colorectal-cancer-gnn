"""
Pipeline for graph creation.
@author: Enric Ferrera González
"""
# Standard library imports
from pathlib import Path
import torch
import os

# Local application imports
from enric import fix_seeds, load_cls_metadata, patient_dict_builder
from enric.graph_creation import fully_connected_graph_creation, euclidean_knn_graph_creation, radius_graph_creation, cosine_knn_graph_creation

# ------ GLOBAL CONFIGURATION -------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Keep in case we add non-deterministic graph creation
r_seed = 123
fix_seeds(r_seed=r_seed)

# Base paths
npz_path = Path(__file__).resolve().parents[2] / "Data" / "NEW_DATASET_cls_2048"

def main():
    # Load metadata
    print(f"Loading metadata from {npz_path}...")
    pat_nx_dict, pat_histodata_dict, features, affectation, hospitals, patients, slides, coords = load_cls_metadata(npz_path)

    print('------ filtering out patients without diagnosis...-------')
    patient_dict = patient_dict_builder(features, affectation, hospitals, patients, slides, coords, pat_nx_dict, pat_histodata_dict)

    # --- Graph creation ---

    # 1. Radius Graphs (60 to 80)
    for r in range(60, 81):
        try:
            print(f"\n[{r}/80] Creating Radius Graphs (r={r})...")
            radius_graph_creation(patient_dict, r=r)
        except Exception as e:
            print(f"\nFAILED Radius r={r}: {e}")

    # 2. KNN Graphs (Euclidean & Cosine)
    k_values = list(range(8, 41)) + [50, 100]
    for k in k_values:
        # Euclidean
        try:
            print(f"\n[k={k}] Creating Euclidean KNN Graphs...")
            euclidean_knn_graph_creation(patient_dict, k=k)
        except Exception as e:
            print(f"\nFAILED Euclidean k={k}: {e}")

        # Cosine
        try:
            print(f"\n[k={k}] Creating Cosine KNN Graphs...")
            cosine_knn_graph_creation(patient_dict, k=k)
        except Exception as e:
            print(f"\nFAILED Cosine k={k}: {e}")

    # 3. Fully Connected Graphs
    try:
        print("\nCreating Fully Connected Graphs...")
        fully_connected_graph_creation(patient_dict)
    except Exception as e:
        print(f"\nFAILED Fully Connected: {e}")

if __name__ == '__main__':
    main()
