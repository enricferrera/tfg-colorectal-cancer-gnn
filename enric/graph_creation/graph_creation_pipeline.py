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
npz_path = Path(__file__).resolve().parents[2] / "Data" / "cls_ALL"

def main():
    # Load metadata
    print(f"Loading metadata from {npz_path}...")
    pat_nx_dict, pat_histodata_dict, features, affectation, hospitals, patients, slides, coords = load_cls_metadata(npz_path)

    print('------ filtering out patients without diagnosis...-------')
    patient_dict = patient_dict_builder(features, affectation, hospitals, patients, slides, coords, pat_nx_dict, pat_histodata_dict)

    # --- Graph creation ---
    # Uncomment the ones you want to run
    
    # print("\nCreating Euclidean KNN Graphs (k=5)...")
    # euclidean_knn_graph_creation(patient_dict, 1)

    #print("\nCreating Cosine KNN Graphs (k=10)...")
    #cosine_knn_graph_creation(patient_dict, 10)

    #print("\nCreating Cosine KNN Graphs (k=50)...")
    #cosine_knn_graph_creation(patient_dict, 50)

    #print("\nCreating Cosine KNN Graphs (k=100)...")
    #cosine_knn_graph_creation(patient_dict, 100)

    # print("\nCreating Radius Graphs (r=120)...")
    # radius_graph_creation(patient_dict, 120)

    print("\nCreating Fully Connected Graphs...")
    fully_connected_graph_creation(patient_dict)

if __name__ == '__main__':
    main()
