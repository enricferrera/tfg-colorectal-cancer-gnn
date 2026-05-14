# ==================================================================
# SCRIPT TO ACCESS PATIENT DATA FOR OFFLINE GRAPH CREATION
# ==================================================================
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from pathlib import Path
import time


def euclidean_knn_graph_creation(patient_dict, k=8):
    """
    Iterates through all the patients, creates a graph for each patient using
    KNN on CLS features, and saves each graph to disk.

    Args:
        patient_dict (dict): Dictionary organized by patient ID containing 
                             labels, megapatches, and histodata.
        k (int): Number of neighbors for the KNN graph.
    """

    # --- 1. Setup the Output Directory ---
    graph_dir = Path(__file__).parent / "graphs" / "euclidean" / f"k_{k}"
    graph_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    print(f"Starting knn graph creation for {len(patient_dict)} patients...")

    # --- 2. Loop and Create a Graph for Each Patient ---
    total_patients = len(patient_dict)
    for i, (patient_id, patient_data) in enumerate(patient_dict.items()):
        
        # Print progress on the same line
        print(f"\rProcessing patient {i+1}/{total_patients} ({patient_id})...", end="", flush=True)

        # Extract patch-level features and coordinates
        node_features = torch.stack([m['features'] for m in patient_data['megapatches']])

        graph_label = patient_data['label']
        histodata = patient_data['histodata']

        # Create graph edges based on the K-Nearest Neighbors in the FEATURE space
        # Note: knn_graph computes directed edges from k-nearest neighbors.
        edge_index = knn_graph(node_features, k=k, loop=True)

        # Assemble the graph data object
        graph = Data(
            x=node_features,
            edge_index=edge_index,
            y=torch.tensor([graph_label]),
        )
        graph.histodata = histodata
        graph.patient_id = patient_id

        # --- 3. Save the Graph to Disk ---
        torch.save(graph, graph_dir / f"{patient_id}.pt")

    end_time = time.time()
    duration = end_time - start_time
    print("-" * 50)
    print(f"KNN graph creation complete.")
    print(f"Saved {len(list(graph_dir.glob('*.pt')))} graphs to '{graph_dir}' directory.")
    print(f"Total time: {duration // 60:.0f}m {duration % 60:.0f}s")
    print("-" * 50)
