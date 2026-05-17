# ==================================================================
# SCRIPT FOR COSINE-SIMILARITY KNN GRAPH CREATION
# ==================================================================
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from pathlib import Path
import time


def cosine_knn_graph_creation(patient_dict, k=8):
    """
    Creates a graph for each patient using KNN based on Cosine Similarity
    of the CLS features, and saves each graph to disk.

    Args:
        patient_dict (dict): Dictionary organized by patient ID containing 
                             labels, megapatches, and histodata.
        k (int): Number of neighbors for the KNN graph.
    """

    # --- 1. Setup the Output Directory ---
    graph_dir = Path(__file__).parent / "graphs" / "cosine" / f"k_{k}"
    graph_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    print(f"Starting cosine KNN graph creation for {len(patient_dict)} patients...")

    # --- 2. Loop and Create a Graph for Each Patient ---
    total_patients = len(patient_dict)
    for i, (patient_id, patient_data) in enumerate(patient_dict.items()):
        
        # Print progress on the same line
        print(f"\rProcessing patient {i+1}/{total_patients} ({patient_id})...", end="", flush=True)

        # Extract patch-level features
        node_features = torch.stack([m['features'] for m in patient_data['megapatches']])
        
        # L2-Normalize the features. Euclidean distance on L2-normalized vectors 
        # is mathematically equivalent to ranking by Cosine Similarity.
        normalized_features = F.normalize(node_features, p=2, dim=1)

        graph_label = patient_data['label']
        histodata = patient_data['histodata']

        # Create graph edges based on Cosine Similarity (via normalized Euclidean)
        edge_index = knn_graph(normalized_features, k=k, loop=True)

        # Calculate edge attributes (cosine similarity)
        # Since features are normalized, cosine similarity is the dot product
        src, dst = edge_index
        edge_attr = (normalized_features[src] * normalized_features[dst]).sum(dim=-1)

        # Assemble the graph data object (store the original unnormalized features for training)
        graph = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([graph_label]),
        )
        graph.histodata = histodata
        graph.patient_id = patient_id

        # --- 3. Save the Graph to Disk ---
        torch.save(graph, graph_dir / f"{patient_id}.pt")

    end_time = time.time()
    duration = end_time - start_time
    print("\n" + "-" * 50)
    print(f"Cosine KNN graph creation complete.")
    print(f"Saved {len(list(graph_dir.glob('*.pt')))} graphs to '{graph_dir}' directory.")
    print(f"Total time: {duration // 60:.0f}m {duration % 60:.0f}s")
    print("-" * 50)
