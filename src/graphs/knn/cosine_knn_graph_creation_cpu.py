# ==================================================================
# SCRIPT FOR COSINE-SIMILARITY KNN GRAPH CREATION (CPU PARALLEL)
# ==================================================================
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from pathlib import Path
import time
import torch.multiprocessing as mp
from functools import partial

def _cosine_knn_worker(patient_item, k, graph_dir):
    """
    Worker function to process a single patient on CPU.
    """
    patient_id, patient_data = patient_item

    # Extract patch-level features (expected to be already stacked in main)
    if 'features_stacked' in patient_data:
        node_features = patient_data['features_stacked']
    else:
        node_features = torch.stack([m['features'] for m in patient_data['megapatches']])
    
    # L2-Normalize the features. 
    normalized_features = F.normalize(node_features, p=2, dim=1)

    # Create graph edges based on Cosine Similarity (via normalized Euclidean)
    edge_index = knn_graph(normalized_features, k=k, loop=True)

    # Calculate edge attributes (cosine similarity)
    src, dst = edge_index
    edge_attr = (normalized_features[src] * normalized_features[dst]).sum(dim=-1)

    graph_label = patient_data['label']
    histodata = patient_data['histodata']

    # Assemble the graph data object (Omit 'x' to save space, will be linked during training)
    graph = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([graph_label]),
    )
    graph.histodata = histodata
    graph.patient_id = patient_id

    # --- 3. Save the Graph to Disk ---
    torch.save(graph, graph_dir / f"{patient_id}.pt")
    return patient_id


def cosine_knn_graph_creation_cpu(patient_dict, k=8, num_workers=8):
    """
    Creates a graph for each patient using KNN based on Cosine Similarity
    using multiple CPU cores.
    """

    # --- 1. Setup the Output Directory ---
    graph_dir = Path(__file__).resolve().parents[3] / "data" / "graphs" / "knn" / "cosine" / f"k_{k}"
    graph_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    print(f"Starting parallel cosine KNN graph creation (CPU) for {len(patient_dict)} patients with k={k}...")

    # --- 2. Process Patients in Parallel ---
    ctx = mp.get_context('spawn')
    worker_fn = partial(_cosine_knn_worker, k=k, graph_dir=graph_dir)

    total_patients = len(patient_dict)
    with ctx.Pool(processes=num_workers) as pool:
        for i, patient_id in enumerate(pool.imap_unordered(worker_fn, patient_dict.items())):
            print(f"\rProgress: {i+1}/{total_patients} patients complete ({patient_id})...", end="", flush=True)

    end_time = time.time()
    duration = end_time - start_time
    print("\n" + "-" * 50)
    print(f"Cosine KNN graph creation complete.")
    print(f"Saved {len(list(graph_dir.glob('*.pt')))} graphs to '{graph_dir}' directory.")
    print(f"Total time: {duration // 60:.0f}m {duration % 60:.0f}s")
    print("-" * 50)
