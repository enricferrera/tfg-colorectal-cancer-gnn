# ==================================================================
# SCRIPT FOR COSINE-SIMILARITY KNN GRAPH CREATION
# ==================================================================
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from pathlib import Path
import time
import torch.multiprocessing as mp
from functools import partial


def _cosine_knn_worker(patient_item, k, graph_dir, device):
    """
    Worker function to process a single patient.
    """
    patient_id, patient_data = patient_item

    # Extract patch-level features and move to GPU
    node_features = torch.stack([m['features'] for m in patient_data['megapatches']]).to(device)
    
    # L2-Normalize the features. Euclidean distance on L2-normalized vectors 
    # is mathematically equivalent to ranking by Cosine Similarity.
    normalized_features = F.normalize(node_features, p=2, dim=1)

    graph_label = patient_data['label']
    histodata = patient_data['histodata']

    # Create graph edges based on Cosine Similarity (via normalized Euclidean)
    # This will automatically run on GPU if normalized_features is a CUDA tensor
    edge_index = knn_graph(normalized_features, k=k, loop=True)

    # Calculate edge attributes (cosine similarity)
    # Since features are normalized, cosine similarity is the dot product
    src, dst = edge_index
    edge_attr = (normalized_features[src] * normalized_features[dst]).sum(dim=-1)

    # Assemble the graph data object
    # Move everything back to CPU for storage to free up VRAM
    graph = Data(
        x=node_features.cpu(),
        edge_index=edge_index.cpu(),
        edge_attr=edge_attr.cpu(),
        y=torch.tensor([graph_label]),
    )
    graph.histodata = histodata
    graph.patient_id = patient_id

    # --- 3. Save the Graph to Disk ---
    torch.save(graph, graph_dir / f"{patient_id}.pt")
    return patient_id


def cosine_knn_graph_creation(patient_dict, k=8, num_workers=4):
    """
    Creates a graph for each patient using KNN based on Cosine Similarity
    of the CLS features, and saves each graph to disk.

    Args:
        patient_dict (dict): Dictionary organized by patient ID containing 
                             labels, megapatches, and histodata.
        k (int): Number of neighbors for the KNN graph.
        num_workers (int): Number of parallel processes to use.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 1. Setup the Output Directory ---
    graph_dir = Path(__file__).parent / "graphs" / "cosine" / f"k_{k}"
    graph_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    print(f"Starting parallel cosine KNN graph creation for {len(patient_dict)} patients on {device}...")

    # --- 2. Process Patients in Parallel ---
    # We use 'spawn' to ensure CUDA contexts are handled correctly in child processes
    ctx = mp.get_context('spawn')
    
    worker_fn = partial(_cosine_knn_worker, k=k, graph_dir=graph_dir, device=device)
    
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
