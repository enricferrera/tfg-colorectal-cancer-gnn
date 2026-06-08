# ==================================================================
# SCRIPT TO ACCESS PATIENT DATA FOR OFFLINE GRAPH CREATION
# ==================================================================
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from pathlib import Path
import time
import torch.multiprocessing as mp
from functools import partial


def _euclidean_knn_worker(patient_item, k, graph_dir, device):
    """
    Worker function to process a single patient.
    """
    patient_id, patient_data = patient_item

    # Extract patch-level features and coordinates and move to GPU
    node_features = torch.stack([m['features'] for m in patient_data['megapatches']]).to(device)

    graph_label = patient_data['label']
    histodata = patient_data['histodata']

    # Create graph edges based on the K-Nearest Neighbors in the FEATURE space
    # This runs on GPU automatically if node_features is on CUDA
    edge_index = knn_graph(node_features, k=k, loop=True)

    # Calculate edge weights (inverse distance or Gaussian kernel)
    src, dst = edge_index
    dist = torch.norm(node_features[src] - node_features[dst], p=2, dim=-1)
    # Use a Gaussian kernel for weights, sigma can be tuned. 
    # Here we use the sqrt of feature dimension as a heuristic scale.
    # Leaves the value between 0 and 1
    edge_attr = torch.exp(-dist / (node_features.size(1) ** 0.5))

    # Assemble the graph data object (Omit 'x' to save space, will be linked during training)
    graph = Data(
        edge_index=edge_index.cpu(),
        edge_attr=edge_attr.cpu(),
        y=torch.tensor([graph_label]),
    )
    graph.histodata = histodata
    graph.patient_id = patient_id

    # --- 3. Save the Graph to Disk ---
    torch.save(graph, graph_dir / f"{patient_id}.pt")
    return patient_id


def euclidean_knn_graph_creation(patient_dict, k=8, num_workers=2):
    """
    Iterates through all the patients, creates a graph for each patient using
    KNN on CLS features, and saves each graph to disk.

    Args:
        patient_dict (dict): Dictionary organized by patient ID containing 
                             labels, megapatches, and histodata.
        k (int): Number of neighbors for the KNN graph.
        num_workers (int): Number of parallel processes to use.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 1. Setup the Output Directory ---
    graph_dir = Path(__file__).resolve().parents[3] / "data" / "graphs" / "knn" / "euclidean" / f"k_{k}"
    graph_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    print(f"Starting parallel knn graph creation for {len(patient_dict)} patients on {device}...")

    # --- 2. Process Patients in Parallel ---
    # We use 'spawn' to ensure CUDA contexts are handled correctly in child processes
    ctx = mp.get_context('spawn')
    
    worker_fn = partial(_euclidean_knn_worker, k=k, graph_dir=graph_dir, device=device)
    
    total_patients = len(patient_dict)
    with ctx.Pool(processes=num_workers) as pool:
        for i, patient_id in enumerate(pool.imap_unordered(worker_fn, patient_dict.items())):
            print(f"\rProgress: {i+1}/{total_patients} patients complete ({patient_id})...", end="", flush=True)

    end_time = time.time()
    duration = end_time - start_time
    print("\n" + "-" * 50)
    print(f"KNN graph creation complete.")
    print(f"Saved {len(list(graph_dir.glob('*.pt')))} graphs to '{graph_dir}' directory.")
    print(f"Total time: {duration // 60:.0f}m {duration % 60:.0f}s")
    print("-" * 50)
