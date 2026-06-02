# ==================================================================
# SCRIPT TO ACCESS PATIENT DATA FOR OFFLINE GRAPH CREATION
# ==================================================================
import torch
from torch_geometric.data import Data
from pathlib import Path
import time
import torch.multiprocessing as mp
from functools import partial


def _fully_connected_worker(patient_item, graph_dir, device):
    """
    Worker function to process a single patient.
    """
    patient_id, patient_data = patient_item

    # Extract patch-level features and move to GPU
    node_features = torch.stack([m['features'] for m in patient_data['megapatches']]).to(device)
    num_nodes = node_features.size(0)

    # --- GPU ACCELERATED DISTANCE CALCULATION ---
    dist_matrix = torch.cdist(node_features, node_features, p=2)
    edge_attr = torch.exp(-dist_matrix / (node_features.size(1) ** 0.5)).view(-1)

    # Create fully connected edges (including self-loops)
    idx = torch.arange(num_nodes, device=device)
    grid_x, grid_y = torch.meshgrid(idx, idx, indexing='ij')
    edge_index = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=0)

    graph_label = patient_data['label']
    histodata = patient_data['histodata']

    # Assemble the graph data object
    # Move back to CPU for storage
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


def fully_connected_graph_creation(patient_dict, num_workers=2):
    """
    Iterates through all the patients, creates a fully connected graph for each,
    and saves each graph to disk.

    Args:
        patient_dict (dict): Dictionary organized by patient ID containing 
                             labels, megapatches, and histodata.
        num_workers (int): Number of parallel processes to use.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 1. Setup the Output Directory ---
    graph_dir = Path(__file__).parent / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    print(f"Starting parallel fully connected graph creation for {len(patient_dict)} patients on {device}...")

    # --- 2. Process Patients in Parallel ---
    # We use 'spawn' to ensure CUDA contexts are handled correctly in child processes
    ctx = mp.get_context('spawn')
    
    worker_fn = partial(_fully_connected_worker, graph_dir=graph_dir, device=device)
    
    total_patients = len(patient_dict)
    with ctx.Pool(processes=num_workers) as pool:
        for i, patient_id in enumerate(pool.imap_unordered(worker_fn, patient_dict.items())):
            print(f"\rProgress: {i+1}/{total_patients} patients complete ({patient_id})...", end="", flush=True)

    end_time = time.time()
    duration = end_time - start_time
    print("\n" + "-" * 50)
    print(f"Fully connected graph creation complete.")
    print(f"Saved {len(list(graph_dir.glob('*.pt')))} graphs to '{graph_dir}' directory.")
    print(f"Total time: {duration // 60:.0f}m {duration % 60:.0f}s")
    print("-" * 50)

