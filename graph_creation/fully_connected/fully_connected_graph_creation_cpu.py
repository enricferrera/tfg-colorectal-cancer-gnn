# ==================================================================
# SCRIPT TO ACCESS PATIENT DATA FOR OFFLINE GRAPH CREATION (CPU PARALLEL)
# ==================================================================
import torch
from torch_geometric.data import Data
from pathlib import Path
import time
import torch.multiprocessing as mp
from functools import partial

def _fully_connected_worker(patient_item, graph_dir):
    """
    Worker function to process a single patient on CPU.
    """
    patient_id, patient_data = patient_item

    # Extract patch-level features (expected to be already stacked in main)
    if 'features_stacked' in patient_data:
        node_features = patient_data['features_stacked']
    else:
        node_features = torch.stack([m['features'] for m in patient_data['megapatches']])
    
    num_nodes = node_features.size(0)

    # --- MEMORY EFFICIENT DISTANCE CALCULATION ---
    dist_matrix = torch.cdist(node_features, node_features, p=2)
    edge_attr = torch.exp(-dist_matrix / (node_features.size(1) ** 0.5)).view(-1)

    # Create fully connected edges (including self-loops)
    idx = torch.arange(num_nodes)
    grid_x, grid_y = torch.meshgrid(idx, idx, indexing='ij')
    edge_index = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=0)

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


def fully_connected_graph_creation_cpu(patient_dict, num_workers=8):
    """
    Iterates through all the patients, creates a fully connected graph for each,
    and saves each graph to disk using multiple CPU cores.
    """

    # --- 1. Setup the Output Directory ---
    graph_dir = Path(__file__).parent / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    print(f"Starting parallel fully connected graph creation (CPU) for {len(patient_dict)} patients...")

    # --- 2. Process Patients in Parallel ---
    ctx = mp.get_context('spawn')
    worker_fn = partial(_fully_connected_worker, graph_dir=graph_dir)

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
