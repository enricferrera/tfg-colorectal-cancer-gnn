# ==================================================================
# SCRIPT TO ACCESS PATIENT DATA FOR OFFLINE GRAPH CREATION (CPU PARALLEL)
# ==================================================================
import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from pathlib import Path
import time
import torch.multiprocessing as mp
from functools import partial

def _radius_worker(patient_item, r, graph_dir):
    """
    Worker function to process a single patient on CPU.
    """
    patient_id, patient_data = patient_item

    # Extract patch-level features (expected to be already stacked in main)
    if 'features_stacked' in patient_data:
        node_features = patient_data['features_stacked']
    else:
        node_features = torch.stack([m['features'] for m in patient_data['megapatches']])

    # Create graph edges based on a set radius from a node in the FEATURE space
    edge_index = radius_graph(node_features, r=r, loop=True)

    # --- MEMORY EFFICIENT DISTANCE CALCULATION ---
    src, dst = edge_index
    num_edges = edge_index.size(1)
    chunk_size = 1000000 
    edge_attr_list = []
    
    for start in range(0, num_edges, chunk_size):
        end = min(start + chunk_size, num_edges)
        s_idx = src[start:end]
        d_idx = dst[start:end]
        
        dist_chunk = torch.norm(node_features[s_idx] - node_features[d_idx], p=2, dim=-1)
        weight_chunk = torch.exp(-dist_chunk / (node_features.size(1) ** 0.5))
        edge_attr_list.append(weight_chunk)
        
    edge_attr = torch.cat(edge_attr_list) if edge_attr_list else torch.tensor([])

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


def radius_graph_creation_cpu(patient_dict, r=5, num_workers=8):
    """
    Iterates through all the patients, creates a connected graph based on a set radius for each,
    and saves each graph to disk using multiple CPU cores.
    """

    # --- 1. Setup the Output Directory ---
    graph_dir = Path(__file__).resolve().parents[3] / "data" / "graphs" / "radius" / f"r_{r}"
    graph_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    print(f"Starting parallel radius graph creation (CPU) for {len(patient_dict)} patients with r={r}...")

    # --- 2. Process Patients in Parallel ---
    ctx = mp.get_context('spawn')
    worker_fn = partial(_radius_worker, r=r, graph_dir=graph_dir)

    total_patients = len(patient_dict)
    with ctx.Pool(processes=num_workers) as pool:
        for i, patient_id in enumerate(pool.imap_unordered(worker_fn, patient_dict.items())):
            print(f"\rProgress: {i+1}/{total_patients} patients complete ({patient_id})...", end="", flush=True)

    end_time = time.time()
    duration = end_time - start_time
    print("\n" + "-" * 50)
    print(f"Radius graph creation complete.")
    print(f"Saved {len(list(graph_dir.glob('*.pt')))} graphs to '{graph_dir}' directory.")
    print(f"Total time: {duration // 60:.0f}m {duration % 60:.0f}s")
    print("-" * 50)
