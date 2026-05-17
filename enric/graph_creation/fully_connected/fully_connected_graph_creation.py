# ==================================================================
# SCRIPT TO ACCESS PATIENT DATA FOR OFFLINE GRAPH CREATION
# ==================================================================
import torch
from torch_geometric.data import Data
from pathlib import Path
import time



def fully_connected_graph_creation(patient_dict):
    """
    Iterates through all the patients, creates a fully connected graph for each,
    and saves each graph to disk.

    Args:
        patient_dict (dict): Dictionary organized by patient ID containing 
                             labels, megapatches, and histodata.
    """

    # --- 1. Setup the Output Directory ---
    graph_dir = Path(__file__).parent / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    print(f"Starting fully connected graph creation for {len(patient_dict)} patients...")

    # --- 2. Loop and Create a Graph for Each Patient ---
    total_patients = len(patient_dict)
    for i, (patient_id, patient_data) in enumerate(patient_dict.items()):
        
        # Print progress on the same line
        print(f"\rProcessing patient {i+1}/{total_patients} ({patient_id})...", end="", flush=True)

        # Extract patch-level features
        node_features = torch.stack([m['features'] for m in patient_data['megapatches']])
        num_nodes = node_features.size(0)

        # --- MEMORY EFFICIENT DISTANCE CALCULATION ---
        # Instead of expanding to (N^2, Dim), we use cdist which returns (N, N)
        # For N=3000, (N, N) is ~36MB, while (N^2, Dim) was ~50GB!
        dist_matrix = torch.cdist(node_features, node_features, p=2)
        edge_attr = torch.exp(-dist_matrix / (node_features.size(1) ** 0.5)).view(-1)

        # Create fully connected edges (including self-loops)
        # Matches the flattened dist_matrix view
        idx = torch.arange(num_nodes)
        grid_x, grid_y = torch.meshgrid(idx, idx, indexing='ij')
        edge_index = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=0)

        graph_label = patient_data['label']
        histodata = patient_data['histodata']

        # Assemble the graph data object
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
    print("-" * 50)
    print(f"Fully connected graph creation complete.")
    print(f"Saved {len(list(graph_dir.glob('*.pt')))} graphs to '{graph_dir}' directory.")
    print(f"Total time: {duration // 60:.0f}m {duration % 60:.0f}s")
    print("-" * 50)
