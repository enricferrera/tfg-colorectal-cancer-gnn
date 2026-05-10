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

        # Create fully connected edges (including self-loops)
        # We create a 2x(N^2) tensor where every node is connected to every other node
        # This is done by getting all indices from an N x N matrix of ones
        adj = torch.ones((num_nodes, num_nodes))
        edge_index = adj.nonzero().t().contiguous()

        graph_label = patient_data['label']
        histodata = patient_data['histodata']

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
    print(f"Fully connected graph creation complete.")
    print(f"Saved {len(list(graph_dir.glob('*.pt')))} graphs to '{graph_dir}' directory.")
    print(f"Total time: {duration // 60:.0f}m {duration % 60:.0f}s")
    print("-" * 50)
