# ==================================================================
# SCRIPT TO ACCESS PATIENT DATA FOR OFFLINE GRAPH CREATION
# ==================================================================
import torch
from torch_geometric.data import Data

from torch_geometric.nn import radius_graph
from pathlib import Path
import time


def radius_graph_creation_cpu(patient_dict, r=5):
    """
    Iterates through all the patients, creates a connected graph based on a set radius for each,
    and saves each graph to disk.

    Args:
        patient_dict (dict): Dictionary organized by patient ID containing
                             labels, megapatches, and histodata.
        r (int): Radius for the radius graph.
    """

    # --- 1. Setup the Output Directory ---
    graph_dir = Path(__file__).parent / "graphs" / f"r_{r}"
    graph_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    print(f"Starting radius graph creation for {len(patient_dict)} patients...")

    # --- 2. Loop and Create a Graph for Each Patient ---
    total_patients = len(patient_dict)
    for i, (patient_id, patient_data) in enumerate(patient_dict.items()):
        # Print progress on the same line
        print(f"\rProcessing patient {i + 1}/{total_patients} ({patient_id})...", end="", flush=True)

        # Extract patch-level features
        node_features = torch.stack([m['features'] for m in patient_data['megapatches']])

        # Create graph edges based on a set radius from a node in the FEATURE space
        edge_index = radius_graph(node_features, r=r, loop=True)

        # --- MEMORY EFFICIENT DISTANCE CALCULATION ---
        # Instead of full expansion, we use pair-wise vectorized subtraction
        # only on the existing edges. If E is huge, we process in chunks.
        src, dst = edge_index
        
        num_edges = edge_index.size(1)
        chunk_size = 1000000 # Process 1M edges at a time to save VRAM/RAM
        edge_attr_list = []
        
        for start in range(0, num_edges, chunk_size):
            end = min(start + chunk_size, num_edges)
            s_idx = src[start:end]
            d_idx = dst[start:end]
            
            # Compute distance for this chunk
            dist_chunk = torch.norm(node_features[s_idx] - node_features[d_idx], p=2, dim=-1)
            # Convert to weight
            weight_chunk = torch.exp(-dist_chunk / (node_features.size(1) ** 0.5))
            edge_attr_list.append(weight_chunk)
            
        edge_attr = torch.cat(edge_attr_list)

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
    print(f"Radiues graph creation complete.")
    print(f"Saved {len(list(graph_dir.glob('*.pt')))} graphs to '{graph_dir}' directory.")
    print(f"Total time: {duration // 60:.0f}m {duration % 60:.0f}s")
    print("-" * 50)
