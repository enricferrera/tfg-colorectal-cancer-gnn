# ==================================================================
# HIGH-PERFORMANCE GPU COSINE KNN GRAPH CREATION (FILE-BASED)
# ==================================================================
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from pathlib import Path
import time
import torch.multiprocessing as mp
from functools import partial

from dataset.load_cls import load_cls_metadata

def _gpu_cosine_worker(npz_file, pat_nx_dict, pat_histodata_dict, k, graph_dir, device):
    """
    Worker function that processes an entire .npz file (one hospital).
    Uses Cosine Similarity on GPU.
    """
    try:
        data = np.load(npz_file, allow_pickle=True)
    except Exception as e:
        print(f"\nError loading {npz_file.name}: {e}")
        return 0

    features_all = data['embeddingCLS']
    patients_all = data['patient_list']
    
    unique_patients = np.unique(patients_all)
    count = 0

    for pat_id in unique_patients:
        pat_id_str = str(pat_id)
        if pat_id_str not in pat_nx_dict:
            continue

        indices = np.where(patients_all == pat_id)[0]
        pat_features = torch.from_numpy(features_all[indices]).float()
        
        label = int(pat_nx_dict[pat_id_str])
        histodata = torch.from_numpy(np.array(pat_histodata_dict[pat_id_str]).astype(float))

        # --- GPU CALCULATION ---
        pat_features_gpu = pat_features.to(device)
        
        # Normalize features for Cosine Similarity
        norm_features_gpu = F.normalize(pat_features_gpu, p=2, dim=1)
        
        edge_index_gpu = knn_graph(norm_features_gpu, k=k, loop=True)
        
        # Calculate edge attributes (cosine similarity)
        src, dst = edge_index_gpu
        edge_attr_gpu = (norm_features_gpu[src] * norm_features_gpu[dst]).sum(dim=-1)

        graph = Data(
            x=pat_features, 
            edge_index=edge_index_gpu.cpu(),
            edge_attr=edge_attr_gpu.cpu(),
            y=torch.tensor([label]),
        )
        graph.histodata = histodata
        graph.patient_id = pat_id_str
        
        torch.save(graph, graph_dir / f"{pat_id_str}.pt")
        count += 1
        
    return count

def cosine_knn_graph_creation_gpu_fast(npz_path, k=8, num_workers=2, max_files=None):
    """
    High-speed GPU graph creation using Cosine Similarity.
    
    Args:
        npz_path (Path): Path to directory.
        k (int): Neighbors.
        num_workers (int): Parallel processes.
        max_files (int, optional): Max files to process.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    graph_dir = Path(__file__).resolve().parents[4] / "data" / "graphs" / "knn" / "cosine_gpu_fast" / f"k_{k}"
    graph_dir.mkdir(parents=True, exist_ok=True)

    print("Loading clinical metadata mapping...")
    pat_nx_dict, pat_histodata_dict, _, _, _, _, _, _ = load_cls_metadata(npz_path)

    npz_files = sorted(list(npz_path.glob("*.npz")))
    if max_files:
        npz_files = npz_files[:max_files]

    print(f"Processing {len(npz_files)} hospital files.")

    start_time = time.time()
    ctx = mp.get_context('spawn')
    
    worker_fn = partial(
        _gpu_cosine_worker, 
        pat_nx_dict=pat_nx_dict, 
        pat_histodata_dict=pat_histodata_dict, 
        k=k, 
        graph_dir=graph_dir,
        device=device
    )

    total_patients_created = 0
    with ctx.Pool(processes=num_workers) as pool:
        for i, count in enumerate(pool.imap_unordered(worker_fn, npz_files)):
            total_patients_created += count
            print(f"\rProgress: {i+1}/{len(npz_files)} files processed. Total patients: {total_patients_created}", end="", flush=True)

    end_time = time.time()
    duration = end_time - start_time
    print("\n" + "-" * 50)
    print(f"Fast GPU Cosine KNN graph creation complete.")
    print(f"Total time: {duration // 60:.0f}m {duration % 60:.0f}s")
    print("-" * 50)
