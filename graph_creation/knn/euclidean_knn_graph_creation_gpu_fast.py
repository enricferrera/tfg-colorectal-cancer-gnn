# ==================================================================
# HIGH-PERFORMANCE GPU KNN GRAPH CREATION (FILE-BASED PARALLELISM)
# ==================================================================
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from pathlib import Path
import time
import torch.multiprocessing as mp
from functools import partial

# We import the metadata loader but not the full data loader to avoid pre-loading everything
from load_cls import load_cls_metadata

def _gpu_fast_worker(npz_file, pat_nx_dict, pat_histodata_dict, k, graph_dir, device):
    """
    Worker function that processes an entire .npz file (one hospital).
    Loads data locally to avoid serialization overhead.
    """
    # 1. Load the NPZ file locally in this process
    try:
        data = np.load(npz_file, allow_pickle=True)
    except Exception as e:
        print(f"\nError loading {npz_file.name}: {e}")
        return 0

    features_all = data['embeddingCLS']
    patients_all = data['patient_list']
    
    # We use 'infiltrations' for affectation based on enric/load_cls.py
    # and other keys if needed, but for KNN features is the main one.

    unique_patients = np.unique(patients_all)
    count = 0

    for pat_id in unique_patients:
        pat_id_str = str(pat_id)
        if pat_id_str not in pat_nx_dict:
            continue

        # Extract data for this specific patient
        indices = np.where(patients_all == pat_id)[0]
        
        # Prepare node features (CLS embeddings)
        pat_features = torch.from_numpy(features_all[indices]).float()
        
        # Metadata
        label = int(pat_nx_dict[pat_id_str])
        histodata = torch.from_numpy(np.array(pat_histodata_dict[pat_id_str]).astype(float))

        # --- GPU CALCULATION ---
        # Move ONLY the features needed for calculation to GPU
        pat_features_gpu = pat_features.to(device)
        
        # Calculate KNN on GPU
        edge_index_gpu = knn_graph(pat_features_gpu, k=k, loop=True)
        
        # Calculate edge weights (Gaussian kernel) on GPU
        src, dst = edge_index_gpu
        dist_gpu = torch.norm(pat_features_gpu[src] - pat_features_gpu[dst], p=2, dim=-1)
        edge_attr_gpu = torch.exp(-dist_gpu / (pat_features.size(1) ** 0.5))

        # Assemble result and move back to CPU
        graph = Data(
            x=pat_features, # keep features on CPU
            edge_index=edge_index_gpu.cpu(),
            edge_attr=edge_attr_gpu.cpu(),
            y=torch.tensor([label]),
        )
        graph.histodata = histodata
        graph.patient_id = pat_id_str
        
        # Save to disk
        torch.save(graph, graph_dir / f"{pat_id_str}.pt")
        count += 1
        
    return count

def euclidean_knn_graph_creation_gpu_fast(npz_path, k=8, num_workers=2, max_files=None):
    """
    High-speed GPU graph creation.
    Parallelizes by hospital file to avoid sending heavy tensors between processes.
    
    Args:
        npz_path (Path): Path to the directory containing .npz files.
        k (int): Number of neighbors for KNN.
        num_workers (int): Number of parallel processes to use.
        max_files (int, optional): Max number of files to process (useful for benchmarking).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("WARNING: CUDA not available. Running 'Fast' logic on CPU cores.")

    # --- 1. Setup Output ---
    graph_dir = Path(__file__).parent / "graphs" / "euclidean_gpu_fast" / f"k_{k}"
    graph_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. Load Metadata (Maps) only ---
    print("Loading clinical metadata mapping...")
    pat_nx_dict, pat_histodata_dict, _, _, _, _, _, _ = load_cls_metadata(npz_path)

    # --- 3. Identify Files ---
    npz_files = sorted(list(npz_path.glob("*.npz")))
    if max_files:
        npz_files = npz_files[:max_files]
    
    print(f"Processing {len(npz_files)} hospital files.")

    start_time = time.time()

    # --- 4. Parallel Process by File ---
    ctx = mp.get_context('spawn')
    
    worker_fn = partial(
        _gpu_fast_worker, 
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
    print(f"Fast GPU KNN graph creation complete.")
    print(f"Processed {len(npz_files)} files, created {total_patients_created} graphs.")
    print(f"Total time: {duration // 60:.0f}m {duration % 60:.0f}s")
    print("-" * 50)
