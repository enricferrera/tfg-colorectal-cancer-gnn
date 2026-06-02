# ==================================================================
# BENCHMARK SCRIPT: SEQUENTIAL CPU VS PARALLEL/GPU
# ==================================================================
import time
import torch
from pathlib import Path
import os
import sys

# Get the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Import data loaders
from enric.load_cls import load_cls_metadata, patient_dict_builder

# Import the two versions to compare
from enric.graph_creation.knn.euclidean_knn_graph_creation import euclidean_knn_graph_creation as parallel_fn
from enric.graph_creation.knn.euclidean_knn_graph_creation_cpu import euclidean_knn_graph_creation_cpu as sequential_fn

def run_benchmark():
    # --- 1. Load Data ---
    data_path = PROJECT_ROOT / "Data" / "NEW_DATASET_cls_2048"
    if not data_path.exists():
        print(f"Error: Data path {data_path} not found.")
        return
        
    print(f"Loading metadata from {data_path}...")
    
    # Correctly unpack metadata based on load_cls_metadata return order:
    # return pat_nx_dict, pat_histodata_dict, features, affectation, hospitals, patients, slides, coords
    pat_nx_dict, pat_histodata_dict, features, affectation, hospitals, patients, slides, coords = load_cls_metadata(data_path)
    
    print('------ filtering out patients without diagnosis...-------')
    # Correctly pass arguments to patient_dict_builder:
    # def patient_dict_builder(features, affectation, hospitals, patients, slides, coords, pat_nx_dict, pat_histodata_dict):
    full_patient_dict = patient_dict_builder(features, affectation, hospitals, patients, slides, coords, pat_nx_dict, pat_histodata_dict)
    
    # --- 2. Take first 40 patients ---
    benchmark_patients = {k: full_patient_dict[k] for k in list(full_patient_dict.keys())[:40]}
    print(f"\nBenchmark set to {len(benchmark_patients)} patients.")
    
    k_value = 8
    
    # --- 3. Test Sequential (Classic) Version ---
    print("\n" + "="*50)
    print("RUNNING SEQUENTIAL (CLASSIC CPU) VERSION...")
    print("="*50)
    
    start_seq = time.time()
    sequential_fn(benchmark_patients, k=k_value)
    end_seq = time.time()
    
    seq_duration = end_seq - start_seq
    
    # --- 4. Test Parallel/GPU Version ---
    print("\n" + "="*50)
    print("RUNNING PARALLEL (GPU/MULTI) VERSION...")
    print("="*50)
    
    # We'll use 4 workers as defined in the parallel version
    start_par = time.time()
    parallel_fn(benchmark_patients, k=k_value, num_workers=3)
    end_par = time.time()
    
    par_duration = end_par - start_par
    
    # --- 5. Results ---
    print("\n" + "#" * 50)
    print("BENCHMARK RESULTS (40 Patients)")
    print("#" * 50)
    print(f"Sequential (Classic CPU): {seq_duration:.2f} seconds")
    print(f"Parallel (GPU/Multi):     {par_duration:.2f} seconds")
    
    if seq_duration < par_duration:
        diff = par_duration - seq_duration
        print(f"\n>>> Sequential is FASTER by {diff:.2f} seconds ({par_duration/seq_duration:.1f}x speedup)")
    else:
        diff = seq_duration - par_duration
        print(f"\n>>> Parallel is FASTER by {diff:.2f} seconds ({seq_duration/par_duration:.1f}x speedup)")
    print("#" * 50)

if __name__ == "__main__":
    run_benchmark()
