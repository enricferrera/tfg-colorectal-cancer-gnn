# ==================================================================
# BENCHMARK SCRIPT: SEQUENTIAL vs SLOW PARALLEL vs GPU FAST
# ==================================================================
import time
from pathlib import Path
import sys

# Get the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Import data loaders
from dataset.load_cls import load_cls_metadata, patient_dict_builder

# Import the three versions to compare
from graphs.knn.euclidean_knn_graph_creation import euclidean_knn_graph_creation as slow_parallel_fn
from graphs.knn.euclidean_knn_graph_creation_gpu_fast import euclidean_knn_graph_creation_gpu_fast as gpu_fast_fn

def run_benchmark():
    # --- 1. SETUP ---
    data_path = PROJECT_ROOT / "data" / "NEW_DATASET_cls_2048"
    if not data_path.exists():
        print(f"Error: Data path {data_path} not found.")
        return
        
    k_value = 8
    num_workers = 2 # Using a consistent low number for fair comparison
    
    # --- 2. PRE-LOAD DATA for Seq and Slow-Parallel ---
    print(f"Loading metadata for Sequential and Slow-Parallel tests...")
    pat_nx_dict, pat_histodata_dict, features, affectation, hospitals, patients, slides, coords = load_cls_metadata(data_path)
    full_patient_dict = patient_dict_builder(features, affectation, hospitals, patients, slides, coords, pat_nx_dict, pat_histodata_dict)
    
    # Take exactly 40 patients
    benchmark_patients = {k: full_patient_dict[k] for k in list(full_patient_dict.keys())[:40]}
    print(f"Benchmark set to {len(benchmark_patients)} patients.")

    '''
    # --- 3. TEST 1: SEQUENTIAL (CLASSIC CPU) ---
    print("\n" + "="*50)
    print("TEST 1: SEQUENTIAL (CLASSIC CPU)")
    print("="*50)
    start_seq = time.time()
    sequential_fn(benchmark_patients, k=k_value)
    duration_seq = time.time() - start_seq
    '''

    # --- 5. TEST 3: GPU FAST (FILE-BASED PARALLELISM) ---
    print("\n" + "="*50)
    print("TEST 3: GPU FAST (OPTIMIZED LOAD + GPU)")
    print("="*50)
    # Limiting to 4 files which is roughly 40-50 patients
    start_fast = time.time()
    gpu_fast_fn(data_path, k=k_value, num_workers=num_workers, max_files=4)
    duration_fast = time.time() - start_fast

    # --- 4. TEST 2: SLOW PARALLEL (GPU + MULTIPROCESSING DICT) ---
    print("\n" + "="*50)
    print("TEST 2: SLOW PARALLEL (GPU + IPC OVERHEAD)")
    print("="*50)
    start_slow = time.time()
    slow_parallel_fn(benchmark_patients, k=k_value, num_workers=num_workers)
    duration_slow = time.time() - start_slow

    # --- 6. FINAL RESULTS ---
    print("\n" + "#" * 60)
    print("FINAL BENCHMARK SUMMARY")
    print("#" * 60)
    #print(f"1. Sequential CPU (40 pat):      {duration_seq:>10.2f}s")
    print(f"2. Slow Parallel GPU (40 pat):   {duration_slow:>10.2f}s")
    print(f"3. GPU Fast (~40-50 pat):        {duration_fast:>10.2f}s")
    
    print("\nNOTE: GPU Fast also includes the time it takes to load metadata from Excel.")
    
    # Calculate per-patient speed
    # We estimate GPU Fast patients as ~44 (4 files * avg patients per file)
    # The Fast version prints the exact count, but for the summary we'll use a fair estimate
    #speed_seq = duration_seq / 40
    speed_slow = duration_slow / 40
    speed_fast = duration_fast / 44 # estimate
    
    print("\nEstimated time PER PATIENT:")
    #print(f" - Sequential CPU: {speed_seq:.4f}s")
    print(f" - Slow Parallel:  {speed_slow:.4f}s")
    print(f" - GPU Fast:       {speed_fast:.4f}s")
    
    speedup = speed_slow / speed_fast
    print(f"\n>>> THE NEW GPU FAST VERSION IS {speedup:.1f}x FASTER THAN THE BASIC GPU!")
    print("#" * 60)

if __name__ == "__main__":
    run_benchmark()
