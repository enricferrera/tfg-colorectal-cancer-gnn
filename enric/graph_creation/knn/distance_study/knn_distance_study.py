# ==================================================================
# SCRIPT FOR CLS NODE DISTANCE ANALYSIS & NEIGHBOR DUAL STUDY
# ==================================================================
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

# Add project root to path
current_file = Path(__file__).resolve()
root_path = current_file.parents[4] 

if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

try:
    from enric import load_cls_metadata, patient_dict_builder
except ImportError:
    print(f"Could not import data loaders. Checked root_path: {root_path}")
    raise

# ==================================================================
# 1. DISTANCE DISTRIBUTION ANALYSIS
# ==================================================================
def analyze_single_patient_distribution(patient_id, patient_data, num_bins=100, range_max=250):
    """
    Analyzes pairwise distances for a SINGLE patient and generates a histogram.
    """
    print(f"\n[Single Patient] Analyzing distance distribution for Patient: {patient_id}")
    
    node_features = torch.stack([m['features'] for m in patient_data['megapatches']])
    pairwise_distances = torch.cdist(node_features, node_features)

    distances_flat = pairwise_distances.flatten()
    distances_flat = distances_flat[distances_flat > 0]

    if len(distances_flat) == 0:
        print(f"No distances to analyze for patient {patient_id} (only 1 node).")
        return

    patient_hist = torch.histc(distances_flat, bins=num_bins, min=0, max=range_max)

    bin_edges = np.linspace(0, range_max, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    print(f"\nDistance Bin Counts for Patient {patient_id}:")
    print(f"{'Bin Center':>12} | {'Frequency':>12}")
    print("-" * 27)
    for center, count in zip(bin_centers, patient_hist):
        if count > 0:
            print(f"{center:12.2f} | {int(count):12d}")
    
    plt.figure(figsize=(12, 7))
    plt.bar(bin_centers, patient_hist.numpy(), width=range_max/num_bins, color='darkgreen', alpha=0.7)
    plt.title(f'Distance Histogram - Patient: {patient_id}')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_dir = Path(__file__).parent / "analysis_plots" / "patients"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"patient_{patient_id}_histogram.png"
    plt.savefig(save_path)
    plt.close()
    print(f"\nPatient Histogram saved to: {save_path}")

def analyze_distance_distribution(patient_dict):
    """
    Analyzes pairwise distances across ALL patients and generates a global histogram.
    """
    print(f"\n[Part 1] Starting Global Distance Distribution Analysis...")
    
    num_bins = 100
    range_max = 250
    global_hist = torch.zeros(num_bins)
    total_patients = len(patient_dict)
    
    for i, (patient_id, patient_data) in enumerate(patient_dict.items()):
        print(f"\rProcessing Distances: {i+1}/{total_patients}...", end="", flush=True)
        
        node_features = torch.stack([m['features'] for m in patient_data['megapatches']])
        # cdist calculates the distances between all the nodes in par1 and par2
        # it takes a third argument p that p=1 manhattan, p=2 euclidean p>2 chebyshev
        pairwise_distances = torch.cdist(node_features, node_features)

        # Collapse 2D matrix into 1D list, easier to perform calculations
        distances_flat = pairwise_distances.flatten()
        # Remove 0 values (diagonal elements of the matrix)
        distances_flat = distances_flat[distances_flat > 0]

        # Check patient has more than 1 node
        if len(distances_flat) > 0:
            # It adds each value to its corresponding bin. h is a vector with the count of distances found in each range
            h = torch.histc(distances_flat, bins=num_bins, min=0, max=range_max)
            global_hist += h

    # Plotting
    # Create num_bins+1 walls to contain each bar in the histo
    bin_edges = np.linspace(0, range_max, num_bins + 1)
    # Calculate the center across the x-axis
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    print("\n\nGlobal Distance Bin Counts:")
    print(f"{'Bin Center':>12} | {'Frequency':>12}")
    print("-" * 27)
    for center, count in zip(bin_centers, global_hist):
        print(f"{center:12.2f} | {int(count):12d}")
    
    plt.figure(figsize=(12, 7))
    plt.bar(bin_centers, global_hist.numpy(), width=range_max/num_bins, color='darkblue', alpha=0.7)
    plt.title(f'Global Histogram of Pairwise Distances ({total_patients} Patients)')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_dir = Path(__file__).parent / "analysis_plots"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "global_distance_histogram.png")
    plt.close()
    print(f"\nGlobal Histogram saved to: {output_dir / 'global_distance_histogram.png'}")

# ==================================================================
# 2. NEIGHBOR DENSITY STUDY (CUMULATIVE & MARGINAL)
# ==================================================================
def analyze_neighbor_density(patient_dict):
    """
    Analyzes node connectivity using both:
    - Cumulative Count: Total neighbors within Radius R (Used for building graphs)
    - Marginal Count: NEW neighbors found in that shell (Matches histogram peaks)
    """
    print(f"\n[Part 2] Starting Dual Neighbor Study...")
    
    thresholds = [0, 5, 60, 80, 100, 120, 140, 160, 200]
    # Create dictionary, key=radius and value=empty list
    cumulative_stats = {t: [] for t in thresholds if t > 0}
    # Create dictionary for representing intervals
    marginal_stats = {f"{thresholds[i]}-{thresholds[i+1]}": [] for i in range(len(thresholds)-1)}
    
    total_patients = len(patient_dict)
    
    for i, (patient_id, patient_data) in enumerate(patient_dict.items()):
        print(f"\rProcessing Neighbors: {i+1}/{total_patients}...", end="", flush=True)
        
        node_features = torch.stack([m['features'] for m in patient_data['megapatches']])
        pairwise_distances = torch.cdist(node_features, node_features)
        
        # 1. Calculate Cumulative Counts (Total within Radius)
        for t in cumulative_stats.keys():
            counts = (pairwise_distances < t).sum(dim=1).float() - 1 # Subtract self
            cumulative_stats[t].extend(counts.tolist())
            
        # 2. Calculate Marginal Counts (New in this shell)
        for j in range(len(thresholds)-1):
            low, high = thresholds[j], thresholds[j+1]
            mask = (pairwise_distances >= low) & (pairwise_distances < high)
            counts = mask.sum(dim=1).float()
            if low == 0: counts -= 1 # Subtract self
            marginal_stats[f"{low}-{high}"].extend(counts.tolist())

    print("\n\nNEIGHBOR STUDY RESULTS:")
    print("-" * 105)
    print(f"{'Radius (R)':<12} | {'Total Neighbors (<R)':<25} | {'Interval':<12} | {'New Neighbors (Shell)':<25}")
    print("-" * 105)
    
    # We'll zip the results to show them side-by-side
    cum_keys = list(cumulative_stats.keys())
    mar_keys = list(marginal_stats.keys())
    
    for i in range(len(cum_keys)):
        c_r = cum_keys[i]
        c_avg = np.mean(cumulative_stats[c_r])
        
        m_int = mar_keys[i+1] if i+1 < len(mar_keys) else "N/A"
        m_avg = np.mean(marginal_stats[m_int]) if i+1 < len(mar_keys) else 0
        
        print(f"{c_r:<12} | {c_avg:<25.2f} | {m_int:<12} | {m_avg:<25.2f}")
    
    print("-" * 105)
    print("Interpretation:")
    print("1. 'Total Neighbors' is what your GNN will actually see for a given radius.")
    print("2. 'New Neighbors' shows the density PEAK. Note how it increases then decreases!")

# ==================================================================
# MAIN EXECUTION
# ==================================================================
if __name__ == '__main__':
    npz_path = root_path / "Data" / "cls_ALL"
    pat_nx_dict, pat_histodata_dict, features, affectation, hospitals, patients, slides, coords = load_cls_metadata(npz_path)
    patient_dict = patient_dict_builder(features, affectation, hospitals, patients, slides, coords, pat_nx_dict, pat_histodata_dict)

    start_total = time.time()
    
    # Example: Analyze first patient specifically
    first_patient_id = list(patient_dict.keys())[0]
    analyze_single_patient_distribution(first_patient_id, patient_dict[first_patient_id])
    
    analyze_distance_distribution(patient_dict)
    analyze_neighbor_density(patient_dict)
    print(f"\nAll studies complete. Total time: {time.time() - start_total:.2f} seconds")
