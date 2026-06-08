"""
@author: Enric Ferrera González
"""
# Standard library imports
import sys
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

import torch
import pandas as pd
import mlflow
import time
import subprocess

# Local application imports
from utils.seed import fix_seeds
from dataset.load_cls import load_cls_metadata, patient_dict_builder
from training.cross_validation import run_cross_validation_graph
from utils.class_weights import calculate_class_weights

# ------ GLOBAL CONFIGURATION -------
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
mlflow.enable_system_metrics_logging() # <-- Added to track CPU/GPU/RAM automatically
r_seed = 123
fix_seeds(r_seed=r_seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Base paths
BASE_GRAPH_DIR = Path(__file__).parent / "graphs"
npz_path = Path(__file__).resolve().parent.parent / "data" / "NEW_DATASET_cls_2048"

# Load metadata once for all experiments
pat_nx_dict, pat_histodata_dict, features, affectation, hospitals, patients, slides, coords = load_cls_metadata(npz_path)

print('------ filtering out patients without diagnosis...-------')
patient_dict = patient_dict_builder(features, affectation, hospitals, patients, slides, coords, pat_nx_dict, pat_histodata_dict)

# Extraction of patient IDs and Labels for Stratification
patient_list = list(patient_dict.keys())
label_list = [patient_dict[pat]['label'] for pat in patient_list]

# Calculate class weights for loss function
weights = calculate_class_weights(patient_dict)

# ==============================================================================
# EXPERIMENT LIST
# Define the combinations of graphs and models you want to test.
# ==============================================================================
# Use an absolute path for the database to keep it inside the results/ folder
db_path = Path(__file__).resolve().parent.parent / "results" / "mlruns.db"
mlflow.set_tracking_uri(f"sqlite:///{db_path}")
mlflow.set_experiment("PT1Diagnosis_GNN")

experimentos = [
    {
        "name": "GAT_KNN_k5_Cosine",
        "model": "GAT",
        "graph_type": "knn",
        "knn_type": "cosine",
        "k": 10,
        "hidden_ch": 512,
        "epochs": 2,
        "use_amp": True,
        "batch_size": 1,
        "minibatch_size": 15
    },
]

# ==============================================================================
# BENCHMARK EXECUTION
# ==============================================================================
all_results = []

# Generate environment file once per benchmark execution
req_path = Path(__file__).parent.parent / "requirements.txt"
try:
    # Try using uv to export requirements
    subprocess.run(["uv", "pip", "freeze"], stdout=open(req_path, "w"), check=True)
except Exception:
    # Fallback to standard pip if uv is not in path or fails
    subprocess.run(["pip", "freeze"], stdout=open(req_path, "w"), check=True)

for exp in experimentos:
    start_time_exp = time.time()
    print(f"\n{'=' * 50}")
    print(f"STARTING EXPERIMENT: {exp['name']}")
    print(f"{'=' * 50}")
    
    with mlflow.start_run(run_name=exp['name']):
        # Log environment artifact
        mlflow.log_artifact(str(req_path))
        
        # Log all configuration parameters for this experiment
        mlflow.log_params(exp)
        
        # Log global environment and data parameters
        mlflow.log_params({
            "r_seed": r_seed,
            "device": str(device),
            "npz_path": str(npz_path),
            "class_weights": weights.tolist()
        })
        
        try:
            res, champion_model = run_cross_validation_graph(
                base_graphs_dir=BASE_GRAPH_DIR,
                graph_type=exp['graph_type'],
                patient_list=patient_list,
                label_list=label_list,
                weights=weights,
                device=device,
                model_type=exp['model'],
                epochs=exp['epochs'],
                hidden_ch=exp['hidden_ch'],
                use_mixed_precision=exp['use_amp'],
                batch_size=exp["batch_size"],
                minibatch_size=exp["minibatch_size"],
                k=exp.get('k'),
                r=exp.get('r'),
                knn_type=exp.get('knn_type', 'euclidean'),
                patience=exp.get('patience', 10)
            )
            
            # --- NEW: Save the Champion Model to the Parent Run ---
            mlflow.pytorch.log_model(champion_model, artifact_path="champion_model")
            
            # Save summary result for the final printed table
            graph_info = str(exp['graph_type'])
            if exp.get('knn_type'): graph_info += f"_{exp['knn_type']}"
            if exp.get('k'): graph_info += f"_k{exp['k']}"
            if exp.get('r'): graph_info += f"_r{exp['r']}"

            summary = {
                "Experiment": exp['name'], 
                "Model": exp['model'], 
                "Graphs": graph_info
            }
            
            # Use the mean values for the final summary printout
            for metric, (mean, std) in res.items():
                summary[f"{metric}_mean"] = round(mean, 4)
                summary[f"{metric}_std"] = round(std, 4)
            
            # Log total duration
            mlflow.log_metric("total_duration_sec", time.time() - start_time_exp)
            
            all_results.append(summary)
            
        except Exception as e:
            print(f"\n!!! ERROR in experiment {exp['name']}: {e}")
            mlflow.set_tag("status", "failed")
            import traceback
            traceback.print_exc()
            continue

# Cleanup temp requirements file
if req_path.exists():
    req_path.unlink()

# ==============================================================================
# FINAL RESULTS TABLE
# ==============================================================================
if all_results:
    df_results = pd.DataFrame(all_results)
    print("\n\n" + "#" * 60)
    print("FINAL BENCHMARK SUMMARY")
    print("#" * 60)
    # Display key metrics
    cols_to_show = ["Experiment", "auc_mean", "f1_1_mean", "recall_1_mean"]
    existing_cols = [c for c in cols_to_show if c in df_results.columns]
    print(df_results[existing_cols].to_string(index=False))
    
    print(f"\nTo view your MLflow dashboard, run:")
    print(f"mlflow ui --backend-store-uri sqlite:///results/mlruns.db")
