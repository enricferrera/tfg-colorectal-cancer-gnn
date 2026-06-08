"""
Optuna search script for PT1Diagnosis GNN models.
This script explores the combinatorial space of graph types, model architectures, 
and training hyperparameters.
"""

import optuna
import mlflow
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from utils.seed import fix_seeds
from dataset.load_cls import load_cls_metadata, patient_dict_builder
from training.cross_validation import run_cross_validation_graph
from utils.class_weights import calculate_class_weights

# Configuration
r_seed = 123
fix_seeds(r_seed=r_seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_GRAPH_DIR = current_dir.parent / "data" / "graphs"
npz_path = current_dir.parent / "data" / "NEW_DATASET_cls_2048"

# Load metadata
pat_nx_dict, pat_histodata_dict, features, affectation, hospitals, patients, slides, coords = load_cls_metadata(npz_path)
patient_dict = patient_dict_builder(features, affectation, hospitals, patients, slides, coords, pat_nx_dict, pat_histodata_dict)
patient_list = list(patient_dict.keys())
label_list = [patient_dict[pat]['label'] for pat in patient_list]
weights = calculate_class_weights(patient_dict)

db_path = Path(__file__).resolve().parent.parent / "results" / "mlruns.db"
mlflow.set_tracking_uri(f"sqlite:///{db_path}")
mlflow.set_experiment("Optuna_GNN_Search")

def objective(trial):
    # 1. Define Search Space
    
    # Graph Level
    graph_type = trial.suggest_categorical("graph_type", ["knn", "radius", "fully_connected"])
    
    k = None
    r = None
    knn_type = 'euclidean'
    
    if graph_type == "knn":
        knn_type = trial.suggest_categorical("knn_type", ["euclidean", "cosine"])
        # Available K values in the project
        k_values = [8, 10, 12, 14, 16, 18, 20]
        if knn_type == "euclidean": k_values.append(50)
        k = trial.suggest_categorical("k", k_values)
    elif graph_type == "radius":
        # Available R values in the project
        r = trial.suggest_categorical("r", [60, 64, 68, 72, 76, 80])
    
    # Model Level
    model_type = trial.suggest_categorical("model_type", ["GCN", "GAT", "GCN_POOL", "GAT_POOL"])
    hidden_ch = trial.suggest_categorical("hidden_ch", [128, 256, 512, 768])
    heads = trial.suggest_categorical("heads", [2, 4, 8]) if "GAT" in model_type else 1
    pool_ratio = trial.suggest_float("pool_ratio", 0.2, 0.8) if "POOL" in model_type else 0.5
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    
    # Training Level
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    minibatch_size = trial.suggest_int("minibatch_size", 5, 20)
    
    # 2. Run Trial
    # We use a smaller number of folds for searching to save time, or 10 if resources allow.
    # Here I'll use 5 folds for the search phase.
    n_folds = 5 
    epochs = 20 # Rely on EarlyStopping
    
    run_name = f"Trial_{trial.number}_{model_type}_{graph_type}"
    with mlflow.start_run(run_name=run_name, nested=True):
        # Log trial attributes to MLflow
        mlflow.log_params(trial.params)
        
        try:
            res, _ = run_cross_validation_graph(
                base_graphs_dir=BASE_GRAPH_DIR,
                graph_type=graph_type,
                patient_list=patient_list,
                label_list=label_list,
                weights=weights,
                device=device,
                model_type=model_type,
                n_folds=n_folds,
                epochs=epochs,
                hidden_ch=hidden_ch,
                batch_size=1,
                minibatch_size=minibatch_size,
                k=k,
                r=r,
                knn_type=knn_type,
                lr=lr,
                weight_decay=weight_decay,
                dropout=dropout,
                heads=heads,
                pool_ratio=pool_ratio,
                patience=5 # Faster search
            )
            
            # The primary metric to optimize
            auc_mean = res['auc'][0]
            return auc_mean
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return 0.0 # Return poor score on failure

if __name__ == "__main__":
    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name="GNN_Hyperparameter_Optimization",
        storage=f"sqlite:///{db_path.parent}/optuna.db", # Persistent storage for Optuna
        load_if_exists=True
    )
    
    print(f"Starting optimization...")
    study.optimize(objective, n_trials=30)
    
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    print(f"Best Trial Score: {study.best_value}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*50)
