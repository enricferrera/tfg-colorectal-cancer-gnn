import torch
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import time
import tempfile
import copy
from torchinfo import summary as model_summary

from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score, 
    balanced_accuracy_score, matthews_corrcoef, average_precision_score, 
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns

# Local application imports
from dataset import GraphDataset
from models import GCNWithAgg, GATWeight_batch, GCNWithAggMaxPool, GATWithAggMaxPool
from utils import clean_vram, EarlyStopping
from training import train_loop_graph, val_loop_graph

def run_cross_validation_graph(base_graphs_dir, graph_type, patient_list, label_list, weights, device, 
                               model_type='GCN', n_folds=10, epochs=30, batch_size=1, 
                               minibatch_size=10, hidden_ch=128, use_mixed_precision=True,
                               k=None, r=None, knn_type='euclidean', patience=10,
                               lr=1e-4, weight_decay=1e-5, dropout=0.0, heads=2, pool_ratio=0.5):
    """
    Runs Stratified K-Fold Cross Validation for Graph Neural Networks.
    """
    graphs_dir = base_graphs_dir / graph_type / "graphs"
    if graph_type == 'knn': graphs_dir = graphs_dir / knn_type
    if k is not None: graphs_dir = graphs_dir / f"k_{k}"
    elif r is not None: graphs_dir = graphs_dir / f"r_{r}"

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    
    metric_keys = [
        'recall_0', 'recall_1', 'precision_0', 'precision_1', 
        'f1_0', 'f1_1', 'auc', 'balanced_acc', 'mcc', 'auprc'
    ]
    metrics = {k: [] for k in metric_keys}
    patient_level_results = []

    # Champion Model Tracking
    best_fold_auc = -1.0
    champion_model_state = None

    patient_list = np.array(patient_list)
    label_list = np.array(label_list)

    print(f"\n>>> Running Experiment: {model_type} on {graph_type} graphs")

    # Helper function to create model
    def create_model():
        if model_type.upper() == 'GCN':
            return GCNWithAgg(in_ch=1536, hidden_ch=hidden_ch, out_ch=2, dropout=dropout)
        elif model_type.upper() == 'GAT':
            return GATWeight_batch(in_ch=1536, hidden_ch=hidden_ch, out_ch=2, heads=heads, dropout=dropout)
        elif model_type.upper() == 'GCN_POOL':
            return GCNWithAggMaxPool(in_ch=1536, hidden_ch=hidden_ch, out_ch=2, pool_ratio=pool_ratio, dropout=dropout)
        elif model_type.upper() == 'GAT_POOL':
            return GATWithAggMaxPool(in_ch=1536, hidden_ch=hidden_ch, out_ch=2, heads=heads, pool_ratio=pool_ratio, dropout=dropout)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    # --- 1. Model Complexity & GPU Stats ---
    temp_model = create_model()
    m_info = model_summary(temp_model, verbose=0)
    mlflow.log_params({
        "trainable_params": m_info.trainable_params,
        "model_size_mb": m_info.total_params * 4 / (1024**2),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "lr": lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "heads": heads,
        "pool_ratio": pool_ratio
    })
    del temp_model

    for fold_num, (tr_idx, va_idx) in enumerate(skf.split(patient_list, label_list)):
        fold_start_time = time.time()
        clean_vram()
        print(f"  Fold {fold_num+1}/{n_folds}...", end=" ", flush=True)
        
        with mlflow.start_run(run_name=f"Fold_{fold_num}", nested=True):
            mlflow.log_param("fold_num", fold_num)
            
            train_pats, val_pats = patient_list[tr_idx], patient_list[va_idx]
            train_loader = DataLoader(GraphDataset(graphs_dir, train_pats, features_dir=features_dir), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(GraphDataset(graphs_dir, val_pats, features_dir=features_dir), batch_size=batch_size, shuffle=False)

            model = create_model().to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device))
            scaler = torch.amp.GradScaler('cuda', enabled=use_mixed_precision) if 'cuda' in str(device) else None
            early_stopping = EarlyStopping(patience=patience)

            for epoch in range(epochs):
                train_loss = train_loop_graph(model, train_loader, optimizer, loss_fn, device, minibatch_size, scaler)
                _, _, _, val_loss = val_loop_graph(model, val_loader, device, loss_fn)
                mlflow.log_metric("train_loss", float(train_loss), step=epoch)
                mlflow.log_metric("val_loss", float(val_loss), step=epoch)
                if early_stopping(val_loss, model): break

            if early_stopping.best_model_state is not None:
                model.load_state_dict(early_stopping.best_model_state)

            y_true, y_pred, y_scores, _ = val_loop_graph(model, val_loader, device)
            fold_auc = roc_auc_score(y_true, y_scores)
            
            # Champion Check
            if fold_auc > best_fold_auc:
                best_fold_auc = fold_auc
                champion_model_state = copy.deepcopy(model.state_dict())

            # Error Analysis data
            for idx, p_id in enumerate(val_pats):
                patient_level_results.append({
                    "Patient_ID": p_id, "Fold": fold_num, "True_Label": y_true[idx],
                    "Predicted_Label": y_pred[idx], "Prob": round(y_scores[idx], 4),
                    "Is_Error": y_true[idx] != y_pred[idx]
                })

            # Record Fold Metrics
            fold_metrics = {
                'recall_0': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
                'recall_1': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
                'precision_0': precision_score(y_true, y_pred, pos_label=0, zero_division=0),
                'precision_1': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
                'f1_0': f1_score(y_true, y_pred, pos_label=0, zero_division=0),
                'f1_1': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
                'auc': fold_auc,
                'balanced_acc': balanced_accuracy_score(y_true, y_pred),
                'mcc': matthews_corrcoef(y_true, y_pred),
                'auprc': average_precision_score(y_true, y_scores)
            }
            for k, v in fold_metrics.items():
                metrics[k].append(v)
                mlflow.log_metric(k, float(v))

            mlflow.log_metric("fold_duration_sec", time.time() - fold_start_time)
            mlflow.log_metric("peak_vram_gb", torch.cuda.max_memory_reserved() / (1024**3))
            mlflow.pytorch.log_model(model, artifact_path="model")

            # Plots
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            mlflow.log_figure(fig, f"plots/cm_fold_{fold_num}.png"); plt.close(fig)

    # --- Final Statistical Summary ---
    results_summary = {}
    for k, vals in metrics.items():
        mean, std = np.mean(vals), np.std(vals)
        ci_bound = 1.96 * (std / np.sqrt(n_folds))
        results_summary[k] = (mean, std)
        mlflow.log_metric(f"{k}_mean", float(mean))
        mlflow.log_metric(f"{k}_std", float(std))
        mlflow.log_metric(f"{k}_95_ci", float(ci_bound))

    # Boxplots
    plt.figure(figsize=(10, 6)); sns.boxplot(data=pd.DataFrame(metrics)); plt.xticks(rotation=45)
    mlflow.log_figure(plt.gcf(), "plots/metric_boxplots.png"); plt.close()

    # Error CSV
    df_errors = pd.DataFrame(patient_level_results)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        df_errors.to_csv(tmp.name, index=False)
        mlflow.log_artifact(tmp.name, "patient_level_analysis.csv")

    # Reconstruct Champion model
    champ_model = create_model().to(device)
    champ_model.load_state_dict(champion_model_state)

    return results_summary, champ_model
