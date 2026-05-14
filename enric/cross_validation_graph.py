import torch
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Local application imports
from enric.dataset import GraphDataset
from enric.models import GCNWithAgg, GATWeight_batch
from enric.utils import clean_vram, EarlyStopping
from enric.training import train_loop_graph, val_loop_graph

def run_cross_validation_graph(base_graphs_dir, graph_type, patient_list, label_list, weights, device, 
                               model_type='GCN', n_folds=10, epochs=30, batch_size=1, 
                               minibatch_size=10, hidden_ch=128, use_mixed_precision=True,
                               k=None, r=None, knn_type='euclidean', patience=10):
    """
    Orchestrator for GNN Cross-Validation.
    Supports GCN and GAT models, and Mixed Precision to save VRAM.
    """
    # Construct the base path for the specific graph type
    graphs_dir = base_graphs_dir / graph_type / "graphs"
    
    if graph_type == 'knn':
        graphs_dir = graphs_dir / knn_type
    
    if k is not None:
        graphs_dir = graphs_dir / f"k_{k}"
    elif r is not None:
        graphs_dir = graphs_dir / f"r_{r}"

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    
    metrics = {k: [] for k in ['recall_0', 'recall_1', 'precision_0', 'precision_1', 'f1_0', 'f1_1', 'auc']}

    patient_list = np.array(patient_list)
    label_list = np.array(label_list)

    print(f"\n>>> Running Experiment: {model_type} on {graph_type} graphs ({graphs_dir.name})")
    print(f">>> Config: hidden_ch={hidden_ch}, effective_batch={batch_size * minibatch_size}, mixed_precision={use_mixed_precision}")

    for fold_num, (tr_idx, va_idx) in enumerate(skf.split(patient_list, label_list)):
        # Ensure VRAM is clean before starting each fold
        clean_vram()
        print(f"  Fold {fold_num+1}/{n_folds}...", end=" ", flush=True)
        
        train_pats = patient_list[tr_idx]
        val_pats = patient_list[va_idx]
        
        # Loaders
        train_ds = GraphDataset(graphs_dir, train_pats)
        val_ds = GraphDataset(graphs_dir, val_pats)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Initialize requested model
        if model_type.upper() == 'GCN':
            model = GCNWithAgg(in_ch=1536, hidden_ch=hidden_ch, out_ch=2).to(device)
        elif model_type.upper() == 'GAT':
            model = GATWeight_batch(in_ch=1536, hidden_ch=hidden_ch, out_ch=2).to(device)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device))
        
        # GradScaler for Mixed Precision (FP16)
        scaler = torch.amp.GradScaler('cuda', enabled=use_mixed_precision) if 'cuda' in str(device) else None

        # Early Stopping initialization
        early_stopping = EarlyStopping(patience=patience)

        # Train loop
        for epoch in range(epochs):
            train_loss = train_loop_graph(model, train_loader, optimizer, loss_fn, device, minibatch_size, scaler)
            
            # Validation at the end of each epoch
            y_true_v, y_pred_v, y_scores_v, val_loss = val_loop_graph(model, val_loader, device, loss_fn)
            
            # Check Early Stopping
            if early_stopping(val_loss, model):
                print(f"      Early stopping at epoch {epoch+1}")
                break

            if (epoch + 1) % 5 == 0:
                print(f"      Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        # Load best model weights found during training
        if early_stopping.best_model_state is not None:
            model.load_state_dict(early_stopping.best_model_state)

        # Final Validation on the best model state
        y_true, y_pred, y_scores, _ = val_loop_graph(model, val_loader, device)
        
        # Record metrics
        fold_auc = roc_auc_score(y_true, y_scores)
        metrics['recall_0'].append(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
        metrics['recall_1'].append(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        metrics['precision_0'].append(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
        metrics['precision_1'].append(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        metrics['f1_0'].append(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
        metrics['f1_1'].append(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        metrics['auc'].append(fold_auc)
        print(f"Best Val AUC: {fold_auc:.4f}")

    # Summary
    results_summary = {k: (np.mean(vals), np.std(vals)) for k, vals in metrics.items()}
    return results_summary
