import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import gc

# Local application imports
from enric.dataset.graph_loaders import GraphDataset
from enric.models.models_Graph import GCNWithAgg, GATWeight_batch

def clean_vram():
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def train_loop_graph(model, loader, optimizer, loss_fn, device, minibatch_size=1, scaler=None):
    """
    Trains the GNN model for one epoch using gradient accumulation and optional Mixed Precision.
    """
    model.train()
    total_loss = 0
    batch_counter = 0
    
    optimizer.zero_grad()
    
    for batch in loader:
        batch = batch.to(device)
        
        # Mixed Precision context
        # Autocast automatically chooses between FP16 and FP32 for operations
        with torch.amp.autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu', enabled=(scaler is not None)):
            edge_weight = batch.edge_attr if hasattr(batch, 'edge_attr') else None
            output = model(batch.x, batch.edge_index, edge_weight, batch.batch)
            loss = loss_fn(output, batch.y)
            loss = loss / minibatch_size
        
        if scaler is not None:
            # Scale loss for FP16
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        total_loss += loss.item() * minibatch_size
        batch_counter += 1
        
        if batch_counter >= minibatch_size:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            batch_counter = 0
            
    if batch_counter > 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
            
    return total_loss / len(loader)

def val_loop_graph(model, loader, device):
    """
    Evaluates the GNN model on the validation set.
    """
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            edge_weight = batch.edge_attr if hasattr(batch, 'edge_attr') else None
            output = model(batch.x, batch.edge_index, edge_weight, batch.batch)
            
            probs = F.softmax(output, dim=1).cpu()
            y_true.extend(batch.y.cpu().tolist())
            y_pred.extend(probs.argmax(dim=1).tolist())
            y_scores.extend(probs[:, 1].tolist())
            
    return y_true, y_pred, y_scores

def run_cross_validation_graph(graphs_dir, patient_list, label_list, weights, device, 
                               model_type='GCN', n_folds=10, epochs=30, batch_size=1, 
                               minibatch_size=10, hidden_ch=128, use_mixed_precision=True):
    """
    Orchestrator for GNN Cross-Validation.
    Supports GCN and GAT models, and Mixed Precision to save VRAM.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    metrics = {k: [] for k in ['recall_0', 'recall_1', 'precision_0', 'precision_1', 'f1_0', 'f1_1', 'auc']}

    patient_list = np.array(patient_list)
    label_list = np.array(label_list)

    print(f"\n>>> Running Experiment: {model_type} on {graphs_dir.parent.name} graphs")
    print(f">>> Config: hidden_ch={hidden_ch}, effective_batch={batch_size*minibatch_size}, mixed_precision={use_mixed_precision}")

    for fold_num, (tr_idx, va_idx) in enumerate(skf.split(patient_list, label_list)):
        clean_vram()
        print(f"  Fold {fold_num+1}/{n_folds}...", end=" ", flush=True)
        
        train_pats, val_pats = patient_list[tr_idx], patient_list[va_idx]
        train_loader = DataLoader(GraphDataset(graphs_dir, train_pats), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(GraphDataset(graphs_dir, val_pats), batch_size=batch_size, shuffle=False)

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
        scaler = torch.amp.GradScaler(device_type='cuda', enabled=use_mixed_precision) if 'cuda' in str(device) else None

        for epoch in range(epochs):
            train_loop_graph(model, train_loader, optimizer, loss_fn, device, minibatch_size, scaler)

        y_true, y_pred, y_scores = val_loop_graph(model, val_loader, device)
        
        # Record metrics
        fold_auc = roc_auc_score(y_true, y_scores)
        metrics['recall_0'].append(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
        metrics['recall_1'].append(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        metrics['precision_0'].append(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
        metrics['precision_1'].append(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        metrics['f1_0'].append(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
        metrics['f1_1'].append(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        metrics['auc'].append(fold_auc)
        print(f"AUC: {fold_auc:.4f}")

    # Summary
    results_summary = {k: (np.mean(vals), np.std(vals)) for k, vals in metrics.items()}
    return results_summary
