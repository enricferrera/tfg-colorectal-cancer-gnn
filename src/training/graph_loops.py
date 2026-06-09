import torch
import torch.nn.functional as F

def train_loop_graph(model, loader, optimizer, loss_fn, device, minibatch_multiplier=1, scaler=None):
    """
    Trains the GNN model for one epoch using gradient accumulation and optional Mixed Precision.
    """
    model.train()
    total_loss = 0
    batch_counter = 0

    # Initialize gradients at the start of the epoch
    optimizer.zero_grad()

    for batch in loader:
        batch = batch.to(device)

        # Mixed Precision context
        # Use newer torch.amp API to remove deprecation warnings
        device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=(scaler is not None)):
            edge_weight = batch.edge_attr if hasattr(batch, 'edge_attr') else None
            output = model(batch.x, batch.edge_index, edge_weight, batch.batch)

            # Calculate and scale loss
            loss = loss_fn(output, batch.y)
            loss = loss / minibatch_multiplier

        if scaler is not None:
            # Scale loss for FP16
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * minibatch_multiplier
        batch_counter += 1

        # Only update weights every 'minibatch_multiplier' steps
        if batch_counter >= minibatch_multiplier:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            batch_counter = 0
            
    # Final step if there are remaining gradients
    if batch_counter > 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
            
    return total_loss / len(loader)

def val_loop_graph(model, loader, device, loss_fn=None):
    """
    Evaluates the GNN model on the validation set.
    """
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            edge_weight = batch.edge_attr if hasattr(batch, 'edge_attr') else None
            output = model(batch.x, batch.edge_index, edge_weight, batch.batch)
            
            if loss_fn is not None:
                loss = loss_fn(output, batch.y)
                total_loss += loss.item()

            probs = F.softmax(output, dim=1).cpu()
            
            y_true.extend(batch.y.cpu().tolist())
            y_pred.extend(probs.argmax(dim=1).tolist())
            y_scores.extend(probs[:, 1].tolist())
            
    avg_loss = total_loss / len(loader) if loss_fn is not None else None
    return y_true, y_pred, y_scores, avg_loss
