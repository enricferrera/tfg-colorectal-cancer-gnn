import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Local application imports
from models import models_attention
from dataset.loaders import AttnDataset

def train_loop(data_dict, idxs, model_classifier, dataset_classifier, loss_function_classifier, device,
               max_l, slide_index_dict, optimizer_classifier, epochs=30, minibatch_size=2, batch_size=12):
    """
    Train loop for the classifier model.
    """
    model_classifier.train()
    model_classifier.to(device)
    loss_function_classifier.to(device)
    
    td = dataset_classifier(data_dict, idxs, max_l, slide_index_dict)
    trainloader_classifier = DataLoader(td, batch_size=batch_size, shuffle=True, pin_memory=False)
    del td

    for epoch in range(epochs):
        batch_counter = 0
        for b in trainloader_classifier:
            x, y, extra_info, _ = b

            n_padding_batch = extra_info['n_padding']
            output, _ = model_classifier(x.to(device), n_padding=n_padding_batch.to(device), histodata=None)

            loss_clf = loss_function_classifier(output, y.long().to(device))
            loss_clf = loss_clf / minibatch_size  # Scale loss
            loss_clf.backward()

            batch_counter += 1
            if batch_counter >= minibatch_size:
                optimizer_classifier.step()
                optimizer_classifier.zero_grad()
                batch_counter = 0

    return [model_classifier]

def val_loop(data_dict, idxs, model_classifier, dataset_classifier, device, max_l, slide_index_dict, batch_size=12):
    """
    Validation loop for the classifier model.
    """
    model_classifier.to(device)
    model_classifier.eval()

    vd = dataset_classifier(data_dict, idxs, max_l, slide_index_dict)
    valoader_classifier = DataLoader(vd, batch_size=batch_size, shuffle=False, pin_memory=False)
    del vd

    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
    # In validation, we do not need to track operations to calculate gradients.
        for b in valoader_classifier:
            x, y, extra_info, _ = b
            n_padding_batch = extra_info['n_padding']

            output, _ = model_classifier(x.to(device), n_padding=n_padding_batch.to(device), histodata=None)
            probs = F.softmax(output, dim=1).cpu()

            y_true.extend(y.cpu().tolist())
            y_pred.extend(probs.argmax(dim=1).cpu().tolist())
            y_scores.extend(probs[:, 1].cpu().tolist())

    return y_true, y_pred, y_scores, {}

def run_cross_validation(patient_dict, weights, slides, device, n_folds=10, epochs=30, batch_size=24, minibatch_size=2):
    """
    Executes K-fold cross-validation.
    """
    # 1. Compute max_l and slide_index_dict
    max_l = 0
    for pat in patient_dict:
        patient_patch_size = len(patient_dict[pat]['megapatches'])
        if patient_patch_size > max_l:
            max_l = patient_patch_size
    
    print(f"Max patches per patient: {max_l}")

    unique_slide_list = list(set(slides))
    slide_index_dict = {word: index for index, word in enumerate(unique_slide_list)}

    # 2. Prepare patient and label lists
    patient_list = list(patient_dict.keys())
    label_list = [patient_dict[pat]['label'] for pat in patient_list]

    # 3. K-Fold Setup
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    pat_split = skf.split(np.array(patient_list), np.array(label_list))

    metrics = {k: [] for k in ['recall_0', 'recall_1', 'precision_0', 'precision_1', 'f1_0', 'f1_1', 'auc']}

    print("///////////////////////////////")
    for fold_num, (trf, vaf) in enumerate(pat_split, 0):
        patient_list_train = np.array(patient_list)[trf]
        patient_list_val = np.array(patient_list)[vaf]
        
        print(f"---- Fold {fold_num+1} ----")
        print(f"TR: {len(patient_list_train)} (N0: {pd.Series(label_list)[trf].value_counts().get(0, 0)}, N1+: {pd.Series(label_list)[trf].value_counts().get(1, 0)})")
        print(f"VA: {len(patient_list_val)} (N0: {pd.Series(label_list)[vaf].value_counts().get(0, 0)}, N1+: {pd.Series(label_list)[vaf].value_counts().get(1, 0)})")

        # Model initialization
        n_features = 1536
        out_ch = 2
        hid1 = 1024
        hid2 = hid1 // 2
        hid3 = hid2 // 2
        
        model = models_attention.Clf_head(max_l, n_features, out_ch=out_ch, hid1=hid1, hid2=hid2, hid3=hid3, dropout=0.3, attention_branches=6)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn_clf = torch.nn.CrossEntropyLoss(weight=weights.to(device))

        # Training
        train_params = {
            'data_dict': patient_dict,
            'idxs': patient_list_train,
            'model_classifier': model,
            'dataset_classifier': AttnDataset,
            'loss_function_classifier': loss_fn_clf,
            'device': device,
            'max_l': max_l,
            'slide_index_dict': slide_index_dict,
            'optimizer_classifier': opt,
            'epochs': epochs,
            'minibatch_size': minibatch_size,
            'batch_size': batch_size
        }
        
        trained_models = train_loop(**train_params)
        model = trained_models[0]

        # Validation
        val_params = {
            'data_dict': patient_dict,
            'idxs': patient_list_val,
            'model_classifier': model,
            'dataset_classifier': AttnDataset,
            'device': device,
            'max_l': max_l,
            'slide_index_dict': slide_index_dict,
            'batch_size': batch_size
        }
        
        y_true, y_pred, y_scores, _ = val_loop(**val_params)

        # Metrics
        val_auc = roc_auc_score(y_true, y_scores)
        metrics['recall_0'].append(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
        metrics['recall_1'].append(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        metrics['precision_0'].append(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
        metrics['precision_1'].append(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        metrics['f1_0'].append(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
        metrics['f1_1'].append(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        metrics['auc'].append(val_auc)

        print(f"Fold {fold_num+1} AUC VA: {val_auc:.4f}")
        print("----------------------------------------")

    print("\nAveraged Metrics over folds:")
    for k, vals in metrics.items():
        mean, std = np.mean(vals), np.std(vals)
        print(f"{k}: {mean:.4f} +- {std:.4f}")
    print("------------")
