# -*- coding: utf-8 -*-
"""
Created on 15/02/2026

@author: Nil Arenos i Carles Sanchez
"""
### Pipeline for CLS processing using persistent homology #######
###################################################################


import numpy as np
import torch
import pickle
import random
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
import pandas as pd
from models import models_attention
from dataset.loaders import *
from dataset.loadCLS import *
from sklearn.model_selection import StratifiedKFold


############## TRAIN - VAL LOOPS ##################################################################
###################################################################################################

def train_loop(data_dict, idxs, model_classifier, dataset_classifier, loss_function_classifier, device,
               optimizer_classifier, epochs=30, minibatch_size=2, batch_size=12):

    model_classifier.train()
    model_classifier.to(device)
    loss_function_classifier.to(device)
    #Haureu de la vostre classe dataset per adaptar-la a la necessitat del model.
    td = dataset_classifier(data_dict, idxs, max_l, slide_index_dict)
    trainloader_classifier = DataLoader(td, batch_size=batch_size, shuffle=True, pin_memory=False)
    del td

    for epoch in range(epochs):

        batch_counter = 0
        for b in trainloader_classifier:
            x, y, extra_info, histodata = b

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

    trained_models = [model_classifier]

    return trained_models


def val_loop(data_dict, idxs, model_classifier, dataset_classifier, device, batch_size=12):

    model_classifier.to(device)
    model_classifier.eval()

    vd = dataset_classifier(data_dict, idxs, max_l, slide_index_dict)

    valoader_classifier = DataLoader(vd, batch_size=batch_size, shuffle=True, pin_memory=False)
    del vd

    attention_output = {}

    y_true, y_pred, y_scores = [], [], []

    for b in valoader_classifier:
        x, y, extra_info, histodata = b

        # Extra info deglossed:
        # hosps_batch = extra_info['hospital']
        # pats_batch = extra_info['patient']
        # slds_batch = extra_info['slides']
        # coords_batch = extra_info['coords']
        # affectation_batch = extra_info['coords']
        n_padding_batch = extra_info['n_padding']

        output, attention_scores = model_classifier(x.to(device), n_padding=n_padding_batch, histodata=None)

        probs = F.softmax(output, dim=1).cpu()

        y_true.extend(y.cpu().tolist())

        y_pred.extend(probs.argmax(dim=1).cpu().tolist())
        y_scores.extend(probs[:, 1].cpu().tolist())



    return y_true, y_pred, y_scores, attention_output

############## END  TRAIN - VAL LOOPS ##################################################################
###################################################################################################

#------FIXED SEEDS for reproducibility-------
r_seed=123

def fix_seeds(r_seed=123):
  torch.manual_seed(r_seed)
  np.random.seed(r_seed)
  random.seed(r_seed)
  torch.cuda.manual_seed(r_seed)
  torch.backends.cudnn.enabled=False
  torch.backends.cudnn.deterministic=True
  
# -----------------------
# CONFIGURATION AND DATA LOADING
fix_seeds(r_seed=123)


############# LOAD CLS, Data paths and indexes ###################
##################################################################

npz_path=r"..\Data\cls_ALL"

print('------ Database totals...-------')
pat_nx_dict, pat_histodata_dict, features, affectation, hospitals, patients, slides, coords, paths= loadCLSMetadata(npz_path)


## Intersecció entre imatges processades i metadades associades 370 a 230 pacients.
print('------ filtering out patients without diagnosis...-------')
patient_dict=patient_dict_builder(features, affectation, hospitals, patients, slides, coords, paths, pat_nx_dict, pat_histodata_dict)


# Busquem quantes mostres té el pacient amb més mostres
max_l=0
for pat in patient_dict:
  patient_patch_size=len(patient_dict[pat]['megapatches'])
  if patient_patch_size > max_l:
    max_l=patient_patch_size
    
print(max_l) # mostres del pacient amb mes mostres


### MAP EACH SLIDE TO A NUMBER SO THAT THE DATALOADER CAN TAKE THEM
unique_slide_list = list(set(slides))
slide_index_dict={word: index for index, word in enumerate(unique_slide_list)}
inv_slide_index_dict = {v: k for k, v in slide_index_dict.items()}


fix_seeds(r_seed=123)

bt_dict=patient_dict
patient_list=list(bt_dict.keys())
label_list=[]
for pat in bt_dict:
  label_list.append(bt_dict[pat]['label'])

# Preparar les dades per poder entrenar els diferents models.
# La preparació de les dades es pot fer previa a la creació dels loaders i guardar a disc (offline) o dins de la propia classe dels loaders (online).
# Tot depèn dels recursos que necessiteu.
# Utilitzar la llibreria torch.geometric per crear els graphs i els batch.
#           from torch_geometric.data import Data,Batch  --> busqueu com implementar-ho.


# ================= STEP 4: Training & Validation =================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weight_0, weight_1 = calculate_class_weights(label_list)
print("weights: ", weight_0, weight_1)
weights=torch.tensor([float(weight_0), float(weight_1)])

########## Parameters #################

n_folds=10
minibatch_size = 2 # cada quants batch s'actualitza el gradient. Si batch_size gran millor deixar-lo a 1
epochs = 30#10 #50
batch_size = 24

##########  --------- DataSpliting and k-fold validation ----------- ###############

skf = StratifiedKFold(n_folds, shuffle=True, random_state=0)

pat_split=skf.split(np.array(patient_list), np.array(label_list))
print("///////////////////////////////")
attention_output={}
metrics = {k:[] for k in ['recall_0','recall_1','precision_0','precision_1','f1_0','f1_1','auc']}

for fold_num, (trf, vaf) in enumerate(pat_split, 0):
    
    ##### From indices to patients #######
    patient_list_train=np.array(patient_list)[trf]
    
    patient_list_val=np.array(patient_list)[vaf]
    print("---- TR: ", len(patient_list_train), " ------")
    print("- N0: ", pd.Series(label_list)[trf].value_counts()[0])
    print("- N1+: ", pd.Series(label_list)[trf].value_counts()[1])
    print("---- VA: ", len(patient_list_val), " ------")
    print("- N0: ", pd.Series(label_list)[vaf].value_counts()[0])
    print("- N1+: ", pd.Series(label_list)[vaf].value_counts()[1])
    
    n_features=1536
    out_ch=2
    
    loss_fn_clf = torch.nn.CrossEntropyLoss(weight=weights)

    # Haureu de canviar el model pel que volgueu utilitzar.
    hid1=1024
    hid2=int(hid1/2)
    hid3=int(hid2/2)
    mx_patches=max_l
    model = models_attention.Clf_head(mx_patches, n_features, out_ch=2, hid1=hid1, hid2=hid2, hid3=hid3, dropout=0.3, attention_branches=6)
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    ##### TRAINING ##############
    
    
    fold_losses=[]
    train_params={
    'data_dict': bt_dict,
    'idxs': patient_list_train,
    'model_classifier': model,
    'dataset_classifier': AttnDataset, # Això s'haurà de substituir pel vostre dataset.
    'device': device,
    'loss_function_classifier': loss_fn_clf,
    'optimizer_classifier': opt,
    'epochs': epochs,
    'minibatch_size': minibatch_size,
    'batch_size': batch_size
    }
    
    
    #affectation_extractor, model = train_loop(**train_params)
    trained_models = train_loop(**train_params)
    if len(trained_models)==2:
      affectation_extractor, model= trained_models
    else:
      model= trained_models[0]
    
    ##### VALIDATION ##########
    
    val_params={
    'data_dict': bt_dict,
    'idxs': patient_list_val,
    'model_classifier': model,
    'dataset_classifier': AttnDataset,#AttnDataset_SLIDE, #AttnDataset,
    'device': device,
    'batch_size': batch_size
    }
    
    y_true, y_pred, y_scores, partial_att_dict = val_loop(**val_params)
    
    
    val_auc=roc_auc_score(y_true, y_scores)
    # record metrics
    metrics['recall_0'].append(recall_score(y_true, y_pred, pos_label=0)) 
    metrics['recall_1'].append(recall_score(y_true, y_pred, pos_label=1)) 
    metrics['precision_0'].append(precision_score(y_true, y_pred, pos_label=0)) 
    metrics['precision_1'].append(precision_score(y_true, y_pred, pos_label=1)) 
    metrics['f1_0'].append(f1_score(y_true, y_pred, pos_label=0))
    metrics['f1_1'].append(f1_score(y_true, y_pred, pos_label=1))
    metrics['auc'].append(val_auc)
    
    print(f"Fold {fold_num+1} AUC VA: {val_auc:.4f}")
    print("----------------------------------------") 
    

print("---------")
print("\nAveraged Metrics over folds:")
for k, vals in metrics.items():
    mean, std = np.mean(vals), np.std(vals)
    print(f"{k}: {mean:.4f} +- {std:.4f}")
print("------------")



##########################################
#-----------------------------------------
##########################################
'''
y_true=[]
y_pred=[]
y_scores=[]
for pat in results_patient_dict:
  label=results_patient_dict[pat]['label']
  soft_voting=np.mean(results_patient_dict[pat]['outputs'], axis=0)
  
  
  preds=np.argmax(soft_voting)
  
  y_true.append(label)
  y_pred.append(preds)
  y_scores.append(soft_voting[1])


val_auc=roc_auc_score(y_true, y_scores)
metrics_boot = {k:[] for k in ['recall_0','recall_1','precision_0','precision_1','f1_0','f1_1','auc']}
metrics_boot['recall_0'].append(recall_score(y_true, y_pred, pos_label=0)) 
metrics_boot['recall_1'].append(recall_score(y_true, y_pred, pos_label=1)) 
metrics_boot['precision_0'].append(precision_score(y_true, y_pred, pos_label=0)) 
metrics_boot['precision_1'].append(precision_score(y_true, y_pred, pos_label=1)) 
metrics_boot['f1_0'].append(f1_score(y_true, y_pred, pos_label=0))
metrics_boot['f1_1'].append(f1_score(y_true, y_pred, pos_label=1))
metrics_boot['auc'].append(val_auc)


print("---------")
print("\nMetrics across BOOTSTRAPS:")
for k, vals in metrics_boot.items():
    print(f"{k}: {vals[0]:.4f}")
print("------------")
  
############# Voting between all bootstraps 
#  with open('attention_dict.pkl', 'wb') as f:
#      pickle.dump(attention_output, f)
'''