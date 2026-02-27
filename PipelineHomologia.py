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

# labels of the different patients
bt_dict=patient_dict
patient_list=list(bt_dict.keys())
label_list=[]
for pat in bt_dict:
  label_list.append(bt_dict[pat]['label'])

