# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 16:13:15 2025

@author: debora
"""
import glob
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
# Torch
from torch.utils.data import DataLoader
import torch.optim as optim
import gc
import torch.nn.functional as F

# Onw Functions
CodeDir=r'D:\Experiments\Pathomics\CodeTFGVAEs'
os.chdir(CodeDir)
from TrainModels.datasets import Standard_Dataset
from Models.AutoEncoder_models import VAECNN as VAE
from TrainModels.train_losses import VAELoss
from TrainModels.train_AutoEncoder import standard_fit




def VAEConfigs(Config):
    net_paramsEnc={}
    net_paramsDec={}
    inputmodule_paramsDec={}
    net_paramsRep={}
    if Config=='1':
        # CONFIG1
        net_paramsEnc['block_configs']=[[32,32],[64,64]]
        net_paramsEnc['stride']=[[1,2],[1,2]]
        net_paramsDec['block_configs']=[[64,32],[32,inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsEnc['block_configs'][-1][-1]
     

        
    elif Config=='2':
        # CONFIG 2
        net_paramsEnc['block_configs']=[[32],[64],[128],[256]]
        net_paramsEnc['stride']=[[2],[2],[2],[2]]
        net_paramsDec['block_configs']=[[128],[64],[32],[inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsEnc['block_configs'][-1][-1]
   
        
    elif Config=='3':  
        # CONFIG3
        net_paramsEnc['block_configs']=[[32],[64],[64]]
        net_paramsEnc['stride']=[[1],[2],[2]]
        net_paramsDec['block_configs']=[[64],[32],[inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsEnc['block_configs'][-1][-1]
    
    return net_paramsEnc,net_paramsDec,inputmodule_paramsDec


#### 1. INPUT DATA
ResDir=r''
# VAE Train config 
batch_size=200
lr=10**(-3)
n_epochs=250

net_paramsEnc={}
net_paramsDec={}
net_paramsRep={}
inputmodule_paramsDec={}
inputmodule_paramsEnc={}

inputmodule_paramsEnc['num_input_channels']=1
inputmodule_paramsEnc['dim_input']=1
net_paramsEnc['block_configs']=[[16]]
net_paramsEnc['stride']=[[2]]
net_paramsDec['block_configs']=[[inputmodule_paramsEnc['num_input_channels']]]
net_paramsDec['stride']=net_paramsEnc['stride']
inputmodule_paramsDec['num_input_channels']=net_paramsEnc['block_configs'][-1][-1]
inputmodule_paramsDec['dim_input']=inputmodule_paramsEnc['dim_input']
net_paramsRep['z_dim']=16
net_paramsRep['h_dim']=10*net_paramsEnc['block_configs'][-1][-1]

### 1.1 LOAD DATA

X=np.random((10,256,256))
    
#### 2. TRAIN VAE/AE
kfold=0    
#### 2.1 VAE
train_ds=Standard_Dataset(X)
  
# VAE Train
batch_size=200
train_dataloader = DataLoader(
            train_ds,
            batch_size,
            shuffle=True,
             )

modelVAE=VAE(inputmodule_paramsEnc,net_paramsEnc,inputmodule_paramsDec,net_paramsDec,net_paramsRep)
modelVAE=modelVAE.to('cuda')
optimizer = optim.Adam(modelVAE.parameters(), 
                       lr=lr)
# 
criterion=VAELoss(ReconLoss='L1')
train_params={}
train_params['optimizer']=optimizer
train_params['lr_scheduler '] =None
train_params['criterion']=criterion
train_params['n_epochs']=n_epochs
data_params={}
data_params['train_dataloader']=  train_dataloader 
modelVAE, avg_costVAE = standard_fit(modelVAE, train_params, data_params)

# SAVE model immediately after training
ae_path = os.path.join(ResDir, f"fold_{kfold+1}_VAE.pt")
torch.save(modelVAE.to('cpu').state_dict(), ae_path)
ae_path = os.path.join(ResDir, f"fold_{kfold+1}_VAELoss")
np.savez(ae_path,avg_cost=avg_costVAE)
 
