# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 16:13:15 2025

@author: debora
"""
import glob
import os
import numpy as np
#import torch
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
# Torch
from torch.utils.data import DataLoader
import torch


# Onw Functions
#CodeDir=r'D:\Experiments\DigiPatics\Results_TFM_Pere\Code'
CodeDir=r'C:\DADES\Experiments\Uncertainty\Code'
os.chdir(CodeDir)
from TrainModels.datasets import Standard_Dataset
from Models.AutoEncoder_models import VAECNN as VAE
from TrainModels.test_models import eval_VAE


def VAEConfigs():
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
    
    return net_paramsEnc,net_paramsDec,net_paramsRep,inputmodule_paramsDec,inputmodule_paramsEnc


#### 1. INPUT DATA
net_paramsEnc,net_paramsDec,net_paramsRep,inputmodule_paramsDec,inputmodule_paramsEnc=VAEConfigs()


### 1.1 LOAD DATA
DataDir=r''
ResDir=r''
data=np.random((10,256,256))
#### 3. TEST VAE/AE
kfold=0

### 3.1 VAE
# VAE Model
modelVAE=VAE(inputmodule_paramsEnc,net_paramsEnc,inputmodule_paramsDec,net_paramsDec,net_paramsRep)
modelVAE=modelVAE.to('cuda')
ae_path = os.path.join(ResDir,f"fold_{kfold+1}_VAE.pt")
state = torch.load(ae_path)
modelVAE.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
latentKL,reconError=eval_VAE(modelVAE,data)

    
### 4. SAVE ERRORS
ae_path = os.path.join(ResDir, 'KLReconVAE')
np.savez(ae_path,latentKL=latentKL,reconError=reconError)



    


