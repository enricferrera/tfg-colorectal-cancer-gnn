# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 14:00:28 2021

@author: debora
"""
import time

import numpy as np
from sklearn import model_selection
from sklearn import preprocessing

from random import shuffle
import itertools
from ismember import ismember

import torch
import torch.nn as nn
import torch.optim as optim
import gc
import copy

#from datasets_util import *
from .datasets import *
from .train_util import *
from .train_losses import *
from .EarlyStopping import *
from .dice_score import *
 

from sklearn.preprocessing import label_binarize


def train_model(train_data_x, train_data_y, TRAIN_CONFIG, MODEL_CONFIG, 
                OPTIMIZER_CONFIG, x_valid=None, y_valid=None):

    ## Input Parameters
    transf=TRAIN_CONFIG['transf']
    batch_size=TRAIN_CONFIG['batch_size']
    n_epochs=TRAIN_CONFIG['n_epochs']
    shuffle=TRAIN_CONFIG['shuffle']
    model=MODEL_CONFIG['model']
   
    # DataLoaders
    train_ds=Standard_Dataset(train_data_x)
    train_dataloader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=shuffle,
                 )
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), 
                           lr=OPTIMIZER_CONFIG['initial_lr'])

    criterion =TRAIN_CONFIG['criterion']

    # Train the model
    train_params={}
    train_params['optimizer']=optimizer
    train_params['lr_scheduler ']=None 
    train_params['criterion']=criterion
    train_params['n_epochs']=n_epochs
       
    data_params={}
    data_params['train_dataloader']=train_dataloader

    model, avg_cost = standard_fit(model, train_params, data_params)
    
    # Empty GPU memory
    del train_dataloader.dataset
    del train_dataloader
    gc.collect()
    torch.cuda.empty_cache()
    
    return model, avg_cost,optimizer


def standard_fit(model, train_params, data_params):

    # Input Parameters
    
    optimizer=train_params['optimizer']
    lr_scheduler=train_params['lr_scheduler '] 
    criterion=train_params['criterion']
    n_epochs=train_params['n_epochs']
   
    
    train_dataloader=data_params['train_dataloader']
    if 'permute' in data_params.keys():  
        permute=data_params['permute']
    else:
        permute=0
    

    early_stopping = EarlyStopping(warm_up=60, patience=30,delta=optimizer.defaults['lr'])
    avg_cost = np.zeros([n_epochs])
    time_start = time.time()
    scaler = torch.amp.GradScaler('cuda')
    
    # Training the model for TOTAL_EPOCHS
    total_train_batch = len(train_dataloader)
    # training
    model.train()
    
    for epoch in range(n_epochs):
        index = epoch
       
        # evaluate model by nodule
        iter_train_dataset = iter(train_dataloader)

        for k in range(total_train_batch):
         
            # batch of nodes 
            seqs = next(iter_train_dataset)
            if list(model.parameters())[1].device.type=='cuda':
                 seqs = seqs.cuda()

            
            if model.name=='VAE':
                outputs,mu,logvar = model(seqs)  
                loss,_,_=criterion(seqs.view(seqs.size(0),-1),outputs.view(outputs.size(0),-1),mu,logvar)
                
            elif model.name=='AE':
                if permute:
                    outputs = model(seqs.permute(0, 3, 1, 2))  
                    outputs=outputs.permute(0,2, 3,1)
                else: 
                    outputs = model(seqs)  
               
                loss=criterion(outputs,seqs)
                                    
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # epoch loss
            avg_cost[index] += loss.item() / total_train_batch

            # Update learning rate
            if lr_scheduler is not None:
                lr_scheduler.step()
            #    lr_scheduler(None)
        
        print(avg_cost[index]) 
        if early_stopping(epoch+1, avg_cost[index], copy.deepcopy(model)):   
            print('Early Stop at' + str(epoch))
            break
    time_elapsed = time.time() - time_start
    display_elapsed_time(time_elapsed)
    
    # Free GPU memory
    outputs= outputs.cpu().detach().numpy()
    del outputs
    seqs= seqs.cpu().detach().numpy()
    del seqs
    
    gc.collect()
    torch.cuda.empty_cache()
    
    
    return model, avg_cost

