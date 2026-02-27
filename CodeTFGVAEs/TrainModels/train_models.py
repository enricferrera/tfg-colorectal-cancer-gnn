# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 14:00:28 2021

@author: debora
"""
import time

import numpy as np
from sklearn import model_selection
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import gc
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup
import copy

from DataSets.datasets_util import *
from DataSets.datasets import *
from TrainModels.train_util import *
from TrainModels.train_losses import *
from TrainModels.EarlyStopping import *

 

from sklearn.preprocessing import label_binarize

def  TrainPipeLine(train_data_x,train_data_y,
                         MODEL_CONFIG,TRAIN_CONFIG,OPTIMIZER_CONFIG):

        # 1) Preprocessing
        scaler=TRAIN_CONFIG['scaler']
        if scaler is not None:
              # scaler=scalers_fit(train_data_x)
              scaler = preprocessing.StandardScaler()
              train_data_x = scaler.fit_transform(train_data_x)
              
        train_data_y = train_data_y.astype(np.int64)
    
        # 2). train model 
        
        # 2.1). train-validation sets    
        x_train, x_valid, y_train, y_valid = model_selection.train_test_split(
            train_data_x, train_data_y,
            test_size=TRAIN_CONFIG['test_size'],                            
            random_state=123)
                                            
            
        # 2.2) Model Fit
        model, optimizer, avg_cost = train_model(x_train, y_train, TRAIN_CONFIG, MODEL_CONFIG, 
                    OPTIMIZER_CONFIG, x_valid, y_valid )
        
        return  model, optimizer, avg_cost, scaler


def DefineLoss(y_train,loss):
    
    criterion=[]
    if loss['type']=='CrossEntW':
        weight=classes_weight(y_train)
        
        criterion  =  torch.nn.CrossEntropyLoss(weight=weight)
        
   
    elif loss['type']=='CrossEnt':
         
            
        criterion  =  torch.nn.CrossEntropyLoss()
  #      criterion  =  torch.nn.CrossEntropyLoss()
    elif loss['type']=='BinCrossEnt':
        weight=classes_weight_binary(y_train)
        
        criterion  =  torch.nn.BCELoss(weight=weight)
    elif loss['type']=='MultiLab':
        NRadWordEncoding=loss['NRadWordEncoding']
        NCat=len(NRadWordEncoding)-1
        weight=[]
        for k in np.arange(NCat):
            y_cat=y_train[:,NRadWordEncoding[k]:NRadWordEncoding[k+1]]
            weight.append(classes_weight_binary(y_cat))
        criterion=MultiLabLoss(weight=weight,CatSplit=NRadWordEncoding)
        
    elif loss['type']=='MSE':
        criterion =torch.nn.MSELoss(reduction='mean')
    elif loss['type']=='TripleLoss':
        criterion=TripletMarginLoss(lamb=loss['lamb'])
    elif loss['type']=='BarlowTwins':
        criterion=BarlowTwins()
    else:

        weight=classes_weight_binary(y_train[0])
        criterion = BCEMSELoss(weight=weight,alfa=loss['alfa'])
    
    return criterion

def Define_Optimizer(OPTIMIZER_CONFIG):
    
    
   
    # warmup_iteration = 10 #3
    # initial_lr = 1e-3
    # warmup_initial_lr = 1e-5 # 0
    
    scheduler_config=OPTIMIZER_CONFIG['scheduler_config']
    warmup_config= OPTIMIZER_CONFIG['warmup_config']
    
    optimizer = optim.Adam(OPTIMIZER_CONFIG['model'].parameters(), lr=OPTIMIZER_CONFIG['initial_lr'])
    
    scheduler=None
    if len(scheduler_config.keys())>0:  
        if scheduler_config['scheduler']=='StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_config['step_size'], gamma=0.1)
        elif scheduler_config['scheduler']=='CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=
                                                                   scheduler_config['T_max'], eta_min=0)
        elif scheduler_config['scheduler']=='CosineAnnealingWarmRestarts':
            # optimizer = optim.SGD(OPTIMIZER_CONFIG['model'].parameters(), 
            #                       lr=OPTIMIZER_CONFIG['initial_lr'], momentum=0.9)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                       T_0=scheduler_config['T_max'], T_mult=1, eta_min=0.001, last_epoch=-1)
        
    if len(warmup_config.keys())>0:  
        lr_scheduler = create_lr_scheduler_with_warmup(
            scheduler,warmup_start_value=warmup_config['warmup_initial_lr'],
                                                       warmup_duration=warmup_config['warmup_iteration'],
                                                       warmup_end_value=OPTIMIZER_CONFIG['initial_lr'])
    else: 
        lr_scheduler=scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6,8,9], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    return optimizer, lr_scheduler

def train_model(x_train, y_train, TRAIN_CONFIG, MODEL_CONFIG, 
                OPTIMIZER_CONFIG, x_valid=None, y_valid=None):

    ## Input Parameters
    transf=TRAIN_CONFIG['transf']
    batch_size=TRAIN_CONFIG['batch_size']
    n_epochs=TRAIN_CONFIG['n_epochs']
    pth_full_name= TRAIN_CONFIG['pth_full_name']
    dataset_type=TRAIN_CONFIG['dataset_type']
    loss=TRAIN_CONFIG['loss']
    shuffle=TRAIN_CONFIG['shuffle']
    model=MODEL_CONFIG['model']
    
    
    # DataLoaders
    if dataset_type=='balanced':
        train_dataloader = create_dataloader_balanced(x_train, y_train, transf, batch_size,shuffle=shuffle)
    elif dataset_type=='paired':
        train_dataloader = create_Paired_dataloader(x_train, y_train, transf, batch_size,shuffle=shuffle)
    else:
        train_dataloader = create_dataloader(x_train, y_train, transf, batch_size,shuffle=shuffle)
    
    valid_dataloader = None
    if x_valid is not None and y_valid is not None:
        if balanced==True:
            valid_dataloader = create_dataloader_balanced(x_valid, y_valid, transf, batch_size, shuffle=False)
        else:
            valid_dataloader = create_dataloader(x_valid, y_valid, transf, batch_size, shuffle=False)

    # Define optimizer
    OPTIMIZER_CONFIG['model']=model
    optimizer, lr_scheduler=Define_Optimizer(OPTIMIZER_CONFIG)
       
    
    criterion=DefineLoss(y_train,loss)
    # Train the model
    
    train_params={}
    train_params['optimizer']=optimizer
    train_params['lr_scheduler ']=lr_scheduler 
    train_params['criterion']=criterion
    train_params['n_epochs']=n_epochs
    data_params={}
    data_params['train_dataloader']=train_dataloader
    data_params['valid_dataloader']=valid_dataloader
    
    # model, avg_cost = standard_fit_balanced(model, train_params, data_params,
    #                             verbose=False,
    #                             save_path=pth_full_name,
    #                             best_val_loss=None)
    
    model, avg_cost = standard_fit(model, train_params, data_params,
                                verbose=False,
                                save_path=pth_full_name,
                                best_val_loss=None)
    del train_dataloader.dataset
    del train_dataloader
    if valid_dataloader is not None:
        del valid_dataloader
        del valid_dataloader.dataset
        
    gc.collect()
    
    return model, optimizer, avg_cost



def standard_fit(model, train_params, data_params,verbose=1, save_path=None,
                best_val_loss=None):

    # Input Parameters
    
    optimizer=train_params['optimizer']
    lr_scheduler=train_params['lr_scheduler '] 
    criterion=train_params['criterion']
    n_epochs=train_params['n_epochs']
    
    train_dataloader=data_params['train_dataloader']
    valid_dataloader=data_params['valid_dataloader']
    
    if best_val_loss is None:
        best_val_loss = float("Inf")

    early_stopping = EarlyStopping(warm_up=60, patience=30)
    avg_cost = np.zeros([n_epochs])
    time_start = time.time()
    scaler = torch.cuda.amp.GradScaler()
    
    # Training the model for TOTAL_EPOCHS
    total_train_batch = len(train_dataloader)
    for epoch in range(n_epochs):
        index = epoch
        cost = np.zeros(6, dtype=np.float32)

        # training
        model.train()
        iter_train_dataset = iter(train_dataloader)
        for k in range(total_train_batch):
            
            # batch 
            seqs, targets = next(iter_train_dataset)
            
            if isinstance(targets,tuple) or isinstance(targets,list):
                seqs, targets = seqs.cuda(), (targets[0].cuda(),targets[1].cuda())
            elif isinstance(seqs,tuple) or isinstance(seqs,list):
                targets,seqs = targets.cuda(), (seqs[0].cuda(),seqs[1].cuda())
            else:
                seqs, targets = seqs.cuda(), targets.cuda()
            
            # model evaluation
            optimizer.zero_grad()
            outputs = model(seqs)
            
            loss = criterion(outputs,targets)
            if isinstance(loss,tuple):
                loss=loss[0]
            # L2 regularization
            # l1 = 0
            # for p in model.parameters():
            #   l1 = l1 + p.norm(2).sum()
            # loss=loss+0.01*l1
            # backpropagation 
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # loss.backward()
            # optimizer.step()
            
            # epoch loss
        #    print(loss/total_train_batch)
            avg_cost[index] += loss.item() / total_train_batch
       
        # Update learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()
        #    lr_scheduler(None)
         
        print(avg_cost[index])
        # if early_stopping(epoch+1, avg_cost[index], copy.deepcopy(model)):   
        #     print('Early Stop at' + str(epoch))
        #     break
        
        # validation
        if valid_dataloader is not None:
            total_valid_batch = len(valid_dataloader)
            model.eval()
            with torch.no_grad():
                iter_valid_dataset = iter(valid_dataloader)
                for k in range(total_valid_batch):
                    seqs, targets = next(iter_valid_dataset)
                    seqs, targets = seqs.cuda(),  (targets[0].cuda(),targets[1].cuda())
                    outputs = model(seqs)
                    
                    loss = criterion(outputs, targets)
                    if isinstance(loss,tuple):
                        loss=loss[0]

                    cost[0] = loss.item()
                    avg_cost[index, 3:] += cost[3:] / total_valid_batch

            # save model
            if (save_path is not None) & (avg_cost[index, 3] < best_val_loss):
                best_val_loss = avg_cost[index, 3]
                save_checkpoint(model, optimizer, best_val_loss, epoch + 1, pth_full_name + '_ep_' + str(epoch + 1) + ext)


        if verbose:
            print(f'Epoch [{epoch + 1}/{n_epochs}] | TRAIN: Loss:{avg_cost[index,0]:.2f} Acc:{avg_cost[index,1]:.2f} Pre:{avg_cost[index,2]:.2f} |' +
              f' TEST: Loss:{avg_cost[index,3]:.2f} Acc:{avg_cost[index,4]:.2f} Pre:{avg_cost[index,5]:.2f}')

    time_elapsed = time.time() - time_start
    display_elapsed_time(time_elapsed)
    
    # Try to free GPU memory
#    seqs.cpu()
#    targets[0].cpu()
#    targets[1].cpu()
#    outputs[0].cpu()
#    outputs[1].cpu()
   
    del outputs
    del seqs
    del targets
    
    return model, avg_cost

def UpdateModelObjective(model,seqs, targets,criterion,optimizer, obj):
        
    outputs = model(seqs.T)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss[obj].backward()
    optimizer.step()
    cost = loss[obj].item()
    
    for out in outputs:
        out.cpu().detach()
    for out in loss:
        out.cpu().detach()
    
    del loss
    del outputs
    
    return cost

def multiobjective_fit(model, optimizer, criterion, train_dataloader, valid_dataloader,
                n_epochs, verbose=1, save_path=None,
                best_val_loss=None):

    if best_val_loss is None:
        best_val_loss = float("Inf")

    avg_cost = np.zeros([n_epochs, 4], dtype=np.float32)
    time_start = time.time()

    # Training the model for TOTAL_EPOCHS
    total_train_batch = len(train_dataloader)
    for epoch in range(n_epochs):
        index = epoch
        

        # training
        model.train()
        iter_train_dataset = iter(train_dataloader)
        for k in range(total_train_batch):
            seqs, targets = next(iter_train_dataset)
            seqs, targets = seqs.cuda(), targets.cuda()
     
            # multiobjective 
            # Obj1
            obj=1
            cost=UpdateModelObjective(model,seqs, targets,criterion, optimizer,obj)
            avg_cost[index, obj-1] += cost / total_train_batch
            # Obj2
            obj=2
            cost=UpdateModelObjective(model,seqs, targets,criterion,optimizer, obj)
            avg_cost[index, obj-1] += cost / total_train_batch

        # validation
        if valid_dataloader is not None:
            total_valid_batch = len(valid_dataloader)
            model.eval()
            with torch.no_grad():
                iter_valid_dataset = iter(valid_dataloader)
                for k in range(total_valid_batch):
                    seqs, targets = next(iter_valid_dataset)
                    seqs, targets = seqs.cuda(), targets.cuda()
                                # multiobjective 
                    # Obj1
                    obj=1
                    cost=UpdateModelObjective(model,seqs, targets,criterion,optimizer,obj)
                    avg_cost[index, 2] += cost / total_train_batch
                    # Obj2
                    obj=2
                    cost=UpdateModelObjective(model,seqs, targets, criterion,optimizer, obj)
                    avg_cost[index, 3] += cost / total_train_batch

            # save model
            if (save_path is not None) & (avg_cost[index, 0] < best_val_loss):
                best_val_loss = avg_cost[index, 0]
                save_checkpoint(model, optimizer, best_val_loss, epoch + 1, pth_full_name + '_ep_' + str(epoch + 1) + ext)


        if verbose:
            print(f'Epoch [{epoch + 1}/{n_epochs}] | TRAIN: Loss1:{avg_cost[index,0]:.2f} Loss2:{avg_cost[index,1]:.2f}' +
              f' TEST: Loss1:{avg_cost[index,2]:.2f} Loss2:{avg_cost[index,4]:.2f}')

    time_elapsed = time.time() - time_start
    display_elapsed_time(time_elapsed)
    
     # Try to free GPU memory
    seqs.cpu().detach()
    targets.cpu().detach()

    del seqs
    del targets
    
    return model, avg_cost