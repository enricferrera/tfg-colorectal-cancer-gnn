"""
Created on Tue Feb 15 19:18:03 2022

@author: debora
"""

import torch
from torch.nn.modules.container import ModuleList
import copy
import gc

# saving the model
def save_full_model(model, save_full_path):
    """
        Save the entire model
    """    
    torch.save(model, save_full_path)
    print(f'\tEntire model saved at  ==> {save_full_path}')
    
# loading the model
def load_full_model(save_full_path):
    """
        Load the entire model
    """    
    model = torch.load(save_full_path)
    print(f'\tEntire model loaded  <== {save_full_path}')
    
    return model

def clear_gpu(tensor):
    # Clear GPU memory in preparation for the next model training
    del tensor
    gc.collect()
    torch.cuda.empty_cache() 

def Cuda2CPU(model): 
    params =model.state_dict()
    for name in params.keys() :
       
        params[name].cpu().detach().numpy()

    model.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()
    
    return model
    
def clone_model(module, N):
    
    return ModuleList([copy.deepcopy(module) for i in range(N)])

# Alternativa:
#def clone_model(model,model_list, N):
#    for k in arange(N):
#        model_list[k].load_state_dict(model.state_dict)
