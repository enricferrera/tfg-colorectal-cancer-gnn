# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 19:18:03 2022

@author: debora
"""
import math
import numpy as np
import itertools

import torch
from torch import nn as nn
import torch.nn.functional as F
from torch import Tensor
from numpy.matlib import repmat

from models_init import *
from NetBlocks import _CNNBlock,LinearBlock



#  BackBone Modules
class Encoder(nn.Module):
    r"""AutoEncoderCNN model class
    `".
    """

    def __init__(self, inputmodule_params,net_params):
        super().__init__()
        
        
        num_input_channels=inputmodule_params['num_input_channels']
        self.dim=net_params['dim']

            
        drop_rate=net_params['drop_rate']
        block_configs=net_params['block_configs']
        n_blocks=len(block_configs)
        
        # Encoder
        self.encoder=nn.Sequential(          
            )
        outchannels_encoder=[]
        for i in np.arange(n_blocks):
            block = _CNNBlock(
                num_input_channels=num_input_channels,
                drop_rate=drop_rate,
                block_config=block_configs[i], 
                dim=self.dim
                
            )
            self.encoder.add_module("cnnblock%d" % (i + 1), block)
            if self.dim==2:
                self.encoder.add_module("mxpool%d" % (i + 1), 
                                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            elif self.dim==3:
                self.encoder.add_module("mxpool%d" % (i + 1), 
                                         nn.MaxPool3d(kernel_size=2, stride=2, padding=0))
            num_input_channels=block_configs[i][-1] 
           # outchannels_encoder.append(num_input_channels)
           
          
               
    def forward(self, x: Tensor) -> Tensor:
        
        x=self.encoder(x)

        return x


##### Classification Models
class ClassificationCNN(nn.Module):
    r"""AutoEncoderCNN model class
    `".
    """

    def __init__(self, inputmodule_params,net_params,outmodule_params):
        super().__init__()
        
                

        # Encoder
        self.encoder=Encoder(inputmodule_params,net_params)
        # 
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(outmodule_params['poolsze'], 
                                                    outmodule_params['poolsze']))
        # FC Net
        outmodule_inputparams={}
        outmodule_inputparams['n_inputs']=outmodule_params['poolsze']*outmodule_params['poolsze']*net_params['block_configs'][-1][-1]

        self.fc=LinearBlock(outmodule_inputparams,outmodule_params)
        
        # weight init
        init_weights_xavier_normal(self)
    def forward(self, x: Tensor) -> Tensor:
        
        
        x=self.encoder(x)
        x=self.avgpool(x)
        x=x.flatten(-3,-1)
        x=self.fc(x)
        
        return F.softmax(x,dim=1)


    



