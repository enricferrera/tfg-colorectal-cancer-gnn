import torch
from matplotlib import pyplot as plt
import numpy as np
#import SSIM
import pandas as pd
import os

import glob
import torchvision
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


  
class AE(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_input_layer0 = torch.nn.Conv2d(3,3,3, padding=1, stride=8)
        self.encoder_input_layer1 = torch.nn.Conv2d(3,32,3, padding=1)
        self.encoder_hidden_layer2 = torch.nn.Conv2d(32,64,3, stride=2, padding=1)
        self.encoder_hidden_layer3 = torch.nn.Conv2d(64,128,3, stride=2, padding=1)
        self.encoder_hidden_layer4 = torch.nn.Conv2d(128,256,3, stride=2, padding=1)
        
        self.decoder_hidden_layer0 = torch.nn.ConvTranspose2d(256,128,3, stride=2, padding=1)
        self.decoder_hidden_layer1 = torch.nn.ConvTranspose2d(128,64,3, stride=2, padding=1)
        self.decoder_hidden_layer2 = torch.nn.ConvTranspose2d(64,32,3, stride=2, padding=1)
        self.decoder_output_layer = torch.nn.ConvTranspose2d(32,3,3, padding=1)
        self.decoder_output_layerto4096 = torch.nn.ConvTranspose2d(3,3,3, padding=1, stride=8)
        
        self.encoder=nn.Sequential(
        #self.encoder_input_layer0,
        self.encoder_input_layer1,
        self.encoder_hidden_layer2,
        self.encoder_hidden_layer3,
        self.encoder_hidden_layer4,
        nn.LeakyReLU(),
        )
        
        self.decoder=nn.Sequential(
        self.decoder_hidden_layer0,
        self.decoder_hidden_layer1,
        self.decoder_hidden_layer2,
        self.decoder_output_layer,
        nn.Sigmoid(),
        )

    
    def forward(self, features):
      '''
      activation=self.encoder_input_layer0(features)
      activation=self.encoder_input_layer1(activation)
      print(activation.shape)
      '''
      latent_space=self.encoder(features)
      sze_enc=latent_space.shape[-1]
      
      activation = self.decoder_hidden_layer0(latent_space,output_size=(sze_enc*2,sze_enc*2))
      activation = torch.nn.functional.leaky_relu(activation)
      activation = self.decoder_hidden_layer1(activation,output_size=(sze_enc*4,sze_enc*4))
      activation = torch.nn.functional.leaky_relu(activation)
      #print("d1",activation.shape)
      activation = self.decoder_hidden_layer2(activation,output_size=(sze_enc*8,sze_enc*8))
      activation = torch.nn.functional.leaky_relu(activation)
      #print("d2",activation.shape)
      activation = self.decoder_output_layer(activation)
      #activation = self.decoder_output_layerto4096(activation, output_size=(4096,4096))
      activation = torch.sigmoid(activation)
      
      
      #activation=self.decoder(latent_space)
      
      return activation, latent_space
      
      
      
      '''
      activation = self.decoder_hidden_layer0(activation,output_size=(sze_enc*2,sze_enc*2))
      activation = torch.nn.functional.leaky_relu(activation)
      activation = self.decoder_hidden_layer1(activation,output_size=(sze_enc*4,sze_enc*4))
      activation = torch.nn.functional.leaky_relu(activation)
      #print("d1",activation.shape)
      activation = self.decoder_hidden_layer2(activation,output_size=(sze_enc*8,sze_enc*8))
      activation = torch.nn.functional.leaky_relu(activation)
      #print("d2",activation.shape)
      activation = self.decoder_output_layer(activation)
      #activation = torch.sigmoid(activation)
      #print("d3",activation.shape)
      return activation
      '''
        
      
class AffectationExtractor(nn.Module):
    def __init__(self, in_ch, out_ch=2, hid1=1024, hid2=512, hid3=128, dropout=0.3):
        super().__init__()
        
        self.lin = torch.nn.Sequential(
            torch.nn.Linear(in_ch, hid1),
            torch.nn.LayerNorm(hid1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(dropout),
            
            torch.nn.Linear(hid1, hid2),
            torch.nn.LayerNorm(hid2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(dropout),
            
            torch.nn.Linear(hid2, hid3),
            torch.nn.LayerNorm(hid3),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(dropout),
            )
        self.head=torch.nn.Linear(hid3, out_ch)


    def forward(self, x):
        
        x = x.squeeze(0)
        
        last_dim=self.lin(x)
        
        out=self.head(last_dim)
        return out, last_dim 
       
class Attention(nn.Module):
    def __init__(self, in_ch=1536, ATTENTION_BRANCHES=1):
        super().__init__()
        self.M = in_ch
        self.L = int(self.M/2)
        
        '''
        self.M = 128
        self.L = 64
        '''
        self.ATTENTION_BRANCHES = ATTENTION_BRANCHES


        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )


    def forward(self, x, n_padding=None, mx_patches=None):
        
        H = x
        
        A = self.attention(H)  # KxATTENTION_BRANCHES
        
        A = torch.transpose(A, 1, 2)  # ATTENTION_BRANCHESxK
        
        if n_padding!=None and mx_patches!=None:
          ### masking 
          list_of_masks=[]
          for value in n_padding:
            mask_zeroes=torch.zeros(mx_patches-int(value), self.ATTENTION_BRANCHES, dtype=torch.bool)
            mask_ones=torch.ones(int(value), self.ATTENTION_BRANCHES, dtype=torch.bool)
            
            mask=torch.cat((mask_zeroes, mask_ones))
            mask=mask.T
            list_of_masks.append(mask)
        
          mask=torch.stack(list_of_masks, dim=0)
          
          
          
          #A[mask] = float('-inf')
          A[mask] = float(0)         
        
        
        
        A = F.softmax(A, dim=2)  # softmax over K
        
        Z = torch.bmm(A, H)  # ATTENTION_BRANCHESxM
        
        Z=torch.reshape(Z, (-1, Z.shape[2]*self.ATTENTION_BRANCHES))
         
        return Z, A

class GatedAttention(nn.Module):
    def __init__(self, in_ch=1536, ATTENTION_BRANCHES=1):
        super().__init__()
        self.M = in_ch
        self.L = int(self.M/2)
        '''
        self.M = 128
        self.L = 64
        '''
        self.ATTENTION_BRANCHES = ATTENTION_BRANCHES

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )
        
        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)


    def forward(self, x, n_padding=None, mx_patches=None):
        
        H = x
        
        A_V = self.attention_V(H)  # KxL
        A_U = self.attention_U(H)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 2)  # ATTENTION_BRANCHESxK
        
        if n_padding!=None and mx_patches!=None:
          ### masking 
          list_of_masks=[]
          for value in n_padding:
            mask_zeroes=torch.zeros(mx_patches-int(value), self.ATTENTION_BRANCHES, dtype=torch.bool)
            mask_ones=torch.ones(int(value), self.ATTENTION_BRANCHES, dtype=torch.bool)
            
            mask=torch.cat((mask_zeroes, mask_ones))
            mask=mask.T
            list_of_masks.append(mask)
        
          mask=torch.stack(list_of_masks, dim=0)
          
          #A[mask] = float('-inf')
          A[mask] = float(0)         
        
        A = F.softmax(A, dim=2)  # softmax over K
        
        Z = torch.bmm(A, H)  # ATTENTION_BRANCHESxM
        
        Z=torch.reshape(Z, (-1, Z.shape[2]*self.ATTENTION_BRANCHES))
         
        return Z, A

class CLAMAttention(nn.Module):
    def __init__(self, in_ch=1536):
        super().__init__()
        self.M = in_ch
        self.L = int(self.M/2)
        self.ATTENTION_BRANCHES = 1
        self.SIZE=512
        
        self.FC=nn.Sequential(
        nn.Linear(self.M, self.L),
        torch.nn.Dropout(0.25),
        )
        
        self.A=nn.Sequential(
        nn.Linear(self.L, self.SIZE),
        nn.Tanh(),
        torch.nn.Dropout(0.25),
        )
        self.B=nn.Sequential(
        nn.Linear(self.L, self.SIZE),
        nn.Sigmoid(),
        torch.nn.Dropout(0.25),
        )
        
        self.FCA=nn.Sequential(
          nn.Linear(self.SIZE, 1),
          )
          
        self.MFC=nn.Sequential(
          nn.Linear(self.L, 2),
          )
        
    def forward(self, x, n_padding=None, mx_patches=None):
        
        x = x.squeeze(0)
        H=self.FC(x)
        
        
        #Attention branch
        a=self.A(H)
        b=self.B(H)
        
        c=torch.mul(a,b)
        
        
        
        if n_padding!=None and mx_patches!=None:
          ### masking 
          list_of_masks=[]
          for value in n_padding:
            mask_zeroes=torch.zeros(mx_patches-int(value), 1, dtype=torch.bool)
            mask_ones=torch.ones(int(value), 1, dtype=torch.bool)
            
            mask=torch.cat((mask_zeroes, mask_ones))
            
            mask=mask.T
            list_of_masks.append(mask)
        
          mask=torch.stack(list_of_masks, dim=0)
          mask=mask.squeeze(1)
          
            
          #c[mask] = float('-inf')
          c[mask] = float(0)
        
        
        AA=self.FCA(c)
        
        AA=AA.squeeze()
        AA=F.softmax(AA, dim=1)
        
        AAT=AA.unsqueeze(dim=1)
        
        #AAT=AA.transpose(-2, -1)
        
        #Matrix multiplication
        
        M=torch.bmm(AAT, H)
        M=M.squeeze()
        
        Z=self.MFC(M)
        

        

        return Z, AA


       
class SelfAttention(nn.Module):
    def __init__(self, in_ch=1536):
        super().__init__()
        
        self.feature_size=in_ch
        
        self.key = nn.Linear(self.feature_size, self.feature_size)
        self.query = nn.Linear(self.feature_size, self.feature_size)
        self.value = nn.Linear(self.feature_size, self.feature_size)
        
    
    def forward(self, x):
        x=x.squeeze(0)
        
        keys=self.key(x) #k [b, n_patches, embedd_size]
        queries=self.query(x) #q [b, n_patches, embedd_size]
        values=self.value(x) #v [b, n_patches, embedd_size]
          
        qk = torch.matmul(queries, keys.transpose(-2, -1)) #qk [b, n_patches, n_patches]
        
        scale_dot_at=qk / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32)) #sda [b, n_patches, n_patches]
        
        
        att_w = F.softmax(scale_dot_at, dim=1) #aw [b, n_patches, n_patches]
        
        
        att_output = att_w.matmul(values) #ao [b, n_patches, embedd_size]
        
        att_output=torch.sum(att_output, dim=1) #ao [b, embedd_size]
        
        
        att_w = torch.mean(att_w, dim=2) #aw [b, n_patches]
        
        return att_output, att_w
    
class Clf_head(torch.nn.Module):
    def __init__(self, mx_patches, in_ch, out_ch=2, hid1=1024, hid2=512, hid3=128, dropout=0.3, attention_branches=6):
        super().__init__()
        
        #in_ch=1543
        self.mx_patches=mx_patches
        self.feature_size=in_ch
        #self.attention=True
        
        self.attention_branches=attention_branches
        self.self_attention=SelfAttention(in_ch=in_ch)
        self.attention=Attention(in_ch=in_ch, ATTENTION_BRANCHES=self.attention_branches)
        #self.attention=GatedAttention(in_ch=in_ch, ATTENTION_BRANCHES=self.attention_branches)
        #self.attention=CLAMAttention(in_ch=in_ch)
        
        ######
        
        self.lin = torch.nn.Sequential(
            torch.nn.Linear(in_ch*self.attention_branches, hid1*self.attention_branches),
            torch.nn.LayerNorm(hid1*self.attention_branches),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(dropout),
            
            torch.nn.Linear(hid1*self.attention_branches, hid2),
            torch.nn.LayerNorm(hid2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(dropout),
            
            torch.nn.Linear(hid2, hid3),
            torch.nn.LayerNorm(hid3),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(dropout),
            
            torch.nn.Linear(hid3, out_ch),
        )
        
    def forward(self, x, n_padding=None, histodata=None):
                
        att_output, att_w= self.attention(x, n_padding=n_padding, mx_patches=self.mx_patches) #[b, n_patches], [b, embedd_dim]
        #att_output, att_w= self.self_attention(x) #[b, n_patches], [b, embedd_dim]
        
        out=self.lin(att_output)
        
        return out, att_w      
        


  
