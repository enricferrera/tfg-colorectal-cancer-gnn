# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 14:01:32 2021

@author: debora
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import _reduction as _Reduction

from torch import Tensor
from typing import Callable, Optional

import numpy as np
from sklearn.preprocessing import label_binarize
import math

def class_sample_count_elias(labels):
    tags = set(labels) # unique categories
    my_dic = {i:labels.count(i) for i in tags}
    my_dic = dict(sorted(my_dic.items())) # weight should be ordered for the optimizer
    print(my_dic)
    samples = list(my_dic.values())
    return samples

def class_sample_count(labels):
    u,indices=np.unique(np.sort(labels.flatten()),return_index=True)
    nword=np.diff(np.array(list(indices)+[len(labels.flatten())]))
    return nword

def WeightedCrossEntropy(y_train):
    
    # computing a weight per class/sample only is util when
    # you are dealing with unbalanced data, however, it does
    # not matter with balanced dataset
    sample_counts = np.array(class_sample_count((y_train)))
    classes_weight=1./sample_counts
    classes_weight=classes_weight/np.sum(classes_weight)
    classes_weight=torch.tensor(classes_weight, dtype=torch.float).cuda()
    criterion = nn.CrossEntropyLoss(weight=classes_weight)
        
    return criterion,classes_weight

def classes_weight(y_train):
    
    
    # computing a weight per class/sample only is util when
    # you are dealing with unbalanced data, however, it does
    # not matter with balanced dataset

    sample_counts = np.array(class_sample_count(y_train))
    #sample_counts = np.array(class_sample_count(y_train))
    classes_weight=1./sample_counts
    classes_weight=classes_weight/np.sum(classes_weight)
    classes_weight=torch.tensor(classes_weight, dtype=torch.float).cuda()
    
        
    return classes_weight


def classes_weight_binary(y_train):
    
    
    # computing a weight per class/sample only is util when
    # you are dealing with unbalanced data, however, it does
    # not matter with balanced dataset

    sample_counts = np.array(np.sum(y_train,axis=0))
    # avoid division by zero, and therefore, inf values in the result
 #   sample_counts[sample_counts==0] = 1
    classes_weight=1./sample_counts
    classes_weight[classes_weight==np.inf]=0
    classes_weight=classes_weight/np.sum(classes_weight)
    classes_weight=torch.tensor(classes_weight, dtype=torch.float).cuda()
    
        
    return classes_weight



        
class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction
            
class _WeightedLoss(_Loss):
    def __init__(self, weight= None, 
                 size_average=None, reduce=None, reduction: str = 'mean') -> None:
      #  super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        super().__init__(size_average, reduce, reduction)
      #  self.register_buffer('weight', weight)
        self.weight=weight


class VAELoss(Module):
    def __init__(self,ReconLoss='L1'):
        super().__init__()
        self.ReconLoss=ReconLoss
        self.name='VAE'
        
    def forward(self,ori,recon,mu,logvar):
       # KL loss of Latent Space
       kl_loss = -torch.mean(1 + logvar - mu**2 - torch.exp(logvar))

       # Similarity loss
       if self.ReconLoss=='KLD':
           crit=nn.KLDivLoss(reduction='batchmean')
           recon_loss=crit(F.log_softmax(recon,dim=1),ori)
          
       elif self.ReconLoss=='L1':
           crit=nn.L1Loss(reduction='mean')
           recon_loss =  crit(recon, ori)
       elif self.ReconLoss=='L2':
           crit=nn.L2Loss(reduction='mean')
           recon_loss =  crit(recon, ori)
       
       vae_loss = torch.mean(kl_loss+recon_loss)
       
       return vae_loss,kl_loss,recon_loss
   
class VAELossCapacity(Module):
    def __init__(self,ReconLoss='L1',Capacity=25,C_warmup=100):
        super().__init__()
        self.ReconLoss=ReconLoss
        self.name='VAE'
        self.c_final=Capacity
        self.C_warmup=C_warmup
        
    def capacity_schedule(self,epoch, mode="linear"):
        """Return C(t) (in nats) that grows from 0 -> c_final over `warmup` steps."""
        if mode == "linear":
            t = min(1.0, epoch / max(1, self.C_warmup))
            return t * self.c_final
        if mode == "cosine":
            t = min(1.0, epoch / max(1, self.C_warmup))
            return (0.5 - 0.5*math.cos(math.pi * t)) * self.c_final
        # fallback
        return min(1.0, epoch / max(1, self.C_warmup)) * self.c_final
    
    def forward(self,ori,recon,mu,logvar,epoch):
       # KL loss of Latent Space
       
       C = self.capacity_schedule(epoch, mode="linear")
       kl_samp=torch.sum((1 + logvar - mu**2 - torch.exp(logvar)).abs(),dim=1)
       kl_loss = -torch.mean(torch.abs(kl_samp - C))
       # Similarity loss
       if self.ReconLoss=='KLD':
           crit=nn.KLDivLoss(reduction='batchmean')
           recon_loss=crit(F.log_softmax(recon,dim=1),ori)
          
       elif self.ReconLoss=='L1':
           crit=nn.L1Loss(reduction='mean')
           recon_loss =  crit(recon, ori)
           
       vae_loss = torch.mean(kl_loss+recon_loss)
       
       return vae_loss,kl_loss,recon_loss


# choose one penalty:
def loss_capacity_abs(recon, kl, step, warmup=50_000, c_final=25.0, beta=50.0):
    C = capacity_schedule(step, warmup, c_final, mode="cosine")
    return recon + beta * (kl - C).abs()

def loss_capacity_hinge(recon, kl, step, warmup=50_000, c_final=25.0, beta=50.0):
    C = capacity_schedule(step, warmup, c_final, mode="cosine")
    return recon + beta * torch.clamp(kl - C, min=0.0)

def loss_capacity_quad(recon, kl, step, warmup=50_000, c_final=25.0, beta=10.0):
    C = capacity_schedule(step, warmup, c_final, mode="cosine")
    return recon + beta * (kl - C)**2


# class Unetloss(Module): 
    
#     def __init__(self):
#         super().__init__()

        
#   #  criterion=WeightedCrossEntropy(true_masks.cpu().detach().numpy())[0]
#     def forward(self,masks_pred,true_masks):
#         criterion=nn.CrossEntropyLoss()
#         loss = criterion(masks_pred, true_masks)
        
#        # loss=0*loss
#         # loss += dice_loss(
#         # F.softmax(masks_pred, dim=1).float(),
#         # F.one_hot(true_masks, masks_pred.shape[1]).permute(0, 3, 1, 2).float(),
#         # multiclass=True
#         # )
        
#       #  pred=torch.argmax(masks_pred, dim=1)
#         loss += dice_loss(
#         F.one_hot(torch.argmax(masks_pred, dim=1), masks_pred.shape[1]).permute(0, 3, 1, 2).float(),
#         F.one_hot(true_masks, masks_pred.shape[1]).permute(0, 3, 1, 2).float(),
#         multiclass=True
#         )
        
#         return loss

class BarlowTwins(Module):
    def __init__(self,lambd=0.0051):
        super().__init__()
        self.lambd=lambd      
    def off_diagonal(self,x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self,c: Tensor, target: Tensor)-> Tensor:
 #       c=input[0]
 
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss
    
class TripletMarginLoss(_Loss):
    r"""Creates a criterion that measures the triplet loss given an input
    tensors :math:`x1`, :math:`x2`, :math:`x3` and a margin with a value greater than :math:`0`.
    This is used for measuring a relative similarity between samples. A triplet
    is composed by `a`, `p` and `n` (i.e., `anchor`, `positive examples` and `negative
    examples` respectively). The shapes of all input tensors should be
    :math:`(N, D)`.

    The distance swap is described in detail in the paper `Learning shallow
    convolutional feature descriptors with triplet losses`_ by
    V. Balntas, E. Riba et al.

    The loss function for each sample in the mini-batch is:

    .. math::
        L(a, p, n) = \max \{d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\}


    where

    .. math::
        d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p

    See also :class:`~torch.nn.TripletMarginWithDistanceLoss`, which computes the
    triplet margin loss for input tensors using a custom distance function.

    Args:
        margin (float, optional): Default: :math:`1`.
        p (int, optional): The norm degree for pairwise distance. Default: :math:`2`.
        swap (bool, optional): The distance swap is described in detail in the paper
            `Learning shallow convolutional feature descriptors with triplet losses` by
            V. Balntas, E. Riba et al. Default: ``False``.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, D)` or :math:`(D)` where :math:`D` is the vector dimension.
        - Output: A Tensor of shape :math:`(N)` if :attr:`reduction` is ``'none'`` and
          input shape is :math:`(N, D)`; a scalar otherwise.

    Examples::

    >>> triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    >>> anchor = torch.randn(100, 128, requires_grad=True)
    >>> positive = torch.randn(100, 128, requires_grad=True)
    >>> negative = torch.randn(100, 128, requires_grad=True)
    >>> output = triplet_loss(anchor, positive, negative)
    >>> output.backward()

    .. _Learning shallow convolutional feature descriptors with triplet losses:
        http://www.bmva.org/bmvc/2016/papers/paper119/index.html
    """
    __constants__ = ['margin', 'p', 'eps', 'swap', 'reduction']
    margin: float
    p: float
    eps: float
    swap: bool

    def __init__(self, margin: float = 1.0, p: float = 2., 
                 eps: float = 1e-6, swap: bool = False, size_average=None,
                 reduce=None, reduction: str = 'mean',lamb=0.1):
        super().__init__(size_average, reduce, reduction)
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap
        self.lamb=lamb
       

    def forward(self, input: Tensor,target: Tensor) -> Tensor:
        
      #  lamb=0.1
        anchor=input[0]
        positive=input[1]
        negative=input[2]
        
        classes=torch.unique(target)
        Nclass=len(torch.unique(target))
        std_min=torch.zeros([Nclass], dtype=torch.float).cuda()
        
        for k in np.arange(Nclass):
         
            
            stdanchor=torch.mean(torch.std(anchor[target==classes[k],:],dim=0))
            stdpositive=torch.mean(torch.std(positive[target==classes[k],:],dim=0))
            stdnegative=torch.mean(torch.std(negative[target==classes[k],:],dim=0))
            std_min[k]=torch.min(torch.min(stdanchor,stdpositive),stdnegative)
        
        std_min=torch.min(std_min)       
        
        triplet_loss=F.triplet_margin_loss(anchor, positive, negative, margin=self.margin, p=self.p,
                                     eps=self.eps, swap=self.swap, reduction=self.reduction)
        
     #   print(lamb,std_min,triplet_loss)
        return triplet_loss + self.lamb*1/(std_min+1)
   # (1/((stdpositive+stdanchor)*0.5+1) + 1/((stdnegative)+1))
    
class MultiLabLoss(_WeightedLoss):
   
    __constants__ = ['reduction']

    def __init__(self, weight= None, size_average=None, 
                 reduce=None, reduction: str = 'mean', CatSplit=None) -> None:
      #  super(BCEMSELoss, self).__init__(weight, size_average, reduce, reduction)
        super().__init__(weight, size_average, reduce, reduction)
       
        self.CatSplit=CatSplit     
   
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        
        if self.CatSplit is not None:
            loss=torch.zeros(1).to(self.weight[0].device)
            NCat=len(self.CatSplit)-1
            Catloss=[]
            for k in np.arange(NCat):
                inputCat=input[:,self.CatSplit[k]:self.CatSplit[k+1]]
                targetCat=target[:,self.CatSplit[k]:self.CatSplit[k+1]]
#                Catloss=F.binary_cross_entropy(inputCat, targetCat, 
#                                         weight=self.weight[k], 
#                                         reduction=self.reduction)
                Catloss=torch.nn.BCELoss(inputCat, targetCat, 
                                         weight=self.weight[k])
                
#                Catloss=F.cross_entropy(torch.argmax(inputCat,dim=1), 
#                                        torch.argmax(targetCat,dim=1), 
#                         weight=self.weight[k])
                loss+=Catloss/NCat
            
        else:
        # Evaluate each loss
            loss=F.binary_cross_entropy(input, target, 
                                         weight=self.weight, reduction=self.reduction)
            Catloss=[loss]
            
        
        
        return loss,Catloss
    
class BCEMSELoss(_WeightedLoss):
   
    __constants__ = ['reduction']

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, 
                 reduce=None, reduction: str = 'mean',
                 alfa:Optional[float]=0.5) -> None:
      #  super(BCEMSELoss, self).__init__(weight, size_average, reduce, reduction)
        super().__init__(weight, size_average, reduce, reduction)
       
        self.alfa=alfa
        
   
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        
        
        # Split output into categorical (target1) and quantitative (target2) variables
        target1bin=target[0]
        target2=target[1]
       
        # Evaluate each loss
        loss1=F.binary_cross_entropy(input[0], target1bin, 
                                     weight=self.weight, reduction=self.reduction)
        loss2=F.mse_loss(input[1], target2, reduction=self.reduction)
        
        loss=self.alfa*loss1+(1-self.alfa)*loss2
        
        return loss,self.alfa*loss1,(1-self.alfa)*loss2
        #return loss,loss1,loss2
       
class BCEMSELossV0(_WeightedLoss):
   
    __constants__ = ['reduction']

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, 
                 reduce=None, reduction: str = 'mean',outcome_split: Optional[list]=[],
                 alfa:Optional[float]=0.5) -> None:
      #  super(BCEMSELoss, self).__init__(weight, size_average, reduce, reduction)
        super().__init__(weight, size_average, reduce, reduction)
        self.outcome_split1=np.array(outcome_split)
        self.alfa=alfa
        
    def target_split(self,target: Tensor):
        outcome_split2=np.setdiff1d(np.arange(target.shape[1]),self.outcome_split1)
        target1=target[:,self.outcome_split1].round().long()
        target2=target[:,outcome_split2]
        
        return target1,target2
    
    def targetbin(self,target1,ntokenout):
          
        target1bin=np.zeros((target1.size(0),ntokenout)).astype(int)
        
        #for k in np.arange(data_y_mxv.shape[1]):
        for k in np.arange(target1.size(1)):
            target1bin=target1bin+label_binarize(target1[:,k].cpu().detach().numpy(),
                                                 classes=np.arange(ntokenout))
        
        target1bin = torch.tensor(target1bin).to(target1.device)
        target1bin = target1bin.float()
        
        return target1bin
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        
        
        # Split output into categorical (target1) and quantitative (target2) variables
        target1,target2=self.target_split(target)
        # Binarize categorical target1
       
        target1bin=self.targetbin(target1,input[0].size(1))
        
       
        loss1=F.binary_cross_entropy(input[0], target1bin, 
                                     weight=self.weight, reduction=self.reduction)
        loss2=F.mse_loss(input[1], target2, reduction=self.reduction)
        
        loss=self.alfa*loss1+(1-self.alfa)*loss2
        
        return loss,self.alfa*loss1,(1-self.alfa)*loss2

