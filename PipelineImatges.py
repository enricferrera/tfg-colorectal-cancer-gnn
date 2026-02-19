
import os
from pathlib import Path
import numpy as np
from random import shuffle
import pickle
import random
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool,global_max_pool, global_add_pool
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models

from sklearn.model_selection import StratifiedGroupKFold,train_test_split
from torch.utils.data import DataLoader

from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

import matplotlib.pyplot as plt
import gc
from PIL import Image

import seaborn as sns
import pandas as pd
import csv
import os
import copy
import models_attention


##############################33
#################################

  
def train_loop(data_dict, idxs, model_classifier, dataset_classifier, loss_function_classifier, device, optimizer_classifier, model_extractor=None, dataset_extractor=None, loss_function_extractor=None, optimizer_extractor=None, epochs=30, minibatch_size=2, batch_size=12):
  
  if model_extractor!=None:
    td=dataset_extractor(data_dict, idxs)
    ppp=len(td)
    trainloader_extractor = DataLoader(td, batch_size=batch_size, shuffle=True, pin_memory=False)
    del td
    model_extractor.to(device)
    model_extractor.train()
    model_classifier.eval()
    loss_function_extractor.to(device)
    
    for epoch in range(epochs):
      
      batch_counter = 0
      
      last_dims, affectation_rec, hospitals_rec, patients_rec, slides_rec, coords_rec, = [], [], [], [], [], []
      
      for b in trainloader_extractor:
          x, y, extra_info = b
          
          #Extra info deglossed:
          hosps_batch = extra_info['hospital']
          pats_batch = extra_info['patient']
          slds_batch = extra_info['slides']
          coords_batch = extra_info['coords']
          affectation_batch = extra_info['coords']
          
          
          output, last_dim = model_extractor(x.to(device))
          
          loss_ex = loss_function_extractor(output.float(), y.long().to(device))
          #loss_ex = loss_ex / minibatch_size  # Scale loss
          loss_ex.backward()
          
          
          batch_counter +=1
          if batch_counter >= minibatch_size:
            optimizer_extractor.step()
            optimizer_extractor.zero_grad()
            batch_counter = 0
          
          #losses.append(loss.item())
          last_dim_cpu=last_dim.cpu().detach()
          
          last_dims.extend(last_dim_cpu)
          affectation_rec.extend(affectation_batch)
          hospitals_rec.extend(hosps_batch)
          patients_rec.extend(pats_batch)
          slides_rec.extend(slds_batch)
          coords_rec.extend(coords_batch)
          
    
    rebuild_params={
    'features': last_dims,
    'affectation': affectation_rec,
    'hospitals': hospitals_rec,
    'patients': patients_rec,
    'slides': slides_rec,
    'coords': coords_rec,
    }
    
    data_dict=patient_dict_builder(**rebuild_params)   
    
    model_extractor.eval()
  
  model_classifier.train()
  model_classifier.to(device)
  loss_function_classifier.to(device)
  td=dataset_classifier(data_dict, idxs)
  trainloader_classifier = DataLoader(td, batch_size=batch_size, shuffle=True, pin_memory=False)
  del td
  
  
  for epoch in range(epochs):
  
    batch_counter = 0
    for b in trainloader_classifier:
        x, y, extra_info, histodata = b
        
        n_padding_batch=extra_info['n_padding']
        output, _ = model_classifier(x.to(device), n_padding=n_padding_batch, histodata=None)
        
        loss_clf = loss_function_classifier(output, y.long().to(device))
        loss_clf = loss_clf / minibatch_size  # Scale loss
        loss_clf.backward()
        
        batch_counter +=1
        if batch_counter >= minibatch_size:
          optimizer_classifier.step()
          optimizer_classifier.zero_grad()
          batch_counter = 0
  
  if model_extractor!=None:
    trained_models=[model_extractor, model_classifier]
    
  else:
    trained_models=[model_classifier]
  return trained_models

def val_loop(data_dict, idxs, model_classifier, dataset_classifier, device, dataset_extractor=None, model_extractor=None, batch_size=12):
  
  if model_extractor!=None:
    vd=dataset_extractor(data_dict, idxs)
    valoader_extractor = DataLoader(vd, batch_size=batch_size, shuffle=True, pin_memory=False)
    del vd
    model_extractor.to(device)
    model_extractor.eval()
    
    
    last_dims, affectation_rec, hospitals_rec, patients_rec, slides_rec, coords_rec, n_padding_rec = [], [], [], [], [], [], []
    for b in valoader_extractor:
        x, y, extra_info = b
        
        #Extra info deglossed:
        hosps_batch = extra_info['hospital']
        pats_batch = extra_info['patient']
        slds_batch = extra_info['slides']
        coords_batch = extra_info['coords']
        affectation_batch = extra_info['coords']
        
        
        output, last_dim = model_extractor(x.to(device))
        
        #losses.append(loss.item())
        last_dim_cpu=last_dim.cpu().detach()
        last_dims.extend(last_dim_cpu)
        affectation_rec.extend(affectation_batch)
        hospitals_rec.extend(hosps_batch)
        patients_rec.extend(pats_batch)
        slides_rec.extend(slds_batch)
        coords_rec.extend(coords_batch)
        
    
    rebuild_params={
    'features': last_dims,
    'affectation': affectation_rec,
    'hospitals': hospitals_rec,
    'patients': patients_rec,
    'slides': slides_rec,
    'coords': coords_rec,
    }
  
    data_dict=patient_dict_builder(**rebuild_params)   
    
    
  
  
  model_classifier.to(device)
  model_classifier.eval()
  
  vd=dataset_classifier(data_dict, idxs)
  
  valoader_classifier = DataLoader(vd, batch_size=batch_size, shuffle=True, pin_memory=False)
  del vd
  
  
  attention_output={}
  
  y_true, y_pred, y_scores = [], [], []
  
  
  for b in valoader_classifier:
      x, y, extra_info, histodata = b
      
      #Extra info deglossed:
      hosps_batch = extra_info['hospital']
      pats_batch = extra_info['patient']
      slds_batch = extra_info['slides']
      coords_batch = extra_info['coords']
      affectation_batch = extra_info['coords']
      n_padding_batch = extra_info['n_padding']
      
      output, attention_scores = model_classifier(x.to(device), n_padding=n_padding_batch, histodata=None)
      
      probs = F.softmax(output, dim=1).cpu()
      
      y_true.extend(y.cpu().tolist())
      
      
      y_pred.extend(probs.argmax(dim=1).cpu().tolist())
      y_scores.extend(probs[:,1].cpu().tolist())     
      
      #storing attention
      for i, hospi in enumerate(hosps_batch):
                  
                  hosp=hosps_batch[i]
                  pat=pats_batch[i]
                  true=y[i].float().item()
                  label_p=true
                  st_probs=probs.cpu().tolist()[i]
                  pred=probs.argmax(dim=1).cpu().tolist()[i]
                  pred=float(pred)
                  n_pad=n_padding_batch[i]
                  n_pad=mx_patches-n_pad
                  
                  
                  slide_ds= slds_batch[i]
                  attention_matrixs=attention_scores[i].cpu()
                  coord_ds=coords_batch[i].detach().cpu()
                  x_ds=x[i].detach().cpu()
                  
                  attention_matrixs=attention_matrixs[:n_pad]
                  
                  coord_ds=coord_ds[:n_pad]
                  
                  slide_ds=slide_ds[:n_pad]
                  x_ds=x_ds[:n_pad]
                  
                  for s, slide_s in enumerate(slide_ds):
                      slide=slide_ds[s]
                      slide=slide.item()
                      slide=inv_slide_index_dict[slide]
                      coord_d=coord_ds[s].numpy()
                      features=x_ds[s].numpy()
                      
                      #attention_matrix=attention_matrixs[0][s].detach().item() #Zero [0] so it doesnt fuck up everything
                      
                      attention_matrix_detached=attention_matrixs.detach()
                      attention_matrix_heads={}
                      for a, att_head in enumerate(attention_matrix_detached):
                        attention_matrix_heads[a]=attention_matrix_detached[a][s]
                      
                      if hosp not in attention_output:
                          attention_output[hosp]={}
            
                      if pat not in attention_output[hosp]:
                          attention_output[hosp][pat]={}
                      
                      
                      if slide not in attention_output[hosp][pat]:
                          attention_output[hosp][pat][slide]=[]
                      
                      attention_output[hosp][pat]['scores']=st_probs
                      attention_output[hosp][pat]['label']=true
                      attention_output[hosp][pat]['predicted']=pred
                      attention_output[hosp][pat][slide].append((features, coord_d, attention_matrix_heads))
                
  return y_true, y_pred, y_scores, attention_output
  


############################3
#############################
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

#------FIXED SEEDS-------
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
# LOAD image and labels
fix_seeds(r_seed=123)

########################################## AUTOENCODER IMATGES ############
class AEDataset(torch.utils.data.Dataset):
    def __init__(self, pat_dict, idxs, ):
      
      #self.pat_dict=pat_dict.copy()
      temp_pat_dict={}
      #self.pat_dict = copy.deepcopy(pat_dict)
      for pat in list(pat_dict.keys()):
        if pat in idxs:
          temp_pat_dict[pat]=pat_dict[pat]
      
      self.pat_dict=temp_pat_dict
      self.patient_dict_list=list(self.pat_dict.keys())
      
      all_megapatches=[]
      for pat in self.patient_dict_list:
        all_megapatches.extend(self.pat_dict[pat]['megapatches'])
      

      self.all_megapatches=all_megapatches
      
    def __len__(self):
        return len(self.all_megapatches)
        
    def __getitem__(self, i):
      
      selc_megapatch=self.all_megapatches[i]
      
      image_paths=selc_megapatch['paths']
      #print(image_paths[0])
      ##############
      '''
      temps_paths=[]
      for i_path in image_paths:
        
        i_path=i_path.replace("\\","/")
        image_name=os.path.split(i_path)[-1]
        temps_paths.append(image_name)
        pass
      
      image_paths=temps_paths
      ##############
      '''
      patches = [transforms.ToTensor()(Image.open(p).convert("RGB"))
                   for p in image_paths]
        
      patches = torch.stack(patches)
      fm=jigsaw_to_image(patches)
      x=fm
      
      
    
      ### if training on affectation #######
      
      y=selc_megapatch['affectation_label']
      
      
      
      extra_info= {
      'hospital': selc_megapatch['hospital'],
      'patient': selc_megapatch['patient'],
      'slides': selc_megapatch['slide'],
      'coords': selc_megapatch['coords'],  
      }
      
            
      return x, y, extra_info#, histodata

###### DATASET FET AFECTACIO i SOBRE CLS
class AffectDataset(torch.utils.data.Dataset):
    def __init__(self, pat_dict, idxs, ):
      
      #self.pat_dict=pat_dict.copy()
      temp_pat_dict={}
      #self.pat_dict = copy.deepcopy(pat_dict)
      for pat in list(pat_dict.keys()):
        if pat in idxs:
          temp_pat_dict[pat]=pat_dict[pat]
      
      self.pat_dict=temp_pat_dict
      self.patient_dict_list=list(self.pat_dict.keys())
      
      all_megapatches=[]
      for pat in self.patient_dict_list:
        all_megapatches.extend(self.pat_dict[pat]['megapatches'])
      
      patch_selection=[]
      for megapatch in all_megapatches:
        affect_percent = megapatch['affectation']
        if affect_percent>=0.75:
          megapatch['affectation_label']=1
        elif affect_percent<=0.25:
          megapatch['affectation_label']=0
        else:
          continue
        patch_selection.append(megapatch)
      
      self.all_megapatches=patch_selection
      
    def __len__(self):
        return len(self.all_megapatches)
        
    def __getitem__(self, i):
      
      selc_megapatch=self.all_megapatches[i]
      
      
      x=selc_megapatch['features']
      
    
      ### if training on affectation #######
      
      y=selc_megapatch['affectation_label']
      
      
      
      extra_info= {
      'hospital': selc_megapatch['hospital'],
      'patient': selc_megapatch['patient'],
      'slides': selc_megapatch['slide'],
      'coords': selc_megapatch['coords'],  
      }
      
            
      return x, y, extra_info#, histodata

##### DATASET CLS
class AttnDataset(torch.utils.data.Dataset):
    def __init__(self, pat_dict, idxs, ):
      
      #self.pat_dict=pat_dict.copy()
      temp_pat_dict={}
      #self.pat_dict = copy.deepcopy(pat_dict)
      for pat in list(pat_dict.keys()):
        if pat in idxs:
          temp_pat_dict[pat]=pat_dict[pat]
      
      self.pat_dict=temp_pat_dict
      self.patient_dict_list=list(self.pat_dict.keys())
      
      
    def __len__(self):
        return len(self.pat_dict)
        
    def __getitem__(self, i):
      
      selc_patient=self.patient_dict_list[i]
      
      
      x=[]
      for megapatch in self.pat_dict[selc_patient]['megapatches']:
        x.append(megapatch['features'])
      
      x=torch.stack(x, dim=0)
      
      #self.max_l=800
      self.max_l=max_l  #maxim de mostres per un pacient
      
      n_padding=self.max_l-x.shape[0]  #completar els pacients amb menys mostres
      
      
      y=self.pat_dict[selc_patient]['label']
      histodata=self.pat_dict[selc_patient]['histodata']
      #### padding coords #####
      coords=[]
      for megapatch in self.pat_dict[selc_patient]['megapatches']:
        coords.append(megapatch['coords'])
        
      
      coords=np.stack(coords, axis=0)
      
      
      zero_pad=np.zeros((self.max_l-coords.shape[0], 2))
      
      coords=np.concatenate((coords,zero_pad), axis=0)
      
      #### padding slides #####
      slides_n=[slide_index_dict[self.pat_dict[selc_patient]['megapatches'][k]['slide']] for k in range(0, len(self.pat_dict[selc_patient]['megapatches']))]
      
      
      
      slides_padded =  slides_n + [000000] * (self.max_l-len(slides_n))
      slides_padded=np.array(slides_padded)
      
      extra_info= {
      'hospital': self.pat_dict[selc_patient]['megapatches'][0]['hospital'],
      'patient': self.pat_dict[selc_patient]['megapatches'][0]['patient'],
      'slides': slides_padded,
      'coords': coords,
      'n_padding': n_padding,
      
      }
      
      
      ## Padding of the tensor
      x=F.pad(x, (0, 0, 0, n_padding), mode='constant', value=0)
      
      '''
      #Boostrapping
      if n_padding>0:
        indiv_tensor_list=list(torch.tensor_split(x, x.shape[0]))
        
        bootstraped_tensor_list=[random.choice(indiv_tensor_list) for _ in range(n_padding)]
        
        bootstraped_tensor_list=torch.stack(bootstraped_tensor_list, dim=0)
        bootstraped_tensor_list=bootstraped_tensor_list.squeeze(1)
        
        x=torch.concat((x, bootstraped_tensor_list), dim=0) 
      
      '''     
      return x, y, extra_info, histodata
      

############# Data paths and indexes ###################

npz_path=r"full_patch_cls"
#npz_path=r"cls_all_nw100_front"
#npz_path=r"last_dim_pack"
#npz_path=r"cls_FRONT"
npz_path=r"cls_ALL"
npz_path=Path(npz_path)

affectation_flag=True#False#True

tensors_list=[]
y_list=[]
patients_list=[]
hosps_list=[]
slides_list=[]
coords_list=[]
paths_list=[]
for i, link in enumerate(npz_path.iterdir()):
  
  data = np.load(link, allow_pickle=True)
  
  features=data['embeddingCLS']
  features=torch.from_numpy(features)
  
  affectation=data['label_list']
  affectation=torch.from_numpy(affectation)
  
  patients=data['patient_list']
  
  hospitals=data['hospitals']
  slides=data['slides']
  coords=data['coords']
  paths=data['paths']
  
  tensors_list.append(features)
  y_list.append(affectation)
  
  patients_list.append(patients)
  hosps_list.append(hospitals)
  slides_list.append(slides)
  paths_list.append(paths)
  
  
  #mini_coords_list=[]
  for coord_pack in coords:
    coords_list.append(coord_pack[0])
    #mini_coords_list.append(coord_pack[0])
  #coords_list.append(mini_coords_list)
  '''
  print("------ Array info ----------")
  print(f"size of CLS: {len(features)}")
  
  print(f"size of affectations: {len(affectation)}")
  print(f"size of patients: {len(patients)}")
  print(f"size of hospitals: {len(hospitals)}")
  print(f"size of slides: {len(slides)}")
  print(f"size of coords: {len(coords)}")
  print(f"size of coords: {len(paths)}")
  '''
  
features=torch.cat(tensors_list, dim=0)
print("Features: ",features.shape)

affectation=torch.cat(y_list, dim=0)
print("Y: ",affectation.shape)

patients=np.concatenate(patients_list)

###
#PatID_no_hosp=np.concatenate(hosps_list)
###

print("Patients: ", np.unique(patients).shape[0])

hospitals=np.concatenate(hosps_list)
print("Hospitals: ",len(np.unique(hospitals)))

slides=np.concatenate(slides_list)
print("Slides: ",len(np.unique(slides)))

slides=np.concatenate(slides_list)
print("Patches: ",features.shape[0])


coords=np.stack(coords_list, axis=0)
print("Coords: ",len(coords))


paths=np.concatenate(paths_list)
print("Image Paths: ",len(paths))
print('------ filtering out bad patients...-------')


#####################################################

histopath=r"24_09_2025_pT1_CRC_CASOS_DEFINITIUS_AMB_ITEMS_HISTOLOGICS_fixed_N0s.xlsx"
#cheat_sheet_csv_path=r"/home/lquerol/Desktop/Projectes_Nil/PATCHES_images_server_testing/temp.csv"


histopath_df=pd.read_excel(histopath)

histopath_df=histopath_df.rename(columns={'Annotated slide ': "slides"})

histopath_df.slides=histopath_df.slides.str.replace(';',',')

histopath_df["PATHOLOGIST SCORE. Stage of CRC (N)TNM Colorectal Cancer 8th edition (NX/ N0/N1/N1a /N1b/N1c /N2 /N2a /N2b): N0=Negatiu; NX=Dubtos; Resta=Positiu"]=histopath_df["PATHOLOGIST SCORE. Stage of CRC (N)TNM Colorectal Cancer 8th edition (NX/ N0/N1/N1a /N1b/N1c /N2 /N2a /N2b): N0=Negatiu; NX=Dubtos; Resta=Positiu"].replace({"NX": -1,"N0": 0, "N1": 1, "N1a": 1, "N1b": 1, "N1c": 1, "N2a": 1, "N2b": 1}).astype(int)

histopath_df=histopath_df.rename(columns={"PATHOLOGIST SCORE. Stage of CRC (N)TNM Colorectal Cancer 8th edition (NX/ N0/N1/N1a /N1b/N1c /N2 /N2a /N2b): N0=Negatiu; NX=Dubtos; Resta=Positiu":"PATHOLOGIST SCORE"})

histopath_df = histopath_df.rename(columns={
"CODE":"Patient_CODE",
"Data Access Group": "hospital",
'Lymphovascular invasion':"LVI",
'Presence of Tumor Budding':"Budding",
'Degree of differentiation':"Degree",
'Vertical margin':"VRM",
'Horizontal margin':"HRM",
'Mucinous ADK':'Mucinous',
'Perineural invasion': 'PI',
'Depth of submucosal invasion (in mm)': 'Depth',

})


histopath_df['Budding']=histopath_df['Budding'].replace({"Bd0":0, "Bd1":0,"bd0": 0,  "Bd2":1, "Bd3":1, "Not applicable":2}).infer_objects()

histopath_df['LVI']=histopath_df['LVI'].replace({"No":0, "Yes":1, "Bd2":1, "Bd3":1, "Not applicable":2}).infer_objects()

histopath_df['Degree']=histopath_df['Degree'].replace({"Low grade (G1 and G2)":0, "High grade (G3 and G4)":1}).infer_objects()

histopath_df['VRM']=histopath_df['VRM'].replace({"Free of lesion (>1 mm)":0, "Free of lesion (0.1-1 mm)":0, "Afected by adenocarcinoma":1, "Indeterminate":2}).infer_objects()

histopath_df['HRM']=histopath_df['HRM'].replace({"Free of lesion":0, "Afected by adenoma or adenocarcinoma in situ (pTis)":1, "Afected by infiltrating adenocarcinoma":1, "Indeterminate":2}).infer_objects()

histopath_df['PI']=histopath_df['PI'].replace({"No":0, "Yes":1}).infer_objects()

histopath_df['Depth'].where(histopath_df['Depth'] <= 1, 0, inplace=True)
histopath_df['Depth'].where(histopath_df['Depth'] > 1, 1, inplace=True)

#histopath_df['Mucinous']=histopath_df['Mucinous'].replace({"No":0, "Yes":1})

histopath_df=histopath_df.replace({np.nan:2})


#Filtrem els pacients amb diagnostic NX
histopath_df = histopath_df[histopath_df["PATHOLOGIST SCORE"] != -1]

pat_histo=histopath_df["Patient_CODE"]

nx_label=histopath_df["PATHOLOGIST SCORE"]
histo_data=histopath_df[["Budding", "LVI", "Degree", "VRM", "PI", "Depth", "HRM"]]

#print("what: ", len(pat_histo))
#print("doublew: ", pd.Series(patients).value_counts())

## FILTREM PACIENTS NX Baixem de 401 a 370.
pat_nx_dict = {str(key): int(value) for key, value in zip(pat_histo, nx_label)}
pat_histodata_dict = {str(key): value for key, value in zip(pat_histo, histo_data.values)}

print("//////////////////////////////////////////////////////////")
print(pd.Series(nx_label).value_counts())
#################

#Funció per muntar la imatge del megapatch donades les imatges dels patches
def jigsaw_to_image(x, grid_size=None, patch_size=None):
    
    n_patches, channels, h, w =x.shape # [N, c, 256, 256]
    
    if grid_size==None:
        grid_n_side=int(n_patches**0.5)
    if patch_size==None:
        patch_size=256
    
    #x=x.permute(0, 3, 1, 2) # [9, c, 256, 256]
    x=x.view(grid_n_side, grid_n_side, channels, h, w) # [n, n, c, 256, 256]
    x = x.permute(2, 1, 3, 0, 4).contiguous()  # shape: [n, n, w, c, h]
    final_image = x.view(channels, grid_n_side*patch_size, grid_n_side*patch_size)
    final_image = transforms.ToPILImage()(final_image)
    # img.show(final_image)
    return  final_image


def patient_dict_builder(features, affectation, hospitals, patients, slides, coords):
  patients_not_found=set()
  patient_dict={}
  for i, datapoint in enumerate(patients):
      hosp=hospitals[i]
      pat=patients[i]
      
      if pat in pat_nx_dict:
        
        label= int(pat_nx_dict[pat])
        histodata=np.array(pat_histodata_dict[pat]).astype(float)
        
        affect_percent=affectation[i]
        afect_label=affect_percent
        
        
        ######################################
        
        megapatch_dict={
        'hospital': hosp,
        'patient': pat,
        'affectation': afect_label,
        'features': features[i],
        'coords': coords[i],
        'slide': slides[i],
        'paths': paths[i]
        
        }
        '''
        if hosp not in patient_dict:
            patient_dict[hosp]={}
        
        if pat not in patient_dict[hosp]:
            patient_dict[hosp][pat]={
        '''
        if pat not in patient_dict:
            patient_dict[pat]={
            'label': label,
            'megapatches': [],
            'histodata': torch.from_numpy(histodata),
            }
            
        patient_dict[pat]['megapatches'].append(megapatch_dict)
      else:
        patients_not_found.add(pat)
  #print(len(patient_dict), "patients")   
  #print(len(patients_not_found), "not in excel!")
  
  return patient_dict

## Intersecció entre imatges processades i metadades associades 370 a 230 pacients.
patient_dict=patient_dict_builder(features, affectation, hospitals, patients, slides, coords)   
max_l=0
for pat in patient_dict:
  patient_patch_size=len(patient_dict[pat]['megapatches'])
  if patient_patch_size > max_l:
    max_l=patient_patch_size
    
print(max_l) # mostres del pacient amb mes mostres
############# Sampling only a percent of each patients patches ##############
############# and creating four dictionaries (then storing them) ##############



### MAP EACH SLIDE TO A NUMBER SO THAT THE DATALOADER CAN TAKE THEM

unique_slide_list = list(set(slides))
slide_index_dict={word: index for index, word in enumerate(unique_slide_list)}
inv_slide_index_dict = {v: k for k, v in slide_index_dict.items()}


#### do it for each bootstraped dictionary

fix_seeds(r_seed=123)

#megapatches_to_use=boostrapped_megapatch_dicts[i]
bt_dict=patient_dict
patient_list=list(bt_dict.keys())
label_list=[]
for pat in bt_dict:
  label_list.append(bt_dict[pat]['label'])  

# ================= STEP 4: Training & Validation =================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= weights =================
def calculate_class_weights(y):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    print("unique classes: ", unique_classes)
    total_samples = len(y)
    class_weights = []
    
    for class_label, class_count in zip(unique_classes, class_counts):
        class_weight = total_samples / (2.0 * class_count)
        class_weights.append(class_weight)
    f_weights=[]
    tot=np.sum(class_weights)
    for weight in class_weights:
        weight=weight/tot
        f_weights.append(weight)
    return f_weights


weight_0, weight_1 = calculate_class_weights(label_list)
print("weights: ", weight_0, weight_1)


weights=torch.tensor([float(weight_0), float(weight_1)])

#loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

########## Parameters #################

n_folds=10
minibatch_size = 2

epochs = 30#10 #50


batch_size = 24
#batch_size = 18 #if 24, because there are 23 patients, the last one gave an error because last input was [1] instead of [N, 1]

##########  ---------  ###############
from sklearn.model_selection import StratifiedKFold


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
    
    loss_fn_ex = torch.nn.CrossEntropyLoss(weight=weights)
    loss_fn_clf = torch.nn.CrossEntropyLoss(weight=weights)
    affectation_extractor = models_attention.AffectationExtractor(n_features, out_ch=2, hid1=1024, hid2=512, hid3=128, dropout=0.3)
    #affectation_extractor = models_attention.AE()
    
    opt_extractor = torch.optim.Adam(affectation_extractor.parameters(), lr=1e-4)
    
    n_features=128
    hid1=64
    hid2=int(hid1/2)
    
    hid3=int(hid2/2)
    
    n_features=1536
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
    'model_extractor': None,#affectation_extractor,
    'model_classifier': model,
    'dataset_extractor': AffectDataset,#AEDataset
    'dataset_classifier': AttnDataset, #AttnDataset_SLIDE
    'device': device,
    'loss_function_classifier': loss_fn_clf,
    'loss_function_extractor': loss_fn_ex,
    'optimizer_classifier': opt,
    'optimizer_extractor': opt_extractor,
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
    'model_extractor': None,#affectation_extractor,
    'model_classifier': model,
    'dataset_extractor': AffectDataset,
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
    
    ########updating complete dictionary
    for hospital in partial_att_dict:
    
      if hospital not in attention_output:
        attention_output[hospital]=partial_att_dict[hospital]
      
      else:
        for patient in partial_att_dict[hospital]:
          if patient not in attention_output[hospital]:
            attention_output[hospital][patient]=partial_att_dict[hospital][patient]
          else:
            for slide in partial_att_dict[hospital][patient]:
              if slide not in attention_output[hospital][patient]:
                attention_output[hospital][patient][slide]=partial_att_dict[hospital][patient][slide]
              else:
                attention_output[hospital][patient][slide].extend(partial_att_dict[hospital][patient][slide])
    
    
    attention_output.update(partial_att_dict)
           
with open(f'attention_dict.pkl', 'wb') as f:
  pickle.dump(attention_output, f)
  
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