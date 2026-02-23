# -*- coding: utf-8 -*-
"""
Created on 15/02/2026

@author: Nil Arenos i Carles Sanchez
"""
### Pipeline for Image processing using Autoencoders
###################################################################


import pickle
import random
from pathlib import Path
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from models import models_attention
from dataset.loaders import *
from train.train import *
from train.val import *

#################################
#################################

#Directori del servidor on es troben les imatges
imatges_of_2048_path = r"Z:\Database\MedicalImaging\HistoPatologia\ColonCancer\PrivateBD\PEARSON\Images\Patches_2048"
imatges_of_2048_path = Path(imatges_of_2048_path)

hospitals = []
patients = []
slides = []
paths = []
coords = []
sections = []
other_info = []
affectation = []
for hospital_path in imatges_of_2048_path.iterdir():
    if Path.is_dir(hospital_path):
        hospital = hospital_path.name

        metadata_center_p = f"Z:\Database\MedicalImaging\HistoPatologia\ColonCancer\PrivateBD\PEARSON\Images/Patches_2048/{hospital}/metadata_{hospital}.csv"

        metadata_center = pd.read_csv(metadata_center_p, dtype=str)
        # metadata_center = metadata_center.rename(columns={'patient_ID': 'slide'})

        metadata_center['blurriness'] = pd.to_numeric(metadata_center['blurriness'], errors='coerce')
        metadata_center['non_white_area'] = pd.to_numeric(metadata_center['non_white_area'], errors='coerce')
        metadata_center['window_min_value'] = pd.to_numeric(metadata_center['window_min_value'], errors='coerce')
        metadata_center['affected_percentage'] = pd.to_numeric(metadata_center['affected_percentage'], errors='coerce')
        metadata_center['i'] = pd.to_numeric(metadata_center['i'], errors='coerce')
        metadata_center['j'] = pd.to_numeric(metadata_center['j'], errors='coerce')
        metadata_center.set_index(['hospital', 'patient_ID', 'slide_ID', 'i', 'j'], inplace=True)

        for patient_path in hospital_path.iterdir():
            if Path.is_dir(patient_path):
                patient = patient_path.name

                for slide_path in patient_path.iterdir():
                    if Path.is_dir(slide_path):
                        slide = slide_path.name
                        for image in slide_path.iterdir():
                            if image.name.endswith(".jpg"):
                                ## hospital, patient, slide, x, y = image.name.split('_') ##
                                _, _, _, x, y = image.stem.split('_')
                                hospitals.append(hospital)
                                patients.append(patient)
                                paths.append(image)
                                coords.append((x, y))
                                slides.append(slide)

                                blurriness = metadata_center.loc[
                                    (hospital, patient, slide, int(y), int(x)), 'blurriness']

                                afect = metadata_center.loc[
                                    (hospital, patient, slide, int(y), int(x)), 'affected_percentage']

                                affectation.append(afect)

                                section = metadata_center.loc[(hospital, patient, slide, int(y), int(x)), 'section_ID']
                                sections.append(section)

                                other_info.append(blurriness)
  




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
# LOAD image and labels
fix_seeds(r_seed=123)


print("Patients: ", np.unique(patients).shape[0])
print("Hospitals: ",len(np.unique(hospitals)))
print("Slides: ",len(np.unique(slides)))
print("Coords: ",len(coords))
print("Image Paths: ",len(paths))
print('------ filtering out bad patients...-------')


########## Llegim les metadates de cada pacient #################################
#################################################################################

histopath=r"24_09_2025_pT1_CRC_CASOS_DEFINITIUS_AMB_ITEMS_HISTOLOGICS_fixed_N0s.xlsx"

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



## Intersecció entre imatges processades i metadades associades 370 a 230 pacients.
## Els features serien les imatges, per això hi guardem els paths.
patient_dict=patient_dict_builder(paths, affectation, hospitals, patients, slides, coords, paths, pat_nx_dict, pat_histodata_dict)
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


fix_seeds(r_seed=123)

#megapatches_to_use=boostrapped_megapatch_dicts[i]
bt_dict=patient_dict
patient_list=list(bt_dict.keys())
label_list=[]
for pat in bt_dict:
  label_list.append(bt_dict[pat]['label'])  

# ================= STEP 4: Training & Validation =================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    fold_losses = []
    train_params = {
        'data_dict': bt_dict,
        'idxs': patient_list_train,
        'model_extractor': None,  # affectation_extractor,
        'model_classifier': model,
        'dataset_extractor': AffectDataset,  # AEDataset
        'dataset_classifier': AttnDataset,  # AttnDataset_SLIDE
        'max_l': max_l,
        'slide_index_dict': slide_index_dict,
        'device': device,
        'loss_function_classifier': loss_fn_clf,
        'loss_function_extractor': loss_fn_ex,
        'optimizer_classifier': opt,
        'optimizer_extractor': opt_extractor,
        'epochs': epochs,
        'minibatch_size': minibatch_size,
        'batch_size': batch_size
    }

    # affectation_extractor, model = train_loop(**train_params)
    trained_models = train_loop(**train_params)
    if len(trained_models) == 2:
        affectation_extractor, model = trained_models
    else:
        model = trained_models[0]

    ##### VALIDATION ##########

    val_params = {
        'data_dict': bt_dict,
        'idxs': patient_list_val,
        'model_extractor': None,  # affectation_extractor,
        'model_classifier': model,
        'dataset_extractor': AffectDataset,
        'dataset_classifier': AttnDataset,  # AttnDataset_SLIDE, #AttnDataset,
        'device': device,
        'batch_size': batch_size,
        'max_l': max_l,
        'inv_slide_index_dict': inv_slide_index_dict
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