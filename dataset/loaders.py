import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image


# ================= weights =================
def calculate_class_weights(y):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    print("unique classes: ", unique_classes)
    total_samples = len(y)
    class_weights = []

    for class_label, class_count in zip(unique_classes, class_counts):
        class_weight = total_samples / (2.0 * class_count)
        class_weights.append(class_weight)
    f_weights = []
    tot = np.sum(class_weights)
    for weight in class_weights:
        weight = weight / tot
        f_weights.append(weight)
    return f_weights

# Funció per muntar la imatge del megapatch donades les imatges dels patches
def jigsaw_to_image(x, grid_size=None, patch_size=None):
    n_patches, channels, h, w = x.shape  # [N, c, 256, 256]

    if grid_size == None:
        grid_n_side = int(n_patches ** 0.5)
    if patch_size == None:
        patch_size = 256

    # x=x.permute(0, 3, 1, 2) # [9, c, 256, 256]
    x = x.view(grid_n_side, grid_n_side, channels, h, w)  # [n, n, c, 256, 256]
    x = x.permute(2, 1, 3, 0, 4).contiguous()  # shape: [n, n, w, c, h]
    final_image = x.view(channels, grid_n_side * patch_size, grid_n_side * patch_size)
    final_image = transforms.ToPILImage()(final_image)
    # img.show(final_image)
    return final_image


def patient_dict_builder(features, affectation, hospitals, patients, slides, coords, paths, pat_nx_dict, pat_histodata_dict):
    patients_not_found = set()
    patient_dict = {}
    for i, datapoint in enumerate(patients):
        hosp = hospitals[i]
        pat = patients[i]

        if pat in pat_nx_dict:

            label = int(pat_nx_dict[pat])
            histodata = np.array(pat_histodata_dict[pat]).astype(float)

            affect_percent = affectation[i]
            afect_label = affect_percent

            ######################################

            megapatch_dict = {
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
                patient_dict[pat] = {
                    'label': label,
                    'megapatches': [],
                    'histodata': torch.from_numpy(histodata),
                }

            patient_dict[pat]['megapatches'].append(megapatch_dict)
        else:
            patients_not_found.add(pat)
    # print(len(patient_dict), "patients")
    # print(len(patients_not_found), "not in excel!")

    return patient_dict


########################################## AUTOENCODER IMATGES ############
class AEDataset(torch.utils.data.Dataset):
    def __init__(self, pat_dict, idxs, ):

        # self.pat_dict=pat_dict.copy()
        temp_pat_dict ={}
        # self.pat_dict = copy.deepcopy(pat_dict)
        for pat in list(pat_dict.keys()):
            if pat in idxs:
                temp_pat_dict[pat] =pat_dict[pat]

        self.pat_dict =temp_pat_dict
        self.patient_dict_list =list(self.pat_dict.keys())

        all_megapatches =[]
        for pat in self.patient_dict_list:
            all_megapatches.extend(self.pat_dict[pat]['megapatches'])


        self.all_megapatches =all_megapatches

    def __len__(self):
        return len(self.all_megapatches)

    def __getitem__(self, i):

        selc_megapatch =self.all_megapatches[i]

        image_paths =selc_megapatch['paths']
        # print(image_paths[0])
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
        fm =jigsaw_to_image(patches)
        x=fm

        ### if training on affectation #######

        y=selc_megapatch['affectation_label']

        extra_info= {
            'hospital': selc_megapatch['hospital'],
            'patient': selc_megapatch['patient'],
            'slides': selc_megapatch['slide'],
            'coords': selc_megapatch['coords'],
        }


        return x, y, extra_info# , histoda

###### DATASET FET AFECTACIO i SOBRE CLS
class AffectDataset(torch.utils.data.Dataset):
    def __init__(self, pat_dict, idxs, ):

        # self.pat_dict=pat_dict.copy()
        temp_pat_dict={}
        # self.pat_dict = copy.deepcopy(pat_dict)
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
                megapatch['affectation_labe l ']=1
            elif affect_percent<=0.25:
                megapatch['affectation_labe l ']=0
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


        return x, y, extra_info# , histoda

##### DATASET CLS
class AttnDataset(torch.utils.data.Dataset):
    def __init__(self, pat_dict, idxs, ):

        # self.pat_dict=pat_dict.copy()
        temp_pat_dict={}
        # self.pat_dict = copy.deepcopy(pat_dict)
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

        # self.max_l=800
        self.max_l=max_l  # maxim de mostres per un pacient

        n_padding=self.max_l-x.shape[0]  # completar els pacients amb menys mostres


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
        slides_n=[slide_index_dict[self.pat_dict[selc_patient]['megapatches'][k]['slide']] for k
                    in range(0, len(self.pat_dict[selc_patient]['megapatches']))]



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