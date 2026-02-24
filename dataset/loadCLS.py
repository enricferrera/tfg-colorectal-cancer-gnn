import numpy as np
import torch
from pathlib import Path
import pandas as pd


def loadCLSMetadata(npz_path):
    npz_path = Path(npz_path)

    affectation_flag = True  # False#True

    tensors_list = []
    y_list = []
    patients_list = []
    hosps_list = []
    slides_list = []
    coords_list = []
    paths_list = []
    for i, link in enumerate(npz_path.iterdir()):

        data = np.load(link, allow_pickle=True)

        features = data['embeddingCLS']
        features = torch.from_numpy(features)

        affectation = data['label_list']
        affectation = torch.from_numpy(affectation)

        patients = data['patient_list']

        hospitals = data['hospitals']
        slides = data['slides']
        coords = data['coords']
        paths = data['paths']

        tensors_list.append(features)
        y_list.append(affectation)

        patients_list.append(patients)
        hosps_list.append(hospitals)
        slides_list.append(slides)
        paths_list.append(paths)

        # mini_coords_list=[]
        for coord_pack in coords:
            coords_list.append(coord_pack[0])
            # mini_coords_list.append(coord_pack[0])
        # coords_list.append(mini_coords_list)
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

    features = torch.cat(tensors_list, dim=0)
    print("Features: ", features.shape)

    affectation = torch.cat(y_list, dim=0)
    print("Y: ", affectation.shape)

    patients = np.concatenate(patients_list)

    ###
    # PatID_no_hosp=np.concatenate(hosps_list)
    ###

    print("Patients: ", np.unique(patients).shape[0])

    hospitals = np.concatenate(hosps_list)
    print("Hospitals: ", len(np.unique(hospitals)))

    slides = np.concatenate(slides_list)
    print("Slides: ", len(np.unique(slides)))

    slides = np.concatenate(slides_list)
    print("Patches: ", features.shape[0])

    coords = np.stack(coords_list, axis=0)
    print("Coords: ", len(coords))

    paths = np.concatenate(paths_list)
    print("Image Paths: ", len(paths))

    ########## Llegim les metadates de cada pacient #################################
    #################################################################################

    histopath = r"24_09_2025_pT1_CRC_CASOS_DEFINITIUS_AMB_ITEMS_HISTOLOGICS_fixed_N0s.xlsx"

    histopath_df = pd.read_excel(histopath)

    histopath_df = histopath_df.rename(columns={'Annotated slide ': "slides"})

    histopath_df.slides = histopath_df.slides.str.replace(';', ',')

    histopath_df[
        "PATHOLOGIST SCORE. Stage of CRC (N)TNM Colorectal Cancer 8th edition (NX/ N0/N1/N1a /N1b/N1c /N2 /N2a /N2b): N0=Negatiu; NX=Dubtos; Resta=Positiu"] = \
    histopath_df[
        "PATHOLOGIST SCORE. Stage of CRC (N)TNM Colorectal Cancer 8th edition (NX/ N0/N1/N1a /N1b/N1c /N2 /N2a /N2b): N0=Negatiu; NX=Dubtos; Resta=Positiu"].replace(
        {"NX": -1, "N0": 0, "N1": 1, "N1a": 1, "N1b": 1, "N1c": 1, "N2a": 1, "N2b": 1}).astype(int)

    histopath_df = histopath_df.rename(columns={
        "PATHOLOGIST SCORE. Stage of CRC (N)TNM Colorectal Cancer 8th edition (NX/ N0/N1/N1a /N1b/N1c /N2 /N2a /N2b): N0=Negatiu; NX=Dubtos; Resta=Positiu": "PATHOLOGIST SCORE"})

    histopath_df = histopath_df.rename(columns={
        "CODE": "Patient_CODE",
        "Data Access Group": "hospital",
        'Lymphovascular invasion': "LVI",
        'Presence of Tumor Budding': "Budding",
        'Degree of differentiation': "Degree",
        'Vertical margin': "VRM",
        'Horizontal margin': "HRM",
        'Mucinous ADK': 'Mucinous',
        'Perineural invasion': 'PI',
        'Depth of submucosal invasion (in mm)': 'Depth',

    })

    histopath_df['Budding'] = histopath_df['Budding'].replace(
        {"Bd0": 0, "Bd1": 0, "bd0": 0, "Bd2": 1, "Bd3": 1, "Not applicable": 2}).infer_objects()

    histopath_df['LVI'] = histopath_df['LVI'].replace(
        {"No": 0, "Yes": 1, "Bd2": 1, "Bd3": 1, "Not applicable": 2}).infer_objects()

    histopath_df['Degree'] = histopath_df['Degree'].replace(
        {"Low grade (G1 and G2)": 0, "High grade (G3 and G4)": 1}).infer_objects()

    histopath_df['VRM'] = histopath_df['VRM'].replace(
        {"Free of lesion (>1 mm)": 0, "Free of lesion (0.1-1 mm)": 0, "Afected by adenocarcinoma": 1,
         "Indeterminate": 2}).infer_objects()

    histopath_df['HRM'] = histopath_df['HRM'].replace(
        {"Free of lesion": 0, "Afected by adenoma or adenocarcinoma in situ (pTis)": 1,
         "Afected by infiltrating adenocarcinoma": 1, "Indeterminate": 2}).infer_objects()

    histopath_df['PI'] = histopath_df['PI'].replace({"No": 0, "Yes": 1}).infer_objects()

    histopath_df['Depth'].where(histopath_df['Depth'] <= 1, 0, inplace=True)
    histopath_df['Depth'].where(histopath_df['Depth'] > 1, 1, inplace=True)

    # histopath_df['Mucinous']=histopath_df['Mucinous'].replace({"No":0, "Yes":1})

    histopath_df = histopath_df.replace({np.nan: 2})

    # Filtrem els pacients amb diagnostic NX
    histopath_df = histopath_df[histopath_df["PATHOLOGIST SCORE"] != -1]

    pat_histo = histopath_df["Patient_CODE"]

    nx_label = histopath_df["PATHOLOGIST SCORE"]
    histo_data = histopath_df[["Budding", "LVI", "Degree", "VRM", "PI", "Depth", "HRM"]]

    # print("what: ", len(pat_histo))
    # print("doublew: ", pd.Series(patients).value_counts())

    ## FILTREM PACIENTS NX Baixem de 401 a 370.
    pat_nx_dict = {str(key): int(value) for key, value in zip(pat_histo, nx_label)}
    pat_histodata_dict = {str(key): value for key, value in zip(pat_histo, histo_data.values)}

    return pat_nx_dict, pat_histodata_dict, features, affectation, hospitals, patients, slides, coords, paths