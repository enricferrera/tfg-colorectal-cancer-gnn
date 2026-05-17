import numpy as np
import torch
from pathlib import Path
import pandas as pd



def loadHistoImagesMetadata(imatges_of_2048_path):

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


    histopath_df['Budding']=histopath_df['Budding'].replace({"Bd0":0, "Bd1":0,"bd0": 0,  "Bd2":1, "Bd3":1, "Not applicable":2}).infer_objects(copy=False)

    histopath_df['LVI']=histopath_df['LVI'].replace({"No":0, "Yes":1, "Bd2":1, "Bd3":1, "Not applicable":2}).infer_objects(copy=False)

    histopath_df['Degree']=histopath_df['Degree'].replace({"Low grade (G1 and G2)":0, "High grade (G3 and G4)":1}).infer_objects(copy=False)

    histopath_df['VRM']=histopath_df['VRM'].replace({"Free of lesion (>1 mm)":0, "Free of lesion (0.1-1 mm)":0, "Afected by adenocarcinoma":1, "Indeterminate":2}).infer_objects(copy=False)

    histopath_df['HRM']=histopath_df['HRM'].replace({"Free of lesion":0, "Afected by adenoma or adenocarcinoma in situ (pTis)":1, "Afected by infiltrating adenocarcinoma":1, "Indeterminate":2}).infer_objects(copy=False)

    histopath_df['PI']=histopath_df['PI'].replace({"No":0, "Yes":1}).infer_objects(copy=False)

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

    #les features son les imatges, és a dir, els paths.
    return pat_nx_dict, pat_histodata_dict, paths, affectation, hospitals, patients, slides, coords, paths