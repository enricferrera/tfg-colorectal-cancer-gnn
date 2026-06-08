import numpy as np
import torch
from pathlib import Path
import pandas as pd


def load_cls_metadata(npz_path):
    """
    Loads patch-level features from .npz files and integrates them with clinical metadata from Excel.

    Args:
        npz_path (Path): Path to the directory containing hospital .npz files.

    Returns:
        tuple: A tuple containing:
            - pat_nx_dict (dict): Map of Patient IDs to metastasis status (0 for N0, 1 for N1+).
            - pat_histodata_dict (dict): Map of Patient IDs to processed clinical feature arrays.
            - features (torch.Tensor): Concatenated image embeddings (CLS tokens) for all patches.
            - affectation (torch.Tensor): Local tumor density values for each patch.
            - hospitals (np.ndarray): Hospital names associated with each patch.
            - patients (np.ndarray): Patient IDs associated with each patch.
            - slides (np.ndarray): Slide IDs associated with each patch.
            - coords (np.ndarray): X, Y spatial coordinates for each patch.
    """

    tensors_list = []
    y_list = []
    patients_list = []
    hosps_list = []
    slides_list = []
    coords_list = []
    for i, link in enumerate(npz_path.glob("*.npz")):

        data = np.load(link, allow_pickle=True)

        features = data['embeddingCLS']
        features = torch.from_numpy(features)

        affectation = data['infiltrations']
        affectation = torch.from_numpy(affectation)

        patients = data['patient_list']

        hospitals = data['hospitals']
        slides = data['slides']
        coords = data['coords']

        tensors_list.append(features)
        y_list.append(affectation)

        patients_list.append(patients)
        hosps_list.append(hospitals)
        slides_list.append(slides)

        for coord_pack in coords:
            coords_list.append(coord_pack[0])

    features = torch.cat(tensors_list, dim=0)
    print("Features: ", features.shape)

    affectation = torch.cat(y_list, dim=0)
    print("Y: ", affectation.shape)

    patients = np.concatenate(patients_list)
    print("Patients: ", np.unique(patients).shape[0])

    hospitals = np.concatenate(hosps_list)
    print("Hospitals: ", len(np.unique(hospitals)))

    slides = np.concatenate(slides_list)
    print("Slides: ", len(np.unique(slides)))

    coords = np.stack(coords_list, axis=0)
    print("Coords: ", len(coords))
    print("Patches: ", features.shape[0])

    ########## Llegim les metadates de cada pacient #################################
    #################################################################################

    filename = "24_09_2025_pT1_CRC_CASOS_DEFINITIUS_AMB_ITEMS_HISTOLOGICS_fixed_N0s.xlsx"
    # Try resolving relative to project root (two levels up from src/dataset/)
    histopath = Path(__file__).resolve().parent.parent.parent / filename

    if not histopath.exists():
        # Fallback: Try current directory (legacy behavior)
        histopath = Path(filename)

    print(f"Loading metadata from: {histopath}")
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
        {"Bd0": 0, "Bd1": 0, "bd0": 0, "bd1": 0, "Bd2": 1, "Bd3": 1, "bd2": 1, "bd3": 1, "Not applicable": 2}).infer_objects(copy=False)

    histopath_df['LVI'] = histopath_df['LVI'].replace(
        {"No": 0, "Yes": 1, "Bd2": 1, "Bd3": 1, "Not applicable": 2}).infer_objects(copy=False)

    histopath_df['Degree'] = histopath_df['Degree'].replace(
        {"Low grade (G1 and G2)": 0, "High grade (G3 and G4)": 1}).infer_objects(copy=False)

    histopath_df['VRM'] = histopath_df['VRM'].replace(
        {"Free of lesion (>1 mm)": 0, "Free of lesion (0.1-1 mm)": 0, "Afected by adenocarcinoma": 1,
         "Indeterminate": 2}).infer_objects(copy=False)

    histopath_df['HRM'] = histopath_df['HRM'].replace(
        {"Free of lesion": 0, "Afected by adenoma or adenocarcinoma in situ (pTis)": 1,
         "Afected by infiltrating adenocarcinoma": 1, "Indeterminate": 2}).infer_objects(copy=False)

    histopath_df['PI'] = histopath_df['PI'].replace({"No": 0, "Yes": 1}).infer_objects(copy=False)

    histopath_df['Depth'] = histopath_df['Depth'].where(histopath_df['Depth'] <= 1, 0)
    histopath_df['Depth'] = histopath_df['Depth'].where(histopath_df['Depth'] > 1, 1)

    histopath_df = histopath_df.replace({np.nan: 2})

    # Filtrem els pacients amb diagnostic NX
    histopath_df = histopath_df[histopath_df["PATHOLOGIST SCORE"] != -1]

    pat_histo = histopath_df["Patient_CODE"]
    nx_label = histopath_df["PATHOLOGIST SCORE"]
    histo_data = histopath_df[["Budding", "LVI", "Degree", "VRM", "PI", "Depth", "HRM"]]

    pat_nx_dict = {str(key): int(value) for key, value in zip(pat_histo, nx_label)}
    pat_histodata_dict = {str(key): value for key, value in zip(pat_histo, histo_data.values)}

    return pat_nx_dict, pat_histodata_dict, features, affectation, hospitals, patients, slides, coords


def patient_dict_builder(features, affectation, hospitals, patients, slides, coords, pat_nx_dict, pat_histodata_dict):
    """
    Groups patch-level data into a hierarchical dictionary organized by patient.
    """

    patient_dict = {}
    for i, pat in enumerate(patients):
        if pat in pat_nx_dict:
            label = int(pat_nx_dict[pat])
            histodata = np.array(pat_histodata_dict[pat]).astype(float)

            megapatch_dict = {
                'hospital': hospitals[i],
                'patient': pat,
                'affectation': affectation[i],
                'features': features[i],
                'coords': coords[i],
                'slide': slides[i],
            }

            if pat not in patient_dict:
                patient_dict[pat] = {
                    'label': label,
                    'megapatches': [],
                    'histodata': torch.from_numpy(histodata),
                }

            patient_dict[pat]['megapatches'].append(megapatch_dict)

    print(len(patient_dict), "patients")
    
    unique_slides = set()
    slides_per_patient = []
    patches_per_patient = []
    
    for pat in patient_dict:
        patient_slides = set()
        num_patches = len(patient_dict[pat]["megapatches"])
        patches_per_patient.append(num_patches)
        
        for mp in patient_dict[pat]["megapatches"]:
            slide_id = mp["slide"]
            unique_slides.add(slide_id)
            patient_slides.add(slide_id)
        slides_per_patient.append(len(patient_slides))

    print(len(unique_slides), "slides")
    if slides_per_patient:
        print(f"Max slides per patient: {max(slides_per_patient)}")
        print(f"Min slides per patient: {min(slides_per_patient)}")
    
    if patches_per_patient:
        print(f"Max patches per patient: {max(patches_per_patient)}")
        print(f"Min patches per patient: {min(patches_per_patient)}")

    return patient_dict
