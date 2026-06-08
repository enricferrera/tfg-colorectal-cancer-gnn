import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Assegurar que podem importar els mòduls locals
current_dir = Path(__file__).resolve().parent
if str(current_dir.parent) not in sys.path:
    sys.path.append(str(current_dir.parent))

from dataset.load_cls import load_cls_metadata, patient_dict_builder

def analyze_n1_patients():
    print("--- INICIANT ESTUDI DE PACIENTS N1 ---")
    npz_path = current_dir.parent.parent / "data" / "NEW_DATASET_cls_2048"
    
    if not npz_path.exists():
        print(f"ERROR: No s'ha trobat la carpeta de dades a {npz_path}")
        return

    # 1. Carregar dades
    print("Carregant dades (això pot trigar uns segons)...")
    pat_nx_dict, pat_histodata_dict, features, affectation, hospitals, patients, slides, coords = load_cls_metadata(npz_path)
    
    patient_dict = patient_dict_builder(features, affectation, hospitals, patients, slides, coords, pat_nx_dict, pat_histodata_dict)

    # Noms de les columnes histològiques segons load_cls.py
    histo_columns = ["Budding", "LVI", "Degree", "VRM", "PI", "Depth", "HRM"]

    # 2. Analitzar els pacients N1
    n1_patients = {pid: data for pid, data in patient_dict.items() if data['label'] == 1}
    print(f"\nS'han trobat {len(n1_patients)} pacients amb etiqueta N1 (Metàstasi).")

    results = []
    
    # Determinar el llindar de "Sa"
    v_max_global = affectation.max().item()
    is_zero_one = v_max_global <= 1.05
    low_th = 0.1 if is_zero_one else 10

    for pid, data in n1_patients.items():
        # Llista d'afectacions per a cada patch d'aquest pacient
        affs = [mp['affectation'].item() if isinstance(mp['affectation'], torch.Tensor) else mp['affectation'] for mp in data['megapatches']]
        affs = np.array(affs)
        
        num_patches = len(affs)
        num_infiltrats = np.sum(affs >= low_th) # Patches que NO són completament sans
        max_aff = affs.max()
        
        # Histodata
        h_data = data['histodata'].numpy()
        
        results.append({
            'Patient_ID': pid,
            'Hospital': data['megapatches'][0]['hospital'],
            'Total_Patches': num_patches,
            'Infiltrated_Patches': num_infiltrats,
            'Max_Affectation': max_aff,
            'Pct_Infiltrated': (num_infiltrats / num_patches) * 100 if num_patches > 0 else 0,
            **{histo_columns[i]: h_data[i] for i in range(len(histo_columns))}
        })

    df = pd.DataFrame(results)

    # 3. Identificar pacients discordants (N1 però sense cap patch infiltrat)
    discordants = df[df['Infiltrated_Patches'] == 0]
    concordants = df[df['Infiltrated_Patches'] > 0]

    print("\n" + "="*50)
    print("RESUM DE L'ESTUDI:")
    print("="*50)
    print(f"Pacients N1 Concordants (tenen infiltració local): {len(concordants)}")
    print(f"Pacients N1 Discordants (tot el teixit analitzat és sa): {len(discordants)}")
    
    print("\n--- PACIENTS DISCORDANTS (Mostrant els primers 15) ---")
    if len(discordants) > 0:
        display_cols = ['Patient_ID', 'Total_Patches', 'Max_Affectation', 'Hospital']
        print(discordants[display_cols].head(15).to_string(index=False))
        
        # Guardar l'estudi complet
        output_file = current_dir.parent.parent / "results" / "n1_discordant_analysis.csv"
        df.sort_values(by='Infiltrated_Patches').to_csv(output_file, index=False)
        print(f"\nS'ha guardat un informe complet a: {output_file}")
    else:
        print("No hi ha pacients discordants!")

    print("\nPOSSIBLES CAUSES D'AQUESTA DISCORDÀNCIA:")
    print("1. El tumor primari ja va ser extirpat anteriorment i les WSI actuals són de marges quirúrgics sans.")
    print("2. Biaix de mostreig: la generació de patches (el pas previ a .npz) va ignorar la zona del tumor.")
    print("3. La infiltració que va causar la N1 era microscòpica (micrometástasis) i no va caure dins de l'àrea escanejada.")
    print("4. Les WSIs processades corresponen a ganglis negatius o teixit sa del pacient, no al tumor primari positiu.")

if __name__ == "__main__":
    analyze_n1_patients()