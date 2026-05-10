"""
@author: Enric Ferrera González
"""
# Standard library imports
from pathlib import Path
import torch
import gc
import numpy as np

# Local application imports
from enric.fix_seed import fix_seeds
from enric.load_cls import load_cls_metadata
from enric.load_cls import patient_dict_builder
from enric.calculate_class_weights import calculate_class_weights
from enric.cross_validation_graph import run_cross_validation_graph

# ------ LIMPIEZA DE VRAM ANTES DE EMPEZAR -------
def clean_vram():
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"VRAM Liberada. Memoria reservada: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

clean_vram()

# ------ FIXED SEEDS for reproducibility -------
r_seed = 123
fix_seeds(r_seed=r_seed)

# ================= Configuration =================
# Si los Fully Connected siguen fallando, cambia 'fully_connected' por 'knn' abajo:
graphs_dir = Path(__file__).parent / "graph_creation" / "knn" / "graphs"
npz_path = Path(r"../Data/cls_ALL")

# ================= Data Loading =================
pat_nx_dict, pat_histodata_dict, features, affectation, hospitals, patients, slides, coords = load_cls_metadata(npz_path)
patient_dict = patient_dict_builder(features, affectation, hospitals, patients, slides, coords, pat_nx_dict, pat_histodata_dict)

patient_list = list(patient_dict.keys())
label_list = [patient_dict[pat]['label'] for pat in patient_list]

# ================= Training & Validation =================
weights = calculate_class_weights(patient_dict)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# EJECUCIÓN CON PARÁMETROS REDUCIDOS PARA EVITAR OOM
metrics = run_cross_validation_graph(
    graphs_dir=graphs_dir,
    patient_list=patient_list,
    label_list=label_list,
    weights=weights,
    device=device,
    n_folds=10,
    epochs=30,
    batch_size=1,        # Mantenemos 1 por seguridad
    minibatch_size=10,   # Acumulación de gradientes
    hidden_ch=128        # Reducido de 512 a 128 para ahorrar memoria
)
clean_vram()
