"""
@author: Enric Ferrera González
"""
#Stardard libray imports

# Third-party imports
from pathlib import Path
import torch

# Local application imports
from enric.fix_seed import fix_seeds
from enric.load_cls import load_cls_metadata
from enric.load_cls import patient_dict_builder

from enric.graph_creation.fully_connected.fully_connected_graph_creation import fully_connected_graph_creation
from enric.graph_creation.knn.knn_graph_creation import knn_graph_creation
from enric.graph_creation.radius.radius_graph_creation import radius_graph_creation

from enric.calculate_class_weights import calculate_class_weights
from enric.cross_validation import run_cross_validation

#------FIXED SEEDS for reproducibility-------
r_seed=123
fix_seeds(r_seed=123)

# Load CLS, data paths and indexes
npz_path = Path(r"../Data/cls_ALL")

# Loading data and creating the patient_dict
pat_nx_dict, pat_histodata_dict, features, affectation, hospitals, patients, slides, coords = load_cls_metadata(npz_path)

# Intersecció entre imatges processades i metadades associades 370 a 230 pacients.
print('------ filtering out patients without diagnosis...-------')
patient_dict = patient_dict_builder(features, affectation, hospitals, patients, slides, coords, pat_nx_dict, pat_histodata_dict)

# Graph creation
#knn_graph_creation(patient_dict)
#fully_connected_graph_creation(patient_dict)
#radius_graph_creation(patient_dict)

# ================= Training & Validation =================
# Calculating class weights
weights = calculate_class_weights(patient_dict)

# Check if cuda available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

run_cross_validation(patient_dict, weights, slides, device, n_folds=10, epochs=30, batch_size=24)

