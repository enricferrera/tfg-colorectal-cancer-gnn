import torch
import numpy as np
from pathlib import Path
from load_cls import load_cls_metadata, patient_dict_builder

npz_path = Path('Data/NEW_DATASET_cls_2048')
results = load_cls_metadata(npz_path)
patient_dict = patient_dict_builder(*results[2:], results[0], results[1])

pats_n1 = [p for p, d in patient_dict.items() if d['label'] == 1]
print(f"Total N1 patients: {len(pats_n1)}")

for pid in pats_n1[:10]:
    affs = np.array([mp['affectation'].item() if isinstance(mp['affectation'], torch.Tensor) else mp['affectation'] for mp in patient_dict[pid]['megapatches']])
    print(f"Patient {pid} (N1): max_aff={affs.max():.4f}, num_patches={len(affs)}, num_above_0.1={(affs >= 0.1).sum()}")
