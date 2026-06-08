import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, trustworthiness
import re
from pathlib import Path

def get_next_version(results_dir):
    """Busca el número de versió vX més alt a la carpeta results/ i retorna el següent."""
    if not os.path.exists(results_dir):
        return 1
    folders = os.listdir(results_dir)
    version_numbers = [0]
    pattern = re.compile(r"^v(\d+)$")
    for f in folders:
        if os.path.isdir(os.path.join(results_dir, f)):
            match = pattern.search(f)
            if match:
                version_numbers.append(int(match.group(1)))
    return max(version_numbers) + 1

def plot_tsne_affectation(embeddings, affectations, save_dir, filename="tsne_affectation.png", perplexity=30, patient_label=None, thresholds=None, info_str=""):
    """Projecta embeddings a 2D i acoloreix segons afectació."""
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if isinstance(affectations, torch.Tensor):
        affectations = affectations.cpu().numpy()

    low_threshold, high_threshold = thresholds if thresholds else (0.1, 0.9)

    print(f"Calculant t-SNE... Punts: {embeddings.shape[0]} | Perplexity: {perplexity}")
    tsne = TSNE(n_components=2, random_state=123, n_jobs=-1, init='pca', learning_rate='auto', verbose=1, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    class_affectation = np.full_like(affectations, 2, dtype=int)
    class_affectation[affectations < low_threshold] = 0
    class_affectation[affectations > high_threshold] = 1

    n_sa = np.sum(class_affectation == 0)
    n_inf = np.sum(class_affectation == 1)
    n_front = np.sum(class_affectation == 2)
    print(f"Distribució visual: {n_sa} Sa, {n_inf} Infiltrat, {n_front} Front")

    plt.figure(figsize=(12, 10))
    dot_size = 6 if len(embeddings) < 2000 else 4 if len(embeddings) < 5000 else 2
    dot_alpha = 0.6 if len(embeddings) < 5000 else 0.4
    
    masks = [
        (class_affectation == 0, 'green', f'Sa (<{low_threshold})'),
        (class_affectation == 1, 'red', f'Infiltrat (>{high_threshold})'),
        (class_affectation == 2, 'orange', f'Front ({low_threshold}-{high_threshold})')
    ]

    for mask, color, label in masks:
        if np.any(mask):
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], s=dot_size, color=color, label=label, alpha=dot_alpha)

    title = f"t-SNE CLS | {filename}"
    if patient_label is not None:
        title += f"\nCLASSE PACIENT: {'N0 (Sa)' if patient_label == 0 else 'N1 (Metàstasi)'}"
    if info_str:
        title += f"\n{info_str}"
    
    plt.title(title)
    plt.legend(markerscale=2)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    import sys
    current_dir = Path(__file__).resolve().parent
    src_dir = current_dir.parent
    if str(src_dir) not in sys.path:
        sys.path.append(str(src_dir))
    
    try:
        from dataset.load_cls import load_cls_metadata, patient_dict_builder
        from utils.seed import fix_seeds
    except ImportError as e:
        print(f"Error carregant mòduls: {e}")
        sys.exit(1)

    fix_seeds(r_seed=123)

    npz_path = src_dir.parent / "data" / "NEW_DATASET_cls_2048"
    results_root = src_dir.parent / "results" / "visualization"
    v_num = get_next_version(results_root)
    v_tag = f"v{v_num}"
    v_folder = results_root / v_tag
    individual_results = v_folder / "patients"

    if not npz_path.exists():
        print(f"ERROR: No dades a {npz_path}")
    else:
        print(f"--- INICIANT EXECUCIÓ {v_tag} (Agregació per pacient, totes les WSI) ---")
        pat_nx_dict, pat_histodata_dict, features, affectation, hospitals, patients, slides, coords = load_cls_metadata(npz_path)
        
        # DETERMINEM ULLS GLOBALS
        v_max_global = affectation.max().item()
        is_zero_one = v_max_global <= 1.05
        low_th = 0.1 if is_zero_one else 10
        high_th = 0.9 if is_zero_one else 90
        global_thresholds = (low_th, high_th)

        # 3. GLOBAL ESTRATIFICAT
        idx_all = np.arange(len(affectation))
        aff_np = affectation.cpu().numpy()
        idx_sa = idx_all[aff_np < low_th]
        idx_inf = idx_all[aff_np > high_th]
        idx_front = idx_all[(aff_np >= low_th) & (aff_np <= high_th)]

        def sample(idx, n): return np.random.choice(idx, min(len(idx), n), replace=False) if len(idx)>0 else np.array([], dtype=int)
        sel_idx = np.concatenate([sample(idx_sa, 10000), sample(idx_inf, 10000), sample(idx_front, 10000)])
        np.random.shuffle(sel_idx)

        print(f"\n>>> Generant GLOBAL ESTRATIFICAT ({v_tag})...")
        plot_tsne_affectation(features[sel_idx], affectation[sel_idx], str(v_folder), f"global_tsne_stratified_{v_tag}.png", perplexity=50, thresholds=global_thresholds)

        # 4. AGREGACIÓ PER PACIENT (Totes les seves WSIs)
        patient_dict = patient_dict_builder(features, affectation, hospitals, patients, slides, coords, pat_nx_dict, pat_histodata_dict)
        
        pats_info = []
        for pid, d in patient_dict.items():
            p_affs = np.array([mp['affectation'].item() if isinstance(mp['affectation'], torch.Tensor) else mp['affectation'] for mp in d['megapatches']])
            p_slides = len(set([mp['slide'] for mp in d['megapatches']]))
            pats_info.append({
                'id': pid,
                'label': d['label'],
                'infiltrat_count': np.sum(p_affs >= low_th),
                'total': len(p_affs),
                'num_slides': p_slides
            })

        # Triem 5 N1 amb MÉS infiltració (sumant totes les seves WSIs) i 5 N0
        pats_n1_sorted = sorted([p for p in pats_info if p['label'] == 1], key=lambda x: x['infiltrat_count'], reverse=True)
        pats_n0 = [p for p in pats_info if p['label'] == 0]
        selected_pats = [p['id'] for p in pats_n0[:5]] + [p['id'] for p in pats_n1_sorted[:5]]
        
        print(f"\n>>> Processant 10 pacients (Combinant totes les seves WSIs)...")
        for pat_id in selected_pats:
            p_data = [p for p in pats_info if p['id'] == pat_id][0]
            data = patient_dict[pat_id]
            filename = f"tsne_{'N0' if p_data['label'] == 0 else 'N1'}_{pat_id}.png"
            info_str = f"Patches: {p_data['total']} | WSIs: {p_data['num_slides']}"
            try:
                pat_feat = torch.stack([mp['features'] for mp in data['megapatches']])
                pat_aff = torch.stack([torch.tensor(mp['affectation']) if not isinstance(mp['affectation'], torch.Tensor) else mp['affectation'] for mp in data['megapatches']])
                plot_tsne_affectation(pat_feat, pat_aff, str(individual_results), filename, patient_label=p_data['label'], thresholds=global_thresholds, info_str=info_str)
            except Exception as e:
                print(f"Error en {pat_id}: {e}")
