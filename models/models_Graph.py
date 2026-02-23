import os
import numpy as np
from random import shuffle
import torch
import torch.nn.functional as F
from alembic.command import heads
from torch_geometric.nn import GATConv, global_mean_pool,global_max_pool, TopKPooling, dense_diff_pool
from torch_geometric.data import Data, Batch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from PIL import Image
from transformers import AutoImageProcessor, ViTModel
import matplotlib.pyplot as plt
import gc
from torch_geometric.utils import to_dense_batch, to_dense_adj

# -----------------------
# CONFIGURATION AND DATA LOADING
# LOAD image and labels
data = np.load(r'D:\CNN\ColonCancer\HistopatDiagnosis\TFGPau2025\Code\LargetissueDades_48_Norm.npz', allow_pickle=True)
X_no_hosp = data['X_no_hosp']; y_no_hosp = data['y_no_hosp']; PatID_no_hosp = data['PatID_no_hosp']
X_hosp    = data['X_hosp'];    y_hosp    = data['y_hosp']
#
# # LOAD Features
data = np.load(r'D:\CNN\ColonCancer\HistopatDiagnosis\TFGPau2025\Code\DBLarge_FeatMatNew_norm.npz', allow_pickle=True)
f_no=data['features_list']
a_no=data['attn_matrices']



class GATWeight_batch(torch.nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch,num_clusters1=8, num_clusters2=4, heads=1, use_edge_attr=True, threshold=0.0):
        super().__init__()
        self.use_edge_attr = use_edge_attr

        edge_dim = 1 if use_edge_attr else None
        self.agg = torch.nn.Conv2d(12, 1, kernel_size=1)
        self.gat1 = GATConv(in_ch, hidden_ch, heads=heads, concat=True, edge_dim=edge_dim)
        self.gat_pool_1 = GATConv(in_ch, num_clusters1, heads=heads, concat=True, edge_dim=edge_dim)
        self.gat2 = GATConv(hidden_ch*heads, hidden_ch, heads=1, concat=True, edge_dim=edge_dim)
        self.gat_pool_2 = GATConv(hidden_ch*heads, num_clusters2, heads=1, concat=True, edge_dim=edge_dim)
        self.classifier = torch.nn.Linear(num_clusters2 * hidden_ch, out_ch)
        self.threshold = threshold

    def forward(self, x, attn_tensor, batch_idx):
        """
            x:           (N_total_nodes, in_ch)
            attn_tensor: (B, 12, 196, 196)
            batch_idx:   (N_total_nodes,) - indica a qué muestra pertenece cada nodo
            """
        x = x.float()
        # x = F.layer_norm(x, x.shape[1:])
        attn_tensor = attn_tensor.float()

        B = attn_tensor.size(0)  # batch size
        #attn_tensor = attn_tensor.view(B, 12, 196, 196)
        agg_mat = self.agg(attn_tensor)  # (B, 1, 196, 196)
        agg_mat = agg_mat.squeeze(1)  # (B, 196, 196)

        edge_indices = []
        edge_attrs = []
        node_offset = 0
        num_nodes_per_graph = x.size(0) // B  # Asumimos mismo número de nodos por muestra

        for i in range(B):
            adj = agg_mat[i]  # (196, 196)
            mask = adj > self.threshold
            edge_index = mask.nonzero(as_tuple=False).t().contiguous()  # (2, E)
            edge_index += node_offset  # shift indices for batch concat
            # edge_attr = torch.exp(adj[mask].unsqueeze(-1)*10) if self.use_edge_attr else None  # (E, 1)
            edge_attr = adj[mask].unsqueeze(-1) if self.use_edge_attr else None  # (E, 1)

            edge_indices.append(edge_index)
            edge_attrs.append(edge_attr)
            node_offset += num_nodes_per_graph

        # Concatenar todos los grafos del batch
        edge_index = torch.cat(edge_indices, dim=1)  # (2, total_E)
        x = x.view(-1, x.size(-1))
        x = F.layer_norm(x, x.shape[1:])

        if self.use_edge_attr:
            edge_attr = torch.cat(edge_attrs, dim=0)  # (total_E, 1)
            z1 = F.relu(self.gat1(x, edge_index, edge_attr))
            s1 = F.softmax(self.gat_pool_1(x, edge_index, edge_attr),dim=1)

            x_dense_1, mask_1 = to_dense_batch(z1, batch_idx)
            s1_dense, _ = to_dense_batch(s1, batch_idx, max_num_nodes=x_dense_1.size(1))
            adj_dense_1 = to_dense_adj(edge_index, batch_idx)

            x_pool_1, adj_pool_1, link_loss_1, ent_loss_1 = dense_diff_pool(x_dense_1, adj_dense_1, s1_dense, mask_1)

            # === Nivel 2 ===
            B, N, D = x_pool_1.size()
            x2 = x_pool_1.view(B * N, D)
            batch2 = torch.arange(B).repeat_interleave(N).to(x.device)

            edge_index2 = (adj_pool_1 > 0).nonzero(as_tuple=False)
            edge_index2 = edge_index2[:, 1:] + edge_index2[:, 0:1] * N  # convierte (b, i, j) -> global idx
            edge_index2 = edge_index2.t().contiguous()  # (2, E)
            adj_block = torch.block_diag(*adj_pool_1)
            edge_attr2 = adj_block[edge_index2[0], edge_index2[1]].unsqueeze(-1)

            z2 = F.relu(self.gat2(x2, edge_index2, edge_attr2))
            s2 = F.softmax(self.gat_pool_2(x2, edge_index2, edge_attr2),dim=1)

            x_dense_2, mask_2 = to_dense_batch(z2, batch2, max_num_nodes=s2.shape[1])
            s2_dense, _ = to_dense_batch(s2, batch2, max_num_nodes=x_dense_2.size(1))
            adj_dense_2 = to_dense_adj(edge_index2, batch2, max_num_nodes=s2.shape[1])

            x_pool_2, adj_pool_2, link_loss_2, ent_loss_2 = dense_diff_pool(x_dense_2, adj_dense_2, s2_dense, mask_2)

            # === Clasificación ===
            out = x_pool_2.view(x_pool_2.size(0), -1)
        else:
            z1 = F.relu(self.gat1(x, edge_index))
            temperature = 0.2  # o incluso 0.2
            s1 = F.softmax(self.gat_pool_1(x, edge_index, edge_attr) / temperature, dim=1)

            x_dense_1, mask_1 = to_dense_batch(z1, batch_idx)
            s1_dense, _ = to_dense_batch(s1, batch_idx, max_num_nodes=x_dense_1.size(1))
            adj_dense_1 = to_dense_adj(edge_index, batch_idx)

            x_pool_1, adj_pool_1, link_loss_1, ent_loss_1 = dense_diff_pool(x_dense_1, adj_dense_1, s1_dense, mask_1)

            # === Nivel 2 ===
            B, N, D = x_pool_1.size()
            x2 = x_pool_1.view(B * N, D)
            batch2 = torch.arange(B).repeat_interleave(N).to(x.device)

            edge_index2 = (adj_pool_1 > 0).nonzero(as_tuple=False)
            edge_index2 = edge_index2[:, 1:] + edge_index2[:, 0:1] * N  # convierte (b, i, j) -> global idx
            edge_index2 = edge_index2.t().contiguous()  # (2, E)
            adj_block = torch.block_diag(*adj_pool_1)

            z2 = F.relu(self.gat2(x2, edge_index2))
            s2 = F.softmax(self.gat_pool_2(x2, edge_index2)/ temperature, dim=1)

            x_dense_2, mask_2 = to_dense_batch(z2, batch2, max_num_nodes=s2.shape[1])
            s2_dense, _ = to_dense_batch(s2, batch2, max_num_nodes=x_dense_2.size(1))
            adj_dense_2 = to_dense_adj(edge_index2, batch2, max_num_nodes=s2.shape[1])

            x_pool_2, adj_pool_2, link_loss_2, ent_loss_2 = dense_diff_pool(x_dense_2, adj_dense_2, s2_dense, mask_2)

            # === Clasificación ===
            out = x_pool_2.view(x_pool_2.size(0), -1)

        # global pooling --> Global representation
        #g = global_mean_pool(x, batch_idx)

        return self.classifier(out), link_loss_1+link_loss_2, ent_loss_1+ent_loss_2, s1, s2

class AttnDataset(torch.utils.data.Dataset):
    def __init__(self, idxs, f_no, a_no, y_no):
        self.idxs = idxs
        self.f_no = f_no
        self.a_no = a_no
        self.y_no = y_no

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        idx = self.idxs[i]
        x=self.f_no[idx]
        attn=self.a_no[idx]
        y=self.y_no[idx]
        return x, attn, y


    # ================= STEP 4: Training & Validation =================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = torch.nn.CrossEntropyLoss().to(device)
skf = StratifiedGroupKFold(5, shuffle=True, random_state=2)
# prepare metrics and loss storage
metrics = {k:[] for k in ['recall_0','recall_1','precision_0','precision_1','f1_0','f1_1','auc','y_pred','y_scores','y_true']}
all_loss = []
best_auc, best_state = 0, None
val_loss_per_fold = []
epochs = 25
batch_size = 32
alpha = 0.0 #1 Penalització pq mantingui conectivitat
beta = 5.0 #0.1

for fold, (tr, va) in enumerate(skf.split(f_no, y_no_hosp, PatID_no_hosp),1):
    print(f"\n--- Fold {fold} ---")
    model = GATWeight_batch(768,256,2, heads=1, use_edge_attr=False,threshold=0.00).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    #opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    td = AttnDataset(tr, f_no, a_no, y_no_hosp)
    tl = DataLoader(td, batch_size=batch_size, shuffle=True)
    del td
    torch.cuda.empty_cache()
    gc.collect()

    fold_losses = []
    v_loss = []
    # training
    for epoch in range(epochs):
        model.train()
        losses = []; lossesLink = []; lossesEntropy = []
        y_true_train, y_pred_train, y_scores_train = [], [], []
        for b in tl:
            opt.zero_grad()
            batch_index = torch.arange(b[0].shape[0]).repeat_interleave(b[0].shape[1]) #creem els index del graf al que correspon cada node
            logits, lossLink, lossEnt, s1, _ = model(b[0].to(device), b[1].to(device), batch_index.to(device))
            loss = loss_fn(logits, b[2].long().to(device)) + beta*lossEnt # + alpha*lossLink
            loss.backward()
            opt.step()
            losses.append(loss.item())
            #lossesLink.append(lossLink.item())
            lossesEntropy.append(lossEnt.item())
            entropy_s1 = -torch.sum(s1 * torch.log(s1 + 1e-9), dim=1).mean()
            print(f"Entropy S1: {entropy_s1.item():.4f}")

        epoch_loss = np.mean(losses)
        #epoch_lossLink = np.mean(lossesLink)
        epoch_lossEntropy = np.mean(lossesEntropy)

        fold_losses.append(epoch_loss)
        #print(f"Epoch {epoch + 1}, Total: {loss:.2f}, Link: {epoch_lossLink:.2f}, Ent: {epoch_lossEntropy:.2f}")
        print(f"Epoch {epoch + 1}, Total: {loss:.2f}, Ent: {epoch_lossEntropy:.2f}")


    all_loss.append(fold_losses)

    # lliberem memoria
    del logits, loss,lossLink,lossEnt, b, opt, tl
    torch.cuda.empty_cache()
    gc.collect()

    vd = AttnDataset(va, f_no, a_no, y_no_hosp)
    vl = DataLoader(vd, batch_size=batch_size, shuffle=False)
    del vd
    torch.cuda.empty_cache()
    gc.collect()

    # validation
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    with torch.no_grad():
        for b in vl:
            batch_index = torch.arange(b[0].shape[0]).repeat_interleave(b[0].shape[1])  # creem els index del graf al que correspon cada node
            logits, lossLink, lossEnt, _ , _= model(b[0].to(device), b[1].to(device), batch_index.to(device))
            loss = loss_fn(logits, b[2].long().to(device)) + alpha*lossLink + beta*lossEnt
            v_loss.append(loss.item())
            probs = F.softmax(logits, dim=1).cpu()
            y_true.extend(b[2].cpu().tolist())
            y_pred.extend(probs.argmax(dim=1).cpu().tolist())
            y_scores.extend(probs[:,1].cpu().tolist())
    val_auc = roc_auc_score(y_true, y_scores)
    val_loss_per_fold.append(np.mean(v_loss))
    print(f"Fold {fold} AUC: {val_auc:.4f}")

    # record metrics
    metrics['recall_0'].append(recall_score(y_true, y_pred, pos_label=0))
    metrics['recall_1'].append(recall_score(y_true, y_pred, pos_label=1))
    metrics['precision_0'].append(precision_score(y_true, y_pred, pos_label=0))
    metrics['precision_1'].append(precision_score(y_true, y_pred, pos_label=1))
    metrics['f1_0'].append(f1_score(y_true, y_pred, pos_label=0))
    metrics['f1_1'].append(f1_score(y_true, y_pred, pos_label=1))
    metrics['auc'].append(val_auc)
    metrics["y_true"].append(y_true)
    metrics["y_pred"].append(y_pred)
    metrics['y_scores'].append(y_scores)

    # update best
    if val_auc > best_auc:
        best_auc, best_state = val_auc, model.state_dict()

    #lliberem memoria
    del logits, loss, b, model, vl
    torch.cuda.empty_cache()
    gc.collect()

# print averaged metrics
print("\nAveraged Metrics over folds:")
for k, vals in metrics.items():
    mean, std = np.mean(vals), np.std(vals)
    print(f"{k}: {mean:.4f} ± {std:.4f}")

np.savez(
    'D:\DOCENCIA\TFG\TFG 2025\PauMarti\Results\VITGATAGGw.npz',
    metrics=metrics,
    val_loss_per_fold=val_loss_per_fold
)

# ================= STEP 5: Test on Holdout =================
td = AttnDataset(range(f_no.shape[0]), f_no, a_no, y_no_hosp)
tl = DataLoader(td, batch_size=batch_size, shuffle=False)
model = GATWeight_batch(768,256,2,heads=2, use_edge_attr=True, threshold=0.2).to(device)
model.load_state_dict(best_state)
model.eval()
yt, yp, ys = [], [], []
with torch.no_grad():
    for b in tl:
        batch_index = torch.arange(b[0].shape[0]).repeat_interleave(b[0].shape[1])  # creem els index del graf al que correspon cada node
        logits, _ = model(b[0].to(device), b[1].to(device), batch_index.to(device))
        probs = F.softmax(logits, dim=1)
        yt.extend(b[2].cpu().tolist())
        yp.extend(probs.argmax(dim=1).cpu().tolist())
        ys.extend(probs[:,1].cpu().tolist())
# final holdout metrics
print("\nHoldout results:")
print(f"AUC: {roc_auc_score(yt, ys):.4f}")
print(f"Recall 0: {recall_score(yt, yp, pos_label=0):.4f}")
print(f"Recall 1: {recall_score(yt, yp, pos_label=1):.4f}")
print(f"Precision 0: {precision_score(yt, yp, pos_label=0):.4f}")
print(f"Precision 1: {precision_score(yt, yp, pos_label=1):.4f}")
print(f"F1 0: {f1_score(yt, yp, pos_label=0):.4f}")
print(f"F1 1: {f1_score(yt, yp, pos_label=1):.4f}")

del logits, b, model, tl, td, yt, yp, ys
torch.cuda.empty_cache()
gc.collect()

# ================= STEP 6: Plot Loss Curve =================
plt.figure(figsize=(10,6))
for i, losses in enumerate(all_loss):
    plt.plot(losses, label=f"Fold {i+1}")
plt.title("Training Loss per Fold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(r'D:\DOCENCIA\TFG\TFG 2025\PauMarti\Code\GATAGG_loss(Entreno).png', dpi=300)
plt.close()

# Validation loss únic per fold (punt)
plt.figure(figsize=(6, 4))

# 1) amb línia
plt.plot(
    range(1, len(val_loss_per_fold) + 1),   # X = folds 1,2,3…
    val_loss_per_fold,                      # Y = les losses
    'o-',                                   # punts amb línia
    label='Val Loss per Fold'
)

# 2) (opcional) o bé si vols un punt per fold sense línia:
# plt.scatter(range(1, len(val_loss_per_fold) + 1), val_loss_per_fold)

plt.xlabel('Fold')
plt.ylabel('Val Loss')
plt.title('Validation Loss per Fold')
plt.xticks(range(1, len(val_loss_per_fold) + 1))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('/fhome/pmarti/TFGPau/GATAGG_loss(validacio).png', dpi=300)
plt.close()

## VISUALIZE LEARNED GRAFS

import matplotlib.pyplot as plt
import seaborn as sns

def plot_assignment_matrix(s, title="Cluster assignment (soft)"):
    """
    s: Tensor of shape (num_nodes, num_clusters)
    """
    s_np = s.detach().cpu().numpy()
    plt.figure(figsize=(10, 6))
    sns.heatmap(s_np, cmap='viridis', cbar=True)
    plt.xlabel("Clusters")
    plt.ylabel("Nodes")
    plt.title(title)
    plt.show()

model = GATWeight_batch(768,256,2,heads=1, use_edge_attr=False, threshold=0.00).to(device)
model.load_state_dict(best_state)
fold, (tr, va) = next(enumerate(skf.split(f_no, y_no_hosp, PatID_no_hosp),1))
vd = AttnDataset(va, f_no, a_no, y_no_hosp)
vl = DataLoader(vd, batch_size=1, shuffle=False)
model.eval()
b = next(iter(vl))
batch_index = torch.arange(b[0].shape[0]).repeat_interleave(b[0].shape[1])  # creem els index del graf al que correspon cada node
logits, lossLink, lossEnt, s1 , s2= model(b[0].to(device), b[1].to(device), batch_index.to(device))
plot_assignment_matrix(s1)


hard_assignments = s2.argmax(dim=1)
unique, counts = torch.unique(hard_assignments, return_counts=True)

for i, c in zip(unique, counts):
    print(f"Cluster {i.item()}: {c.item()} nodes")