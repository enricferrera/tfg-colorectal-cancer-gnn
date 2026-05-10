# -*- coding: utf-8 -*-
"""
Created on 23/02/2026

@author: Carles Sanchez
"""


import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv,GCNConv, global_mean_pool


##########################################################################################################
################  MODELS BASATS EN GRAPH NEURAL NETWORKS (GNN) ###########################################
##########################################################################################################

# Ara mateix són models molt simples de dues capes de convolucio i un global mean pool per extreure un node representatiu per graf.
# Finalment una capa de classificació per passar de la dimensió d'entrada (dimensió de la representació del node o mostra de cada pacient) a les classes de sortida (diagnòstic).
# A implementar capes de polling per reduir el nombre de nodes i diferents maneres d'agregar l'informació dels grafs resultants per la seva classificació.


# MODEL Graph Attention Network (GAT)
class GATWeight_batch(torch.nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, heads=2, use_edge_attr=True):
        super().__init__()
        """ 
        Consulteu el help de GATConv per entendre els paràmetres d'entrada.
        """
        self.use_edge_attr = use_edge_attr
        edge_dim = 1 if use_edge_attr else None
        self.gat1 = GATConv(in_ch, hidden_ch, heads=heads, concat=True, edge_dim=edge_dim)
        self.gat2 = GATConv(hidden_ch*heads, hidden_ch, heads=1, concat=True, edge_dim=edge_dim)
        self.classifier = torch.nn.Linear(hidden_ch, out_ch)

    def forward(self, nodes, edges, weight_edges , batch_idx):
        """
            nodes:           (N_total_nodes, in_ch)
            edges: (2, TotalEdgesBatch) - És el graf expressat per les arestes. Parelles de nodes.
            weight_edges: (1,TotalEdgesBatch) - pesos de cada aresta
            batch_idx:   (N_total_nodes,) - indica a quin graf pertany cada node
            """

        if self.use_edge_attr:

            x1 = F.relu(self.gat1(nodes, edges, weight_edges))
            x2 = F.relu(self.gat2(x1, edges, weight_edges))

        else:
            x1 = F.relu(self.gat1(nodes, edges))
            x2 = F.relu(self.gat2(x1, edges))

        # === Classification ===
        g = global_mean_pool(x2, batch_idx)

        return self.classifier(g)


# MODEL Graph Convolutional Network (GCN)
class GCNWithAgg(torch.nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, use_edge_weight=True):
        super().__init__()
        """ 
                Consulteu el help de GCNConv per entendre els paràmetres d'entrada.
        """
        self.use_edge_weight = use_edge_weight
        self.gcn1 = GCNConv(in_ch, hidden_ch)
        self.gcn2 = GCNConv(hidden_ch, hidden_ch)
        self.lin = torch.nn.Sequential(
            torch.nn.Linear(hidden_ch, hidden_ch),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(hidden_ch),
            torch.nn.Linear(hidden_ch, out_ch)
        )

    def forward(self, nodes, edges, weight_edges , batch_idx):
        """
            nodes:           (N_total_nodes, in_ch)
            edges: (2, TotalEdgesBatch) - És el graf expressat per les arestes. Parelles de nodes.
            weight_edges: (1,TotalEdgesBatch) - pesos de cada aresta
            batch_idx:   (N_total_nodes,) - indica a quin graf pertany cada node
        """

        if self.use_edge_weight:
            x1 = F.relu(self.gcn1(nodes, edges, edge_weight=weight_edges))
            x2 = F.relu(self.gcn2(x1, edges, edge_weight=weight_edges))
        else:
            x1 = F.relu(self.gcn1(nodes, edges))
            x2 = F.relu(self.gcn2(x1, edges))

        g = global_mean_pool(x2, batch_idx)
        return self.lin(g)


# Altres models de l'estat de l'art??