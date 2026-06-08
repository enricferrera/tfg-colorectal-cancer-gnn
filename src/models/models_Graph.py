# -*- coding: utf-8 -*-
"""
Created on 23/02/2026

@author: Carles Sanchez
"""


import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_max_pool, TopKPooling


##########################################################################################################
################  MODELS BASATS EN GRAPH NEURAL NETWORKS (GNN) ###########################################
##########################################################################################################

# Ara mateix són models molt simples de dues capes de convolucio i un global mean pool per extreure un node representatiu per graf.
# Finalment una capa de classificació per passar de la dimensió d'entrada (dimensió de la representació del node o mostra de cada pacient) a les classes de sortida (diagnòstic).
# A implementar capes de polling per reduir el nombre de nodes i diferents maneres d'agregar l'informació dels grafs resultants per la seva classificació.


# MODEL Graph Attention Network (GAT)
class GATWeight_batch(torch.nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, heads=2, use_edge_attr=True, dropout=0.0):
        super().__init__()
        """ 
        Consulteu el help de GATConv per entendre els paràmetres d'entrada.
        """
        self.use_edge_attr = use_edge_attr
        self.dropout = dropout
        edge_dim = 1 if use_edge_attr else None
        self.gat1 = GATConv(in_ch, hidden_ch, heads=heads, concat=True, edge_dim=edge_dim, dropout=dropout)
        self.gat2 = GATConv(hidden_ch*heads, hidden_ch, heads=1, concat=True, edge_dim=edge_dim, dropout=dropout)
        # The classifier receives features after pooling.
        # Since the last GAT layer (gat2) has hidden_ch output and 1 head (with concat=True), 
        # the dimension here is hidden_ch.
        self.classifier = torch.nn.Linear(hidden_ch, out_ch)

    def forward(self, nodes, edges, weight_edges , batch_idx):
        """
            nodes:           (N_total_nodes, in_ch)
            edges: (2, TotalEdgesBatch) - És el graf expressat per les arestes. Parelles de nodes.
            weight_edges: (TotalEdgesBatch, 1) or (TotalEdgesBatch,) - pesos de cada aresta
            batch_idx:   (N_total_nodes,) - indica a quin graf pertany cada node
            """

        if self.use_edge_attr and weight_edges is not None:
            # GATConv with edge_dim=1 expects (E, 1)
            if weight_edges.dim() == 1:
                weight_edges = weight_edges.unsqueeze(-1)
            x1 = F.relu(self.gat1(nodes, edges, weight_edges))
            x2 = F.relu(self.gat2(x1, edges, weight_edges))
        else:
            x1 = F.relu(self.gat1(nodes, edges))
            x2 = F.relu(self.gat2(x1, edges))

        # === Classification ===
        g = global_mean_pool(x2, batch_idx)
        g = F.dropout(g, p=self.dropout, training=self.training)

        return self.classifier(g)


# MODEL GAT amb Max Pooling jeràrquic
class GATWithAggMaxPool(torch.nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, heads=2, use_edge_attr=True, pool_ratio=0.5, dropout=0.0):
        super().__init__()
        """ 
        GAT amb dues capes de convolució d'atenció i dues capes de Max Pooling (TopKPooling).
        """
        self.use_edge_attr = use_edge_attr
        self.dropout = dropout
        edge_dim = 1 if use_edge_attr else None
        
        self.gat1 = GATConv(in_ch, hidden_ch, heads=heads, concat=True, edge_dim=edge_dim, dropout=dropout)
        self.pool1 = TopKPooling(hidden_ch * heads, ratio=pool_ratio)
        
        self.gat2 = GATConv(hidden_ch * heads, hidden_ch, heads=1, concat=True, edge_dim=edge_dim, dropout=dropout)
        self.pool2 = TopKPooling(hidden_ch, ratio=pool_ratio)
        
        self.classifier = torch.nn.Linear(hidden_ch, out_ch)

    def forward(self, nodes, edges, weight_edges, batch_idx):
        """
            nodes:           (N_total_nodes, in_ch)
            edges: (2, TotalEdgesBatch) - És el graf expressat per les arestes. Parelles de nodes.
            weight_edges: (TotalEdgesBatch, 1) or (TotalEdgesBatch,) - pesos de cada aresta
            batch_idx:   (N_total_nodes,) - indica a quin graf pertany cada node
        """
        
        if self.use_edge_attr and weight_edges is not None:
            if weight_edges.dim() == 1:
                weight_edges = weight_edges.unsqueeze(-1)
            
            # Primera capa GAT i pooling
            x = F.relu(self.gat1(nodes, edges, weight_edges))
            x, edges, weight_edges, batch_idx, _, _ = self.pool1(x, edges, weight_edges, batch_idx)
            
            # Segona capa GAT i pooling
            x = F.relu(self.gat2(x, edges, weight_edges))
            x, edges, weight_edges, batch_idx, _, _ = self.pool2(x, edges, weight_edges, batch_idx)
        else:
            # Primera capa GAT i pooling sense atributs d'arestes
            x = F.relu(self.gat1(nodes, edges))
            x, edges, _, batch_idx, _, _ = self.pool1(x, edges, None, batch_idx)
            
            # Segona capa GAT i pooling sense atributs d'arestes
            x = F.relu(self.gat2(x, edges))
            x, edges, _, batch_idx, _, _ = self.pool2(x, edges, None, batch_idx)

        # Agregació global (Max Pooling) i classificació
        g = global_mean_pool(x, batch_idx)
        g = F.dropout(g, p=self.dropout, training=self.training)
        return self.classifier(g)


# MODEL Graph Convolutional Network (GCN)
class GCNWithAgg(torch.nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, use_edge_weight=True, dropout=0.0):
        super().__init__()
        """ 
                Consulteu el help de GCNConv per entendre els paràmetres d'entrada.
        """
        self.use_edge_weight = use_edge_weight
        self.dropout = dropout
        self.gcn1 = GCNConv(in_ch, hidden_ch)
        self.gcn2 = GCNConv(hidden_ch, hidden_ch)
        self.lin = torch.nn.Sequential(
            torch.nn.Linear(hidden_ch, hidden_ch),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
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


# MODEL GCN amb Max Pooling jeràrquic
class GCNWithAggMaxPool(torch.nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, use_edge_weight=True, pool_ratio=0.5, dropout=0.0):
        super().__init__()
        """ 
        GCN amb dues capes de convolució i dues capes de Max Pooling (TopKPooling) per reduir el nombre de nodes.
        """
        self.use_edge_weight = use_edge_weight
        self.dropout = dropout
        self.gcn1 = GCNConv(in_ch, hidden_ch)
        self.pool1 = TopKPooling(hidden_ch, ratio=pool_ratio)
        self.gcn2 = GCNConv(hidden_ch, hidden_ch)
        self.pool2 = TopKPooling(hidden_ch, ratio=pool_ratio)
        
        self.lin = torch.nn.Sequential(
            torch.nn.Linear(hidden_ch, hidden_ch),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.LayerNorm(hidden_ch),
            torch.nn.Linear(hidden_ch, out_ch)
        )

    def forward(self, nodes, edges, weight_edges, batch_idx):
        """
            nodes:           (N_total_nodes, in_ch)
            edges: (2, TotalEdgesBatch) - És el graf expressat per les arestes. Parelles de nodes.
            weight_edges: (TotalEdgesBatch,) - pesos de cada aresta
            batch_idx:   (N_total_nodes,) - indica a quin graf pertany cada node
        """
        
        # Primera capa convolucional i pooling
        if self.use_edge_weight:
            x = F.relu(self.gcn1(nodes, edges, edge_weight=weight_edges))
            x, edges, weight_edges, batch_idx, _, _ = self.pool1(x, edges, weight_edges, batch_idx)
        else:
            x = F.relu(self.gcn1(nodes, edges))
            x, edges, _, batch_idx, _, _ = self.pool1(x, edges, None, batch_idx)
            
        # Segona capa convolucional i pooling
        if self.use_edge_weight:
            x = F.relu(self.gcn2(x, edges, edge_weight=weight_edges))
            x, edges, weight_edges, batch_idx, _, _ = self.pool2(x, edges, weight_edges, batch_idx)
        else:
            x = F.relu(self.gcn2(x, edges))
            x, edges, _, batch_idx, _, _ = self.pool2(x, edges, None, batch_idx)

        # Agregació global (Max Pooling) i classificació
        g = global_max_pool(x, batch_idx)
        return self.lin(g)


# Altres models de l'estat de l'art??