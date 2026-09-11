
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool


class GraphEncoder(nn.Module):


    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        encoder_type: str = "graphsage",
        dropout: float = 0.2,
        gat_heads: int = 4,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if encoder_type not in ("graphsage", "gat"):
            raise ValueError(f"Unknown encoder_type: {encoder_type!r}. Expected 'graphsage' or 'gat'.")

        self.encoder_type = encoder_type
        self.dropout = dropout
        self.hidden_dim = hidden_dim  

        self.convs = nn.ModuleList()
        dims_in = [in_dim] + [hidden_dim] * (num_layers - 1)

        for layer_in_dim in dims_in:
            if encoder_type == "graphsage":
                self.convs.append(SAGEConv(layer_in_dim, hidden_dim))
            else:  
                self.convs.append(
                    GATConv(layer_in_dim, hidden_dim, heads=gat_heads, concat=False, dropout=dropout)
                )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):
       
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:  
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        node_embeddings = h
        graph_embedding = global_mean_pool(node_embeddings, batch)
        return node_embeddings, graph_embedding


class GraphTagHead(nn.Module):

    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        return self.linear(graph_embedding)
