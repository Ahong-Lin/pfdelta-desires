"""
Graph Attention Network (GAT) for Power Flow prediction.

Based on the PFNet data interface:
  - Input:  data["bus"].x (N, 16) = [one-hot(4), features(6), mask(6)]
  - Edge:   data["bus", "branch", "bus"].edge_index + edge_attr (E, 5)
  - Output: (N, 6) = [vm, va, pg_pd, qg_qd, gs, bs]

Uses standard GAT propagation rule (Velickovic et al., 2018):
    alpha_ij = softmax_{j in N(i)} LeakyReLU(a^T [W*h_i || W*h_j])
    h_i' = sigma(sum_{j in N(i)} alpha_ij * W*h_j)

Edge features are incorporated via edge-aware attention bias and
a separate edge embedding that modulates node features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, HeteroData

from core.utils.registry import registry


@registry.register_model("gat_pf")
class GAT_PF(nn.Module):
    """
    Graph Attention Network for Power Flow prediction.

    Architecture:
      1. Mask embedding (same as PFNet)
      2. Edge feature embedding MLP
      3. N layers of GATConv with multi-head attention + ReLU + Dropout
      4. Output MLP: hidden_dim -> output_dim

    Parameters
    ----------
    nfeature_dim : int
        Dimensionality of node input features (default: 6).
    efeature_dim : int
        Dimensionality of edge features (default: 5).
    output_dim : int
        Dimensionality of output predictions (default: 6).
    hidden_dim : int
        Hidden layer dimensionality.
    n_gnn_layers : int
        Number of GAT layers.
    n_heads : int
        Number of attention heads per GAT layer.
    dropout_rate : float
        Dropout rate for regularization.
    """

    def __init__(self, nfeature_dim=6, efeature_dim=5, output_dim=6,
                 hidden_dim=256, n_gnn_layers=5, n_heads=4,
                 dropout_rate=0.2, **kwargs):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate

        # Mask embedding (same as PFNet)
        self.mask_embd = nn.Sequential(
            nn.Linear(nfeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nfeature_dim),
        )

        # Input projection: nfeature_dim -> hidden_dim
        self.input_proj = nn.Linear(nfeature_dim, hidden_dim)

        # Edge feature embedding: efeature_dim -> hidden_dim
        self.edge_embd = nn.Sequential(
            nn.Linear(efeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # GAT layers with multi-head attention
        self.gat_layers = nn.ModuleList()
        for _ in range(n_gnn_layers):
            self.gat_layers.append(
                GATConv(hidden_dim, hidden_dim // n_heads,
                        heads=n_heads, dropout=dropout_rate,
                        edge_dim=hidden_dim, fill_value='mean')
            )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def data_gatherer(self, data):
        """Extract features from HeteroData or Data (same as PFNet)."""
        if isinstance(data, HeteroData):
            x = data["bus"].x[:, 4: 4 + self.nfeature_dim]
            mask = data["bus"].x[:, -self.nfeature_dim:]
            edge_index = data["bus", "branch", "bus"].edge_index
            edge_features = data["bus", "branch", "bus"].edge_attr
        elif isinstance(data, Data):
            assert data.x.shape[-1] == self.nfeature_dim * 2 + 4
            x = data.x[:, 4: 4 + self.nfeature_dim]
            mask = data.x[:, -self.nfeature_dim:]
            edge_index = data.edge_index
            edge_features = data.edge_attr
        return x, mask, edge_index, edge_features

    def undirect_graph(self, edge_index, edge_attr):
        """Convert directed graph to undirected by duplicating edges."""
        if edge_index.shape[1] == 0:
            return edge_index, edge_attr
        src0, dst0 = edge_index[0, 0], edge_index[1, 0]
        has_reverse = (edge_index[0] == dst0) & (edge_index[1] == src0)
        if has_reverse.any():
            return edge_index, edge_attr
        edge_index_rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
        edge_index = torch.cat([edge_index, edge_index_rev], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        return edge_index, edge_attr

    def forward(self, data):
        """
        Forward pass.

        Parameters
        ----------
        data : HeteroData or Data
            Input graph data.

        Returns
        -------
        x : Tensor (N, output_dim)
            Predicted power flow quantities.
        """
        x, mask, edge_index, edge_features = self.data_gatherer(data)

        # Mask embedding + input features
        x = self.mask_embd(mask) + x

        # Make graph undirected
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)

        # Project to hidden dim
        x = self.input_proj(x)

        # Embed edge features
        edge_embd = self.edge_embd(edge_features)  # (E, hidden_dim)

        # GAT message passing layers
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index, edge_attr=edge_embd)
            x = F.relu(x)
            x = self.dropout(x)

        # Output projection
        x = self.output_proj(x)
        return x
