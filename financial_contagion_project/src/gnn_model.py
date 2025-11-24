import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATConv,
    BatchNorm
)


# ============================================================
# ⭐ 1. GCN MODEL (with BatchNorm + Dropout)
# ============================================================
class GCNNet(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=1):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)

        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=0.4, training=self.training)

        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=0.4, training=self.training)

        x = self.lin(x)
        return x.squeeze()


# ============================================================
# ⭐ 2. GRAPH SAGE (best for banking networks)
# ============================================================
class GraphSAGENet(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=1):
        super().__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)

        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)

        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=0.4, training=self.training)

        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=0.4, training=self.training)

        x = self.lin(x)
        return x.squeeze()


# ============================================================
# ⭐ 3. GAT MODEL (multi-head attention)
# ============================================================
class GATNet(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, out_channels=1, heads=4):
        super().__init__()

        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.4)
        self.bn1 = BatchNorm(hidden_channels * heads)

        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=0.4)
        self.bn2 = BatchNorm(hidden_channels)

        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=0.4, training=self.training)

        x = F.elu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=0.4, training=self.training)

        x = self.lin(x)
        return x.squeeze()
