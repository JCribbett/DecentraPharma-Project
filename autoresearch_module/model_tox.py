import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

# Assuming atom_features and mol_to_graph are available from prepare_graph.py or similar utilities
# For simplicity, these are not included here, but would be needed if this script were run standalone.

# Placeholder for atom_features and mol_to_graph if needed for standalone execution
# In the context of the full system, these would be imported from data_loader.py


class AntiviralGNN_Tox(nn.Module):
    def __init__(self, num_features, hidden_dim, dropout, num_classes=1):
        super(AntiviralGNN_Tox, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # GNN Layers (same architecture as HIV model for now)
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
        self.conv3 = GCNConv(hidden_dim * 2, hidden_dim * 4)
        self.conv4 = GCNConv(hidden_dim * 4, hidden_dim * 8)

        # Classifier Head for Toxicity
        self.classifier = nn.Linear(hidden_dim * 8 * 2, num_classes) # *2 for mean and max pooling

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = torch.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = torch.relu(self.conv3(x, edge_index))
        x = self.dropout(x)
        x = torch.relu(self.conv4(x, edge_index))
        x = self.dropout(x)

        # Global pooling (mean and max)
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_mean_pool(x, batch) # Placeholder for global_max_pool

        # Concatenate pooled features
        pooled_x = torch.cat([mean_pool, max_pool], dim=1)

        # Classification
        return self.classifier(pooled_x)
