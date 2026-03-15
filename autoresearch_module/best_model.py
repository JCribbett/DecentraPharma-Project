import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool

NUM_ATOM_FEATURES = 16
NUM_BOND_FEATURES = 6

class AntiviralGNN(nn.Module):
    def __init__(self, num_node_features=NUM_ATOM_FEATURES, num_edge_features=NUM_BOND_FEATURES):
        super(AntiviralGNN, self).__init__()

        hidden_dim = 192
        heads = 4
        head_dim = hidden_dim // heads

        self.atom_encoder = nn.Linear(num_node_features, hidden_dim)

        # 4 Layers of GATv2Conv
        self.conv1 = GATv2Conv(hidden_dim, head_dim, heads=heads, edge_dim=num_edge_features)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = GATv2Conv(hidden_dim, head_dim, heads=heads, edge_dim=num_edge_features)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.conv3 = GATv2Conv(hidden_dim, head_dim, heads=heads, edge_dim=num_edge_features)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.conv4 = GATv2Conv(hidden_dim, head_dim, heads=heads, edge_dim=num_edge_features)
        self.bn4 = nn.BatchNorm1d(hidden_dim)

        # Classifier head (input is 384, which is 192 * 2 for mean + max pooling)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index, batch, edge_attr=None, return_attention_weights=False):
        x = self.atom_encoder(x)
        
        attn_weights = []
        
        def apply_conv(conv, x_in):
            if return_attention_weights:
                out, (e_idx, alpha) = conv(x_in, edge_index, edge_attr=edge_attr, return_attention_weights=True)
                attn_weights.append((e_idx, alpha))
                return out
            return conv(x_in, edge_index, edge_attr=edge_attr)

        # Residual connections
        x1 = F.elu(self.bn1(apply_conv(self.conv1, x)))
        x2 = F.elu(self.bn2(apply_conv(self.conv2, x1))) + x1
        x3 = F.elu(self.bn3(apply_conv(self.conv3, x2))) + x2
        x4 = F.elu(self.bn4(apply_conv(self.conv4, x3))) + x3

        # Jumping Knowledge (sum of all representations)
        x_jk = x1 + x2 + x3 + x4

        # Pooling
        x_mean = global_mean_pool(x_jk, batch)
        x_max = global_max_pool(x_jk, batch)
        x_pool = torch.cat([x_mean, x_max], dim=1)

        out = self.classifier(x_pool)

        if return_attention_weights:
            return out.squeeze(-1), attn_weights
        return out.squeeze(-1)