"""
DecentraPharma Autoresearch GNN training script.
Task: HIV Inhibition classification (Antiviral Research).
Metric: ROC-AUC (Higher is better)
Model: Graph Convolutional Network (GCN) with global pooling.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
## START OF AGENT IMPORTS ##
from torch_geometric.nn import (
    GINEConv, GCNConv, GATConv, GATv2Conv,
    global_mean_pool, global_max_pool, global_add_pool, 
    Set2Set, GlobalAttention
)
import torch.nn.functional as F
import torch.nn as nn
import torch
## END OF AGENT IMPORTS ##
from torch_geometric.loader import DataLoader
from prepare_graph import prepare_graph_data, evaluate_graph_metric, NUM_ATOM_FEATURES, NUM_BOND_FEATURES

import warnings
warnings.filterwarnings("ignore", message=".*scatter.*")

# ---------------------------------------------------------------------------
# GNN Model
# ---------------------------------------------------------------------------

## START OF AGENT MODIFIABLE SECTION ##

class AntiviralGNN(nn.Module):
    """
    GINE-256 with Set2Set Pooling (LSTM Readout).
    
    Rationale:
    1. GINEConv: Explicitly models edge features (bonds) within the aggregation, crucial for chemistry.
    2. Deep Residual Architecture: 5 layers with residuals allows learning complex substructures.
    3. Set2Set Pooling: Replaces static mean/max pooling with an LSTM-based mechanism 
       that processes the set of atoms to produce a graph embedding. This is often SOTA for molecules.
    4. Regularization: Higher dropout (0.4) to combat overfitting on the imbalanced HIV dataset.
    """
    def __init__(self, num_node_features=NUM_ATOM_FEATURES, num_edge_features=NUM_BOND_FEATURES):
        super(AntiviralGNN, self).__init__()

        hidden_dim = 256
        num_layers = 5
        dropout_rate = 0.4
        
        self.dropout_rate = dropout_rate
        
        # Encoders for atoms and bonds
        self.atom_encoder = nn.Linear(num_node_features, hidden_dim)
        self.bond_encoder = nn.Linear(num_edge_features, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for _ in range(num_layers):
            # MLP for GINE: Hidden -> 2*Hidden -> Hidden
            # The MLP is applied after aggregating neighbors + edge features
            nn_blk = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate), 
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            # train_eps=True lets the model learn to distinguish center node vs neighbors
            self.convs.append(GINEConv(nn_blk, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Set2Set Pooling: Maps node set to graph embedding
        # processing_steps=3 is standard for molecules
        # Output size is 2 * hidden_dim
        self.pooling = Set2Set(hidden_dim, processing_steps=3, num_layers=1)

        # Classifier head
        # Input dim is 2 * hidden_dim because Set2Set concatenates q_t and r_t (LSTM states)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index, batch, edge_attr=None, return_attention_weights=False):
        # 1. Encoding
        x = self.atom_encoder(x)
        if edge_attr is not None:
            edge_attr = self.bond_encoder(edge_attr)
        
        # 2. Message Passing
        for i, conv in enumerate(self.convs):
            x_in = x
            
            # GINE update
            # Note: GINEConv expects edge_attr to be same dim as x (handled by encoders above)
            x = conv(x, edge_index, edge_attr=edge_attr)
            
            # Batch Norm + Act + Dropout
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
            # Residual Connection
            x = x + x_in

        # 3. Pooling (Set2Set)
        x_graph = self.pooling(x, batch)
        
        # 4. Classification
        out = self.classifier(x_graph)
        
        if return_attention_weights:
            return out.squeeze(-1), []
            
        return out.squeeze(-1)

# Hyperparameters
LEARNING_RATE = 0.0005  # Reduced LR for stability with Set2Set
BATCH_SIZE = 128        # Smaller batch size to improve generalization
WEIGHT_DECAY = 1e-4     # L2 Regularization
SCHEDULER_PATIENCE = 5
TIME_BUDGET = 900
DESCRIPTION = "GINE-256 (5 layers) + Set2Set + Dropout 0.4"

## END OF AGENT MODIFIABLE SECTION ##

# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train():
    print("=" * 60)
    print("DecentraPharma Autoresearch - GNN Antiviral Research")
    print("=" * 60)

    # Load data
    train_graphs, val_graphs = prepare_graph_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # DataLoaders
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE)

    # Model
    model = AntiviralGNN().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=SCHEDULER_PATIENCE, factor=0.5
    )

    # Loss with class imbalance handling
    labels = [g.y.item() for g in train_graphs]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Class imbalance: {int(n_pos)} active / {int(n_neg)} inactive, pos_weight={pos_weight.item():.2f}")

    # Training
    start_time = time.time()
    best_auc = 0.0
    epoch = 0

    print(f"\nTime budget: {TIME_BUDGET}s")
    print("-" * 60)

    while True:
        elapsed = time.time() - start_time
        if elapsed >= TIME_BUDGET:
            print("Time budget exceeded.")
            break

        epoch += 1
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            # Check time budget within epoch
            if time.time() - start_time >= TIME_BUDGET:
                break

            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
            loss = criterion(out, batch.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # Evaluate
        val_auc = evaluate_graph_metric(model, val_loader, device)
        scheduler.step(val_auc)

        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]['lr']
        improved = " *NEW BEST*" if val_auc > best_auc else ""
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f} | "
              f"LR: {lr:.2e} | Time: {elapsed:.0f}s{improved}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'best_gnn_model.pt')

    # Final summary
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"val_auc:          {best_auc:.6f}")
    print(f"training_seconds: {total_time:.1f}")
    print(f"num_epochs:       {epoch}")
    print(f"num_params:       {num_params}")

    # Log to results
    from datetime import datetime
    results_path = "results_gnn.tsv"
    file_exists = os.path.exists(results_path)
    with open(results_path, "a") as f:
        if not file_exists or os.path.getsize(results_path) == 0:
            f.write("timestamp\tval_auc\tnum_params\tdescription\n")
        f.write(f"{datetime.now().isoformat()}\t{best_auc:.6f}\t{num_params}\t{DESCRIPTION}\n")

    return best_auc


if __name__ == "__main__":
    train()
