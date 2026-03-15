import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import os
import time

## START OF AGENT IMPORTS ##
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
import torch.nn.functional as F
from torch.nn import BatchNorm1d
## END OF AGENT IMPORTS ##

# Assume data_loader.py and model_tox.py are in the same directory or accessible
from data_loader import prepare_tox_data # Assuming this loads Tox21 data
from training_utils import train_epoch, evaluate, save_checkpoint, load_checkpoint, get_best_score, save_results

## START OF AGENT MODIFIABLE SECTION ##
# Configuration Parameters (these will be tuned by the optimizer)
HIDDEN_DIM = 64
DROPOUT = 0.1
LEARNING_RATE = 5e-4
BATCH_SIZE = 128
WEIGHT_DECAY = 1e-4
SCHEDULER_PATIENCE = 5
TIME_BUDGET = 1325 # seconds
DESCRIPTION = "GATv2 3-Layer with Multi-Head Attention"

class AntiviralGNN(nn.Module):
    def __init__(self, num_features=16, hidden_dim=64, dropout=0.1, num_classes=1):
        super(AntiviralGNN, self).__init__()
        self.dropout_ratio = dropout
        
        # GATv2 with multi-head attention to capture partial structures (toxicophores)
        heads = 4
        
        # Layer 1: Input -> Hidden (Expanded by heads)
        self.conv1 = GATv2Conv(num_features, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.bn1 = BatchNorm1d(hidden_dim * heads)
        
        # Layer 2: Hidden -> Hidden
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.bn2 = BatchNorm1d(hidden_dim * heads)
        
        # Layer 3: Hidden -> Hidden (Collapsed to single head for pooling)
        self.conv3 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, concat=False)
        self.bn3 = BatchNorm1d(hidden_dim)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # *2 for pooled_res (mean + max)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x) 
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        
        # Layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        # No strong activation before pooling, allow negative values to be handled by max_pool or mixed
        
        # Global Pooling (Readout)
        # Combining Mean and Max pooling captures both the "average" property and "extreme" features (like a single highly toxic group)
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        
        pooled_x = torch.cat([mean_pool, max_pool], dim=1)

        out = self.classifier(pooled_x)
        return out
## END OF AGENT MODIFIABLE SECTION ##

# --- File Paths ---
RESULTS_FILE = "results_tox.tsv"
CHECKPOINT_FILE = "tox_model_checkpoint.pth.tar"
MODEL_BACKUP = "tox_model_backup.pth" # For saving the best model state dict

# --- Constants ---
NUM_FEATURES = 16 # Assuming this is the number of atom features from prepare_tox_data
NUM_CLASSES_TOX = 1 # Binary classification for Toxicity

def main():
    # --- Load Data ---
    print("Loading Tox21 data...")
    # prepare_tox_data should return train_loader, val_loader, test_loader (all PyG DataLoaders)
    train_loader, val_loader, _ = prepare_tox_data() # Ignoring test_loader for now

    # --- Initialize Model ---
    print("Initializing Tox GNN model...")
    model = AntiviralGNN()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {num_params} trainable parameters.")

    # --- Optimization Setup ---
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss() # Suitable for binary classification

    # --- Load Checkpoint (if exists) ---
    checkpoint = load_checkpoint(CHECKPOINT_FILE)
    start_epoch = 0
    if checkpoint:
        try:
            model.load_state_dict(checkpoint['state_dict'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint.get('epoch', 0)
        except RuntimeError as e:
            print(f"⚠️ Architecture mismatch detected. Ignoring old checkpoint and starting fresh.")
            start_epoch = 0

    # --- Training Loop ---
    print("Starting training loop...")
    best_val_auc = get_best_score(RESULTS_FILE)
    
    start_time = time.time()

    for epoch in range(start_epoch, 1000): # Train for a large number of epochs, rely on TIME_BUDGET or early stopping
        avg_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_auc = evaluate(model, val_loader)

        print(f"Epoch {epoch+1}/{1000} - Loss: {avg_loss:.4f}, Val AUC: {val_auc:.4f}")

        # Save results and checkpoint
        save_results(RESULTS_FILE, DESCRIPTION, val_auc, num_params)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_auc': best_val_auc,
        }, CHECKPOINT_FILE)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            # Save the best model state dict separately if needed
            torch.save(model.state_dict(), MODEL_BACKUP)
            print(f"New best validation AUC: {best_val_auc:.4f}")

        # Time budget check
        elapsed_time = time.time() - start_time
        if elapsed_time > TIME_BUDGET:
             print(f"Approaching time budget. Stopping training.")
             break

    print(f"Training finished. Best Val AUC: {best_val_auc:.4f}")

if __name__ == "__main__":
    main()
