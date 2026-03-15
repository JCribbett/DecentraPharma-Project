import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import os
import time

# Assume data_loader.py and model_hiv.py are in the same directory or accessible
from data_loader import prepare_graph_data # Assuming this loads HIV data
from model_hiv import AntiviralGNN
from training_utils import train_epoch, evaluate, save_checkpoint, load_checkpoint, get_best_score, save_results

# Configuration Parameters (these will be tuned by the optimizer)
HIDDEN_DIM = 64
DROPOUT = 0.2
LEARNING_RATE = 3e-4
BATCH_SIZE = 109
WEIGHT_DECAY = 1e-5
SCHEDULER_PATIENCE = 7
TIME_BUDGET = 1325 # seconds
DESCRIPTION = "HIV GNN Baseline"

# --- File Paths ---
RESULTS_FILE = "results_hiv.tsv"
CHECKPOINT_FILE = "hiv_model_checkpoint.pth.tar"
MODEL_BACKUP = "hiv_model_backup.pth" # For saving the best model state dict

# --- Constants ---
NUM_FEATURES = 16 # Assuming this is the number of atom features from prepare_graph_data
NUM_CLASSES_HIV = 1 # Binary classification for HIV activity

def main():
    # --- Load Data ---
    print("Loading HIV data...")
    # prepare_graph_data should return train_loader, val_loader, test_loader (all PyG DataLoaders)
    train_loader, val_loader, _ = prepare_graph_data() # Ignoring test_loader for now

    # --- Initialize Model ---
    print("Initializing HIV GNN model...")
    model = AntiviralGNN(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, num_classes=NUM_CLASSES_HIV)
    
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
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0)

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
