"""
DecentraPharma Autoresearch training script.
Adapted from karpathy/autoresearch for Molecular Property Prediction.
Task: HIV Inhibition classification (Antiviral Research).
Metric: ROC-AUC (Higher is better)
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from prepare import prepare_data, evaluate_metric, TIME_BUDGET, FINGERPRINT_SIZE

# ---------------------------------------------------------------------------
# Molecular Model (MLP)
# ---------------------------------------------------------------------------

## START OF AGENT MODIFIABLE SECTION ##

class MolecularModel(nn.Module):
    def __init__(self, input_size):
        super(MolecularModel, self).__init__()
        # Simple MLP architecture for the agent to optimize
        # Note: HIV dataset is imbalanced (~3.5% active)
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1) # Output logit for binary classification
        )

    def forward(self, x):
        return self.net(x)

# Hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-4

## END OF AGENT MODIFIABLE SECTION ##

# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train():
    print("Starting DecentraPharma Autoresearch (Antiviral Research)...")
    X_train, y_train, X_val, y_val = prepare_data()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    
    model = MolecularModel(FINGERPRINT_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Calculate positive weight for imbalanced classes
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = (n_neg / n_pos).to(device)
    print(f"Using pos_weight: {pos_weight:.2f} for imbalanced classes")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    start_time = time.time()
    step = 0
    
    print(f"Time budget: {TIME_BUDGET}s")
    
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time >= TIME_BUDGET:
            break
            
        # Manual batching
        indices = torch.randperm(X_train.size(0))[:BATCH_SIZE]
        batch_X, batch_y = X_train[indices], y_train[indices]
        
        model.train()
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f} | Elapsed: {elapsed_time:.1f}s")
            
        step += 1

    # Final Evaluation
    val_auc = evaluate_metric(model, X_val, y_val)
    
    print("---")
    print(f"val_auc:          {val_auc:.6f}")
    print(f"training_seconds: {time.time() - start_time:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params:       {sum(p.numel() for p in model.parameters())}")

if __name__ == "__main__":
    train()
