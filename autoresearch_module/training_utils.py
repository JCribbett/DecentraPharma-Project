import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import os
import pandas as pd
import numpy as np

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        # Ensure target is on the same device as the model output
        loss = criterion(out, data.y.view(-1, 1).to(out.device).float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in loader:
            out = model(data)
            y_true.append(data.y.view(-1, 1).cpu().numpy())
            y_pred.append(torch.sigmoid(out).cpu().numpy())
            
    y_true_all = np.concatenate(y_true)
    y_pred_all = np.concatenate(y_pred)
    return roc_auc_score(y_true_all, y_pred_all)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def load_checkpoint(filename='checkpoint.pth.tar'):
    if os.path.isfile(filename):
        print(f"=> loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        return checkpoint
    else:
        print(f"=> no checkpoint found at '{filename}'")
        return None

def get_best_score(results_file):
    if not os.path.exists(results_file):
        return -1.0
    try:
        df = pd.read_csv(results_file, sep='\t')
        if not df.empty and 'val_auc' in df.columns:
            # Return the maximum AUC from the file, which is more robust
            return df['val_auc'].max()
        else:
            return -1.0
    except Exception as e:
        print(f"Error reading results file {results_file}: {e}")
        return -1.0

def save_results(results_file, description, val_auc, num_params):
    # Check if results file exists, create if not
    if not os.path.exists(results_file):
        with open(results_file, 'w') as f:
            f.write("timestamp\tdescription\tval_auc\tnum_params\n")

    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(results_file, 'a') as f:
        f.write(f"{timestamp}\t{description}\t{val_auc:.6f}\t{num_params}\n")
