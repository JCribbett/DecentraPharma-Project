import os
import requests
import pandas as pd
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Constants
DATA_DIR = 'data'
DATASET_URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bbbp.csv'
DATASET_PATH = os.path.join(DATA_DIR, 'bbbp.csv')
TIME_BUDGET = 300 # 5 minutes
FINGERPRINT_SIZE = 2048

def download_dataset():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(DATASET_PATH):
        print(f"Downloading dataset from {DATASET_URL}...")
        response = requests.get(DATASET_URL)
        with open(DATASET_PATH, 'wb') as f:
            f.write(response.content)

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FINGERPRINT_SIZE))

def prepare_data():
    download_dataset()
    df = pd.read_csv(DATASET_PATH)
    
    # BBBP dataset has 'smiles' and 'p_np' (target) columns
    fps = []
    labels = []
    
    for _, row in df.iterrows():
        fp = smiles_to_fp(row['smiles'])
        if fp is not None:
            fps.append(fp)
            labels.append(row['p_np'])
            
    X = np.array(fps)
    y = np.array(labels)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return (torch.tensor(X_train, dtype=torch.float32), 
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(X_val, dtype=torch.float32), 
            torch.tensor(y_val, dtype=torch.float32))

def evaluate_metric(model, X_val, y_val):
    model.eval()
    with torch.no_grad():
        outputs = model(X_val).squeeze()
        # For AUC, we need probabilities or scores
        # We assume the model outputs logits or sigmoid probabilities
        probs = torch.sigmoid(outputs).cpu().numpy()
        try:
            auc = roc_auc_score(y_val.cpu().numpy(), probs)
        except ValueError:
            auc = 0.5 # Default for single-class or error
    return auc

if __name__ == '__main__':
    X_train, y_train, X_val, y_val = prepare_data()
    print(f"Data prepared. Train size: {len(X_train)}, Val size: {len(X_val)}")
    print(f"Fingerprint size: {FINGERPRINT_SIZE}")
