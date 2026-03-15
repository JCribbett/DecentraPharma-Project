"""
Graph-based data preparation for DecentraPharma Autoresearch.
Converts SMILES strings into PyTorch Geometric Data objects.
Task: HIV Inhibition classification (Antiviral Research).
"""

import os
import requests
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split

# Constants
DATA_DIR = 'data'
DATASET_URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv'
DATASET_PATH = os.path.join(DATA_DIR, 'HIV.csv')
GRAPH_CACHE_PATH = os.path.join(DATA_DIR, 'graph_data.pt')
TIME_BUDGET = 300  # 5 minutes
NUM_ATOM_FEATURES = 16  # 9 element + 7 properties
NUM_BOND_FEATURES = 6  # bond type (4) + conjugated + in_ring
FINGERPRINT_SIZE = 2048

# ---------------------------------------------------------------------------
# Atom featurization
# ---------------------------------------------------------------------------

def atom_features(atom):
    """Extract a feature vector for a single atom."""
    # One-hot encode common elements
    ELEMENTS = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
    element = atom.GetSymbol()
    element_enc = [1 if element == e else 0 for e in ELEMENTS]

    features = [
        *element_enc,                          # 9 features: element type
        atom.GetDegree() / 6.0,                # normalized degree
        atom.GetFormalCharge() / 3.0,          # normalized formal charge
        atom.GetNumRadicalElectrons() / 2.0,   # normalized radical electrons
        int(atom.GetIsAromatic()),              # aromaticity
        atom.GetTotalNumHs() / 4.0,            # normalized H count
        atom.GetNumExplicitHs() / 4.0,         # normalized explicit H
        int(atom.IsInRing()),                   # in ring
    ]
    return features


def mol_to_graph(smiles, label):
    """Convert a SMILES string to a PyTorch Geometric Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    node_feats = []
    for atom in mol.GetAtoms():
        node_feats.append(atom_features(atom))
    x = torch.tensor(node_feats, dtype=torch.float)

    # Edge index and features (bonds -> edges, bidirectional)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Bond features
        bt = bond.GetBondType()
        fbond = [
            1 if bt == Chem.rdchem.BondType.SINGLE else 0,
            1 if bt == Chem.rdchem.BondType.DOUBLE else 0,
            1 if bt == Chem.rdchem.BondType.TRIPLE else 0,
            1 if bt == Chem.rdchem.BondType.AROMATIC else 0,
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
        ]
        
        edge_index.append([i, j])
        edge_attr.append(fbond)
        edge_index.append([j, i])
        edge_attr.append(fbond)

    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, NUM_BOND_FEATURES), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Global fingerprint (Morgan 2048-bit)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FINGERPRINT_SIZE)
    fp_tensor = torch.tensor(list(fp), dtype=torch.float).unsqueeze(0) # [1, 2048]

    y = torch.tensor([label], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, fp=fp_tensor)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def download_dataset():
    """Download HIV dataset if not present."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(DATASET_PATH):
        print(f"Downloading dataset from {DATASET_URL}...")
        response = requests.get(DATASET_URL)
        with open(DATASET_PATH, 'wb') as f:
            f.write(response.content)
        print("Download complete.")


def prepare_graph_data():
    """
    Prepare graph data for training.
    Returns: (train_graphs, val_graphs) as lists of Data objects.
    """
    # Check cache
    # IMPORTANT: We force rebuild if we suspect old cache lacks fingerprints
    # or just rely on user to delete it. For now, let's assume we delete it manually or use a new name.
    # Let's try to load, and if 'fp' is missing in the first element, we rebuild.
    
    if os.path.exists(GRAPH_CACHE_PATH):
        print("Loading cached graph data...")
        try:
            cached = torch.load(GRAPH_CACHE_PATH, weights_only=False)
            if 'train' in cached and len(cached['train']) > 0:
                if not hasattr(cached['train'][0], 'fp'):
                    print("Cache outdated (missing fingerprints). Rebuilding...")
                else:
                    print(f"Loaded: Train={len(cached['train'])}, Val={len(cached['val'])}")
                    return cached['train'], cached['val']
        except Exception as e:
            print(f"Error loading cache: {e}. Rebuilding...")

    # Build from scratch
    print("Building graph data from SMILES (this may take a few minutes)...")
    download_dataset()
    df = pd.read_csv(DATASET_PATH)

    graphs = []
    skipped = 0
    for idx, row in df.iterrows():
        g = mol_to_graph(row['smiles'], row['HIV_active'])
        if g is not None:
            graphs.append(g)
        else:
            skipped += 1
        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1}/{len(df)} molecules...")

    print(f"Total graphs: {len(graphs)}, skipped: {skipped}")

    # Class balance
    labels = [g.y.item() for g in graphs]
    n_active = sum(labels)
    n_inactive = len(labels) - n_active
    print(f"Class balance: {int(n_active)} active ({100*n_active/len(labels):.1f}%), "
          f"{int(n_inactive)} inactive ({100*n_inactive/len(labels):.1f}%)")

    # Split
    train_graphs, val_graphs = train_test_split(
        graphs, test_size=0.2, random_state=42,
        stratify=[g.y.item() for g in graphs]
    )

    # Cache
    print("Caching graph data...")
    torch.save({'train': train_graphs, 'val': val_graphs}, GRAPH_CACHE_PATH)
    print(f"Cached to {GRAPH_CACHE_PATH}")

    return train_graphs, val_graphs


def evaluate_graph_metric(model, val_loader, device):
    """Evaluate ROC-AUC on validation set."""
    from sklearn.metrics import roc_auc_score

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            # Pass edge_attr if available and model accepts it
            edge_attr = getattr(batch, 'edge_attr', None)
            fp = getattr(batch, 'fp', None)
            
            # Handle fingerprint batching
            # In PyG, batch.fp will be stacked [batch_size, 1, 2048] -> [batch_size, 2048]
            if fp is not None:
                fp = fp.view(-1, 2048)

            try:
                out = model(batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr, fp=fp)
            except TypeError:
                 # Fallback for models that don't support fp yet
                try:
                    out = model(batch.x, batch.edge_index, batch.batch)
                except:
                     out = model(batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr)
                     
            probs = torch.sigmoid(out).cpu().numpy()
            all_preds.extend(probs.flatten())
            all_labels.extend(batch.y.cpu().numpy().flatten())

    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5
    return auc


if __name__ == '__main__':
    train_graphs, val_graphs = prepare_graph_data()
    print(f"\nGraph data ready.")
    print(f"  Train: {len(train_graphs)} graphs")
    print(f"  Val:   {len(val_graphs)} graphs")
    print(f"  Atom features per node: {NUM_ATOM_FEATURES}")
    print(f"  Example graph: {train_graphs[0]}")
