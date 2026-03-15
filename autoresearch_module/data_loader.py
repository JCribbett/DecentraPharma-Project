import os
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph
from rdkit import Chem
from rdkit.Chem import AllChem
import gzip
import requests
import pickle

# --- Configuration ---
HIV_DATA_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv"
TOX_DATA_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"

HIV_PROCESSED_DATA_PATH = "hiv_data.pt"
TOX_PROCESSED_DATA_PATH = "tox_data.pt"

# Assuming default number of atom features based on common practice
# This should ideally match what prepare_graph.py generated if it was different
DEFAULT_NUM_ATOM_FEATURES = 16 

# --- Atom Feature Engineering ---
def atom_features(atom):
    """
    Extract a 16-dimensional feature vector for a single atom.
    This is consistent with prepare_graph.py to fix feature dimension mismatch.
    """
    # One-hot encode common elements
    ELEMENTS = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
    element = atom.GetSymbol()
    element_enc = [1 if element == e else 0 for e in ELEMENTS]

    features = [
        *element_enc,                          # 9 features: element type
        atom.GetDegree() / 6.0,                # normalized degree
        atom.GetFormalCharge() / 3.0,          # normalized formal charge
        atom.GetNumRadicalElectrons() / 2.0,   # normalized radical electrons
        float(atom.GetIsAromatic()),           # aromaticity
        atom.GetTotalNumHs() / 4.0,            # normalized H count
        atom.GetNumExplicitHs() / 4.0,         # normalized explicit H
        float(atom.IsInRing()),                # in ring
    ]
    # The feature vector must have a consistent size of 16
    return torch.tensor(features, dtype=torch.float)

# --- Molecule to Graph Conversion ---
def mol_to_graph(mol):
    if mol is None:
        return None
        
    num_node_features = DEFAULT_NUM_ATOM_FEATURES # Expected number of features per atom
    
    node_features_list = []
    edge_list = []
    edge_features_list = []

    for atom in mol.GetAtoms():
        feats = atom_features(atom)
        if len(feats) != num_node_features:
            continue
        node_features_list.append(feats)

    for bond in mol.GetBonds():
        start_node = bond.GetBeginAtomIdx()
        end_node = bond.GetEndAtomIdx()
        
        bt = bond.GetBondType()
        fbond = [
            1 if bt == Chem.rdchem.BondType.SINGLE else 0,
            1 if bt == Chem.rdchem.BondType.DOUBLE else 0,
            1 if bt == Chem.rdchem.BondType.TRIPLE else 0,
            1 if bt == Chem.rdchem.BondType.AROMATIC else 0,
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
        ]
        
        edge_list.append([start_node, end_node])
        edge_features_list.append(torch.tensor(fbond, dtype=torch.float))

        # Add reverse edge for undirected graph
        edge_list.append([end_node, start_node])
        edge_features_list.append(torch.tensor(fbond, dtype=torch.float))

    x = torch.stack(node_features_list, dim=0) if node_features_list else torch.empty((0, num_node_features))
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.stack(edge_features_list, dim=0) if edge_features_list else torch.empty((0, 6), dtype=torch.float)
    
    # Ensure correct shape for edge_attr if it's empty
    if edge_attr.numel() == 0:
        edge_attr = torch.empty((0, 6), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# --- HIV Data Loading and Processing ---
def prepare_graph_data(url=HIV_DATA_URL, processed_path=HIV_DATA_URL, cache_path=HIV_PROCESSED_DATA_PATH):
    if os.path.exists(cache_path):
        print(f"Loading cached HIV data from {cache_path}...")
        try:
            with open(cache_path, 'rb') as f:
                train_dataset, val_dataset = pickle.load(f)
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
            return train_loader, val_loader, None
        except Exception as e:
            print(f"Failed to load cached HIV data ({e}). Rebuilding cache...")
            
    print(f"Downloading and processing HIV data from {url}...")
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"Failed to download or read CSV from {url}: {e}")
        return None, None, None

    # Filter out rows with missing SMILES or labels
    df.dropna(subset=['smiles', 'HIV_active'], inplace=True)
    
    dataset = []
    for index, row in df.iterrows():
        smiles = row['smiles']
        label = row['HIV_active'] # 1 for active, 0 for inactive
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Could not parse SMILES: {smiles}")
            continue
            
        graph = mol_to_graph(mol)
        if graph is None:
            print(f"Warning: Could not convert molecule to graph for SMILES: {smiles}")
            continue

        graph.y = torch.tensor([label], dtype=torch.float)
        graph.smiles = smiles # Store SMILES for potential debugging
        dataset.append(graph)

    # Simple train/val split (e.g., 80/20) - A proper split is better for reproducibility
    split_idx = int(len(dataset) * 0.8)
    train_dataset = dataset[:split_idx]
    val_dataset = dataset[split_idx:]
    # test_dataset = dataset[split_idx:] # Assuming no separate test set for now

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Cache the processed data
    with open(cache_path, 'wb') as f:
        pickle.dump((train_dataset, val_dataset), f)

    print(f"HIV data processed: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
    return train_loader, val_loader, None


# --- Tox21 Data Loading and Processing ---
def prepare_tox_data(url=TOX_DATA_URL, cache_path=TOX_PROCESSED_DATA_PATH):
    if os.path.exists(cache_path):
        print(f"Loading cached Tox21 data from {cache_path}...")
        try:
            with open(cache_path, 'rb') as f:
                train_dataset, val_dataset = pickle.load(f)
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
            return train_loader, val_loader, None
        except Exception as e:
            print(f"Failed to load cached Tox21 data ({e}). Rebuilding cache...")

    print(f"Downloading and processing Tox21 data from {url}...")
    try:
        # Use pandas to read gzipped CSV directly
        df = pd.read_csv(url, compression='gzip')
    except Exception as e:
        print(f"Failed to download or read gzipped CSV from {url}: {e}")
        return None, None, None

    # Tox21 has 12 tasks. We'll create a binary 'toxic' label: 1 if active in ANY assay, 0 otherwise.
    # Task columns are typically named like '（'(NR-AR)', '（'(SR-MMP)', etc.
    # Let's find them dynamically. Assuming 'smiles' is the first column.
    task_columns = df.columns[2:14] # Assuming columns 2 through 13 are the tasks
    
    # Filter out rows where all task labels are NaN
    df.dropna(subset=task_columns, how='all', inplace=True)

    dataset = []
    for index, row in df.iterrows():
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Could not parse SMILES: {smiles}")
            continue

        # Determine toxicity label: 1 if active in any task, 0 otherwise
        # Tasks might have NaN values, treat NaN as 0 (not active) for this binary label
        is_toxic = 0
        for col in task_columns:
            label = row[col]
            if pd.notna(label) and label == 1.0: # Check if it's specifically 1.0 (active)
                is_toxic = 1
                break
        
        graph = mol_to_graph(mol)
        if graph is None:
            print(f"Warning: Could not convert molecule to graph for SMILES: {smiles}")
            continue
            
        graph.y = torch.tensor([is_toxic], dtype=torch.float)
        graph.smiles = smiles # Store SMILES
        dataset.append(graph)

    # Simple train/val split (e.g., 80/20)
    split_idx = int(len(dataset) * 0.8)
    train_dataset = dataset[:split_idx]
    val_dataset = dataset[split_idx:]
    # test_dataset = dataset[split_idx:]

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Cache the processed data
    with open(cache_path, 'wb') as f:
        pickle.dump((train_dataset, val_dataset), f)

    print(f"Tox21 data processed: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
    return train_loader, val_loader, None

# --- Combined Data Loader Function ---
def get_data_loaders(hiv_train_loader=None, hiv_val_loader=None, tox_train_loader=None, tox_val_loader=None):
    """
    Provides access to data loaders. If loaders are not provided, it loads them.
    This function is a placeholder for potentially more complex data handling,
    like creating combined iterators for multi-task learning if needed later.
    For now, it ensures data is loaded and returns them.
    """
    if hiv_train_loader is None:
        print("Loading HIV data loaders...")
        hiv_train_loader, hiv_val_loader, _ = prepare_graph_data()
        
    if tox_train_loader is None:
        print("Loading Tox21 data loaders...")
        tox_train_loader, tox_val_loader, _ = prepare_tox_data()
        
    return hiv_train_loader, hiv_val_loader, tox_train_loader, tox_val_loader
