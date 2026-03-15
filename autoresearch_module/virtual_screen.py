import torch
import pandas as pd
from rdkit import Chem
from torch_geometric.loader import DataLoader
import warnings
import os
import time

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

from data_loader import mol_to_graph, TOX_DATA_URL
from train_gnn import AntiviralGNN

def run_screen(model_path="best_gnn_model.pt", top_k=20):
    print(f"--- DecentraPharma Virtual Screening ---")
    print(f"Downloading screening library from {TOX_DATA_URL}...")
    
    try:
        df = pd.read_csv(TOX_DATA_URL, compression='gzip')
    except Exception as e:
        print(f"Failed to load screening dataset: {e}")
        return

    # Deduplicate and filter
    df = df.dropna(subset=['smiles']).drop_duplicates(subset=['smiles'])
    print(f"Library loaded. Screening {len(df)} unique compounds...")

    # Initialize your HIV Model
    model = AntiviralGNN()
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Model '{model_path}' loaded successfully.")
    except Exception as e:
        print(f"Could not load model (Did you run train_hiv.py first?): {e}")
        return

    model.eval()
    
    results = []
    start_time = time.time()

    print("Converting library to graphs and predicting (this may take a minute)...")
    for idx, row in df.iterrows():
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: continue
        
        data = mol_to_graph(mol)
        if data is None or data.x.numel() == 0: continue
        
        loader = DataLoader([data], batch_size=1)
        batch = next(iter(loader))
        
        with torch.no_grad():
            try:
                edge_attr = getattr(batch, 'edge_attr', None)
                out = model(batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr)
            except TypeError:
                out = model(batch.x, batch.edge_index, batch.batch)
            prob = torch.sigmoid(out).item()
            
        results.append({
            'mol_id': row.get('mol_id', f'Compound_{idx}'),
            'smiles': smiles,
            'hiv_active_probability': prob
        })

    elapsed = time.time() - start_time
    results_df = pd.DataFrame(results).sort_values(by='hiv_active_probability', ascending=False)
    
    print(f"\nScreening completed in {elapsed:.1f} seconds.")
    print(f"\n🔥 TOP {top_k} DISCOVERED HITS 🔥")
    print(results_df.head(top_k).to_string(index=False))
    
    results_df.to_csv("virtual_screening_hits.csv", index=False)
    print("\nFull ranked results saved to 'virtual_screening_hits.csv'")

if __name__ == "__main__":
    run_screen()