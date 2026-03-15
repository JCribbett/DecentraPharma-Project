import argparse
import torch
import pandas as pd
from rdkit import Chem
from torch_geometric.loader import DataLoader
import warnings
import os

# Suppress RDKit warnings for cleaner output
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

from data_loader import mol_to_graph
from model_hiv import AntiviralGNN

def predict_from_csv(csv_path, smiles_col, model_path, output_path, hidden_dim=64, dropout=0.2, batch_size=128):
    if not os.path.exists(csv_path):
        print(f"Error: Input CSV file '{csv_path}' not found.")
        return

    # 1. Initialize the model architecture
    model = AntiviralGNN(num_features=16, hidden_dim=hidden_dim, dropout=dropout, num_classes=1)
    
    # 2. Load the trained weights
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Successfully loaded model weights from '{model_path}'")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.eval()

    # 3. Read the CSV file
    try:
        df = pd.read_csv(csv_path)
        if smiles_col not in df.columns:
            print(f"Error: Column '{smiles_col}' not found in the CSV.")
            return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    print(f"Processing {len(df)} molecules from '{csv_path}'...")

    # 4. Prepare data for batching
    valid_data = []
    valid_indices = []
    predictions = [None] * len(df)

    for idx, row in df.iterrows():
        smiles = str(row[smiles_col])
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            data = mol_to_graph(mol)
            if data is not None and data.x.numel() > 0:
                valid_data.append(data)
                valid_indices.append(idx)
    
    print(f"Successfully parsed {len(valid_data)} valid molecules.")

    # 5. Run inference in batches
    if len(valid_data) > 0:
        loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
        
        all_probs = []
        with torch.no_grad():
            for batch in loader:
                out = model(batch)
                probs = torch.sigmoid(out).cpu().numpy().flatten()
                all_probs.extend(probs)
        
        # 6. Map predictions back to the original dataframe row indices
        for idx, prob in zip(valid_indices, all_probs):
            predictions[idx] = round(float(prob), 6)

    # 7. Add predictions and save
    df['prediction'] = predictions
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to '{output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch inference on a CSV of molecules using trained GNN models.")
    parser.add_argument("--csv", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Path to save the output CSV with predictions.")
    parser.add_argument("--smiles_col", type=str, default="smiles", help="Name of the column containing SMILES strings (default: 'smiles').")
    parser.add_argument("--model", type=str, default="hiv_model_backup.pth", help="Path to the trained model weights (.pth or .pth.tar).")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size of the trained model (default: 64).")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for inference (default: 128).")
    
    args = parser.parse_args()
    predict_from_csv(args.csv, args.smiles_col, args.model, args.output, args.hidden_dim, args.batch_size)