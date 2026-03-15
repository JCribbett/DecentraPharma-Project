import argparse
import torch
from rdkit import Chem
from torch_geometric.loader import DataLoader
import warnings

# Suppress RDKit warnings for cleaner output
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

from data_loader import mol_to_graph
from model_hiv import AntiviralGNN

def predict(smiles_list, model_path, hidden_dim=64, dropout=0.2):
    # 1. Initialize the model architecture
    # Note: hidden_dim and dropout must match what was used during training
    model = AntiviralGNN(num_features=16, hidden_dim=hidden_dim, dropout=dropout, num_classes=1)
    
    # 2. Load the trained weights
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        # Handle both raw state_dicts (from MODEL_BACKUP) and full checkpoints (from CHECKPOINT_FILE)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Successfully loaded model weights from '{model_path}'\n")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.eval()

    # 3. Process and predict each SMILES string
    print(f"{'SMILES':<50} | {'Probability (Active/Toxic)':<25}")
    print("-" * 78)
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"{smiles:<50} | Invalid SMILES")
            continue
            
        data = mol_to_graph(mol)
        if data is None or data.x.numel() == 0:
            print(f"{smiles:<50} | Could not create graph")
            continue
            
        # Use a DataLoader of size 1 to easily add the required 'batch' indices vector
        loader = DataLoader([data], batch_size=1)
        batch_data = next(iter(loader))
        
        with torch.no_grad():
            out = model(batch_data)
            prob = torch.sigmoid(out).item()
            
        print(f"{smiles:<50} | {prob:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on unseen molecules using trained GNN models.")
    parser.add_argument("--smiles", nargs='+', required=True, help="One or more SMILES strings to test.")
    parser.add_argument("--model", type=str, default="hiv_model_backup.pth", help="Path to the trained model weights (.pth or .pth.tar).")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size of the trained model (default: 64).")
    
    args = parser.parse_args()
    predict(args.smiles, args.model, args.hidden_dim)