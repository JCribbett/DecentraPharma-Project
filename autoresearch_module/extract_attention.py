import argparse
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.loader import DataLoader
import warnings
import os

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

from prepare_graph import mol_to_graph
from best_model import AntiviralGNN

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_gnn_model.pt")

def analyze_attention(smiles, model_path=DEFAULT_MODEL_PATH):
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return

    # Create Graph Data (Passing dummy label 0)
    data = mol_to_graph(smiles, label=0)
    if data is None or data.x.numel() == 0:
        print("Could not create graph representation.")
        return

    loader = DataLoader([data], batch_size=1)
    batch = next(iter(loader))

    # Load Model
    model = AntiviralGNN()
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()

    with torch.no_grad():
        out, attn_weights = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr, return_attention_weights=True)
        prob = torch.sigmoid(out).item()

    print("-" * 50)
    print(f"Molecule: {smiles}")
    print(f"Predicted HIV Active Probability: {prob:.4f}")
    print("-" * 50)

    # Analyze the very last graph layer (Layer 4)
    edge_index, alpha = attn_weights[-1]
    
    # The attention tensor 'alpha' has shape [num_edges, num_heads]. 
    # We average across the attention heads to get a single score per bond.
    alpha_mean = alpha.mean(dim=1).squeeze()
    
    # Sort edges by attention weight
    sorted_indices = torch.argsort(alpha_mean, descending=True)
    
    print("Top 5 highest attention bonds (edges) in the final layer:\n")
    highlight_bonds = []
    for idx in sorted_indices[:5]:
        src, dst = edge_index[0, idx].item(), edge_index[1, idx].item()
        weight = alpha_mean[idx].item()
        src_atom = mol.GetAtomWithIdx(src).GetSymbol()
        dst_atom = mol.GetAtomWithIdx(dst).GetSymbol()
        
        print(f"  Atom {src:2d} ({src_atom}) -> Atom {dst:2d} ({dst_atom}) | Attention Score: {weight:.4f}")
        
        bond = mol.GetBondBetweenAtoms(src, dst)
        if bond: highlight_bonds.append(bond.GetIdx())

    img = Draw.MolToImage(mol, highlightBonds=highlight_bonds, size=(600, 600))
    img.save("attention_map.png")
    print("\n📸 Saved visual attention map to 'attention_map.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and interpret GATv2 attention weights for a molecule.")
    parser.add_argument("--smiles", type=str, required=True, help="SMILES string of the molecule.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to the trained GNN weights.")
    
    args = parser.parse_args()
    analyze_attention(args.smiles, args.model)