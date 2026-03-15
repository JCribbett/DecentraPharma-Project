import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
import os
import argparse
import warnings

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

def evaluate_csv(csv_path, top_k=20):
    if not os.path.exists(csv_path):
        print(f"Error: '{csv_path}' not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"Evaluating top {top_k} molecules from '{os.path.basename(csv_path)}'...\n")
    
    print(f"{'SMILES Fragment':<45} | {'MolWt (<=500)':<13} | {'LogP (<=5)':<11} | {'H-Donors (<=5)':<14} | {'H-Acceptors (<=10)':<18} | {'Violations'}")
    print("-" * 125)
    
    for idx, row in df.head(top_k).iterrows():
        smiles = str(row['smiles'])
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"{smiles[:42] + '...':<45} | {'Invalid SMILES':<80}")
            continue
            
        mol_wt = Descriptors.MolWt(mol)
        log_p = Crippen.MolLogP(mol)
        h_donors = Descriptors.NumHDonors(mol)
        h_acceptors = Descriptors.NumHAcceptors(mol)
        
        violations = 0
        if mol_wt > 500: violations += 1
        if log_p > 5: violations += 1
        if h_donors > 5: violations += 1
        if h_acceptors > 10: violations += 1
        
        wt_str = f"{mol_wt:.2f} {'❌' if mol_wt > 500 else '✅'}"
        logp_str = f"{log_p:.2f} {'❌' if log_p > 5 else '✅'}"
        hd_str = f"{h_donors} {'❌' if h_donors > 5 else '✅'}"
        ha_str = f"{h_acceptors} {'❌' if h_acceptors > 10 else '✅'}"
        
        status = "Pass" if violations <= 1 else "Fail"
        viol_str = f"{violations} ({status})"
        
        print(f"{smiles[:42] + '...':<45} | {wt_str:<13} | {logp_str:<11} | {hd_str:<14} | {ha_str:<18} | {viol_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a CSV of molecules against Lipinski's Rule of Five.")
    parser.add_argument("--csv", type=str, default="rl_generated_hits.csv", help="Path to the input CSV file.")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top molecules to evaluate.")
    args = parser.parse_args()
    evaluate_csv(args.csv, args.top_k)