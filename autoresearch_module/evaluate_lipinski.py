import argparse
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen

def evaluate_lipinski(smiles_list):
    print(f"{'SMILES Fragment':<25} | {'MolWt (<=500)':<13} | {'LogP (<=5)':<11} | {'H-Donors (<=5)':<14} | {'H-Acceptors (<=10)':<18} | {'Violations'}")
    print("-" * 105)
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"{smiles[:22] + '...':<25} | {'Invalid SMILES':<80}")
            continue
            
        # Calculate properties
        mol_wt = Descriptors.MolWt(mol)
        log_p = Crippen.MolLogP(mol)
        h_donors = Descriptors.NumHDonors(mol)
        h_acceptors = Descriptors.NumHAcceptors(mol)
        
        # Check violations
        violations = 0
        if mol_wt > 500: violations += 1
        if log_p > 5: violations += 1
        if h_donors > 5: violations += 1
        if h_acceptors > 10: violations += 1
        
        # Format output
        wt_str = f"{mol_wt:.2f} {'❌' if mol_wt > 500 else '✅'}"
        logp_str = f"{log_p:.2f} {'❌' if log_p > 5 else '✅'}"
        hd_str = f"{h_donors} {'❌' if h_donors > 5 else '✅'}"
        ha_str = f"{h_acceptors} {'❌' if h_acceptors > 10 else '✅'}"
        
        status = "Pass" if violations <= 1 else "Fail"
        viol_str = f"{violations} ({status})"
        
        print(f"{smiles[:22] + '...':<25} | {wt_str:<13} | {logp_str:<11} | {hd_str:<14} | {ha_str:<18} | {viol_str}")

if __name__ == "__main__":
    # The 3 AI-generated AZT variants
    default_variants = [
        # The Runner Up: Iodine-Lipid Tail
        "Cc1cn(C2CC(N=[N+]=[NH2+])C(COC(=O)CCCCCCCCCCCI)O2)c(=O)[nH]c1=O",
        # The Miss: Phosphate Ester
        "Cc1cn(C2CCC(COP(=O)(OCC(=O)c3ccccc3)OCC(=O)c3ccccc3)O2)c(=O)[nH]c1=O",
        # The Winner: Fluorinated Lipid-Azide
        "Cc1cn(C2CC(F)C(COC(=O)CCCCCCCCCCCN=[N+]=[N-])O2)c(=O)[nH]c1=O",
        # Original AZT for comparison
        "Cc1cn([C@H]2C[C@H](N=[N+]=[N-])[C@@H](CO)O2)c(=O)[nH]c1=O"
    ]

    parser = argparse.ArgumentParser(description="Evaluate molecules against Lipinski's Rule of Five.")
    parser.add_argument("--smiles", nargs='*', help="List of SMILES strings to evaluate. If omitted, uses the default AZT variants.")
    
    args = parser.parse_args()
    
    smiles_to_test = args.smiles if args.smiles else default_variants
    
    print("\n--- Lipinski's Rule of Five Evaluation ---")
    evaluate_lipinski(smiles_to_test)
    print("\n*Note: An orally active drug typically has no more than 1 violation.*")