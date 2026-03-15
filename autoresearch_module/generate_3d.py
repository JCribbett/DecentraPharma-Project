import argparse
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import warnings

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

def generate_3d_conformer(smiles, output_filename="molecule.sdf"):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"❌ Invalid SMILES: {smiles}")
        return False
    
    # 1. Add Hydrogens (Essential for 3D geometry and docking interactions)
    mol = Chem.AddHs(mol)
    
    # 2. Generate initial 3D coordinates using ETKDGv3 (cutting-edge geometry algorithm)
    print(f"Generating 3D coordinates for {smiles}...")
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    res = AllChem.EmbedMolecule(mol, params)
    
    if res != 0:
        print("⚠️ Initial embedding failed. Trying random coordinates...")
        res = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
        if res != 0:
            print("❌ Failed to generate 3D conformer.")
            return False

    # 3. Optimize the 3D geometry using the MMFF94 force field (Energy Minimization)
    print("Optimizing geometry (Energy Minimization)...")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500, nonBondedThresh=100.0)
    except Exception as e:
        print(f"⚠️ Optimization warning: {e}")

    # 4. Save to SDF format (Universal 3D molecular format used by docking software)
    writer = Chem.SDWriter(output_filename)
    writer.write(mol)
    writer.close()
    
    print(f"✅ 3D Conformer saved to {output_filename}")
    return True

def batch_generate_3d(csv_path, smiles_col="smiles", output_dir="3d_molecules"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_csv(csv_path)
    print(f"Processing {len(df)} molecules from {csv_path}...")
    
    success_count = 0
    for idx, row in df.iterrows():
        smiles = str(row[smiles_col])
        mol_id = str(row.get('mol_id', f'mol_{idx}'))
        
        # Clean up filename
        safe_id = "".join([c for c in mol_id if c.isalpha() or c.isdigit() or c=='_']).rstrip()
        output_file = os.path.join(output_dir, f"{safe_id}.sdf")
        
        if generate_3d_conformer(smiles, output_file):
            success_count += 1
            
    print(f"\n🎉 Successfully generated {success_count}/{len(df)} 3D models in '{output_dir}/'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate optimized 3D conformers for molecular docking.")
    parser.add_argument("--smiles", type=str, help="A single SMILES string to process.")
    parser.add_argument("--csv", type=str, help="Path to a CSV file containing multiple SMILES.")
    parser.add_argument("--smiles_col", type=str, default="smiles", help="Column name for SMILES in CSV.")
    parser.add_argument("--output", type=str, default="molecule.sdf", help="Output SDF filename (for single SMILES).")
    parser.add_argument("--output_dir", type=str, default="3d_molecules", help="Output directory (for CSV batch).")
    
    args = parser.parse_args()
    
    if args.csv:
        batch_generate_3d(args.csv, args.smiles_col, args.output_dir)
    elif args.smiles:
        generate_3d_conformer(args.smiles, args.output)
    else:
        print("Please provide either --smiles or --csv")