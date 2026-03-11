# src/utils/cheminformatics.py

from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import List, Optional

class MoleculeHandler:
    """
    Handles RDKit molecule objects, loading from various formats,
    and calculating molecular descriptors.
    """
    def __init__(self):
        """Initializes the MoleculeHandler."""
        pass

    def load_molecule_from_file(self, filepath: str, file_format: str = 'sdf') -> Optional[Chem.Mol]:
        """
        Loads a molecule from a given file.

        Args:
            filepath (str): The path to the molecule file.
            file_format (str): The format of the file (e.g., 'sdf', 'mol', 'smi').
                               Defaults to 'sdf'.

        Returns:
            Optional[Chem.Mol]: The RDKit molecule object, or None if loading fails.
        """
        mol = None
        if file_format.lower() == 'sdf':
            # RDKit's SDMolSupplier reads a file containing multiple molecules
            # We'll assume for simplicity here it's a single molecule file or we take the first one
            suppl = Chem.SDMolSupplier(filepath)
            if suppl and len(suppl) > 0:
                mol = suppl[0]
        elif file_format.lower() == 'mol':
            mol = Chem.MolFromMolFile(filepath)
        elif file_format.lower() == 'smi':
            mol = Chem.MolFromSmiles(filepath) # Assumes filepath is actually SMILES string
        else:
            print(f"Unsupported file format: {file_format}")
            return None

        if not mol:
            print(f"Failed to load molecule from {filepath} in format {file_format}")
        return mol

    def calculate_descriptors(self, mol: Chem.Mol) -> dict:
        """
        Calculates a standard set of molecular descriptors for a given RDKit molecule.

        Args:
            mol (Chem.Mol): The RDKit molecule object.

        Returns:
            dict: A dictionary of descriptor names and their calculated values.
        """
        if not mol:
            return {}

        descriptors = {}
        # Common descriptors
        descriptors['MolLogP'] = Descriptors.MolLogP(mol)
        descriptors['MolWt'] = Descriptors.MolWt(mol)
        descriptors['NumHeavyAtoms'] = Descriptors.NumHeavyAtoms(mol)
        descriptors['NumRings'] = Descriptors.RingCount(mol)
        descriptors['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
        # Add more descriptors as needed
        # Example: Topological Polar Surface Area (TPSA)
        try:
            descriptors['TPSA'] = Chem.rdMolDescr.CalcTPSA(mol)
        except:
            descriptors['TPSA'] = 0.0 # Handle cases where TPSA might not be calculable

        return descriptors

    def get_molecule_fingerprints(self, mol: Chem.Mol, fingerprint_type: str = 'morgan') -> Optional[List[int]]:
        """
        Generates molecular fingerprints.

        Args:
            mol (Chem.Mol): The RDKit molecule object.
            fingerprint_type (str): Type of fingerprint ('morgan', 'atompair', 'topological').
                                    Defaults to 'morgan'.

        Returns:
            Optional[List[int]]: A list representing the fingerprint, or None if generation fails.
        """
        if not mol:
            return None

        if fingerprint_type.lower() == 'morgan':
            # Morgan fingerprints (similar to ECFP)
            # radius=2, nBits=1024 are common parameters
            fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            return list(fp.ToBitString()) # Convert to list of '0's and '1's
        # Add other fingerprint types if needed
        else:
            print(f"Unsupported fingerprint type: {fingerprint_type}")
            return None

# Example Usage (for testing purposes, can be removed or commented out)
if __name__ == '__main__':
    handler = MoleculeHandler()

    # Example 1: Load from SMILES string
    smiles = "CCO" # Ethanol
    mol_ethanol = handler.load_molecule_from_file(smiles, file_format='smi')
    if mol_ethanol:
        print(f"Loaded molecule from SMILES: {smiles}")
        desc_ethanol = handler.calculate_descriptors(mol_ethanol)
        print("Descriptors for Ethanol:", desc_ethanol)
        fp_ethanol = handler.get_molecule_fingerprints(mol_ethanol)
        # print("Morgan Fingerprint (first 50 bits):", "".join(fp_ethanol[:50]) + "...") # Print first 50 bits

    print("-" * 20)

    # Example 2: Load from a hypothetical SDF file (requires an actual SDF file)
    # Create a dummy SDF file for testing if you don't have one
    dummy_sdf_content = """
     RDKit          2D

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
"""
    # Save this content to a temporary file named 'dummy.sdf'
    try:
        with open("dummy.sdf", "w") as f:
            f.write(dummy_sdf_content)

        mol_sdf = handler.load_molecule_from_file("dummy.sdf", file_format='sdf')
        if mol_sdf:
            print("Loaded molecule from dummy.sdf")
            desc_sdf = handler.calculate_descriptors(mol_sdf)
            print("Descriptors for dummy molecule:", desc_sdf)
        else:
            print("Could not load from dummy.sdf. Ensure the content is valid.")

    except Exception as e:
        print(f"Error during dummy SDF file handling: {e}")
    finally:
        # Clean up the dummy file
        import os
        if os.path.exists("dummy.sdf"):
            os.remove("dummy.sdf")
