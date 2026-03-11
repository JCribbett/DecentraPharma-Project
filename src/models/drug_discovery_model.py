# Example: Model for predicting molecular properties
from typing import List, Dict, Any, Union
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.core.models import BaseModel
from src.utils.cheminformatics import MoleculeHandler

class SimpleQSAR(nn.Module):
    """A simple Multi-Layer Perceptron for QSAR prediction."""
    def __init__(self, input_size=1024, hidden_size=128, output_size=1):
        super(SimpleQSAR, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)

class DrugDiscoveryModel(BaseModel):
    def __init__(self, input_size=1024, learning_rate=0.001):
        super().__init__()
        self.input_size = input_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleQSAR(input_size=input_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.molecule_handler = MoleculeHandler()

    def _featurize(self, smiles_list: List[str]) -> torch.Tensor:
        """Converts SMILES strings to tensor fingerprints."""
        features = []
        for smiles in smiles_list:
            mol = self.molecule_handler.load_molecule_from_file(smiles, file_format='smi')
            if mol:
                # Returns list of '0'/'1' strings
                fp_str_list = self.molecule_handler.get_molecule_fingerprints(mol, fingerprint_type='morgan')
                if fp_str_list:
                    features.append([int(bit) for bit in fp_str_list])
                    continue
            # Fallback for invalid molecules or failed fingerprint generation
            features.append([0] * self.input_size)
        return torch.tensor(features, dtype=torch.float32).to(self.device)

    def train(self, data: List[Dict[str, Any]], epochs=5):
        print("Training Drug Discovery Model...")
        self.model.train()
        smiles = [item['smiles'] for item in data]
        targets = torch.tensor([item['target'] for item in data], dtype=torch.float32).view(-1, 1).to(self.device)
        features = self._featurize(smiles)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    def predict(self, molecular_data: Union[str, List[str]]):
        print("Predicting molecular properties...")
        self.model.eval()
        if isinstance(molecular_data, str):
            molecular_data = [molecular_data]
            
        features = self._featurize(molecular_data)
        with torch.no_grad():
            predictions = self.model(features).cpu().numpy().flatten().tolist()
            
        return {"predictions": predictions}
