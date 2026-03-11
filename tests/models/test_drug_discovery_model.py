import unittest

try:
    import torch
    from src.models.drug_discovery_model import DrugDiscoveryModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch is not installed")
class TestDrugDiscoveryModel(unittest.TestCase):
    def setUp(self):
        # Initialize model with standard parameters
        self.model = DrugDiscoveryModel(input_size=1024, learning_rate=0.01)
        
        # Simple dataset: Methane, Ethane, Propane with dummy targets
        self.training_data = [
            {'smiles': 'C', 'target': 1.0},
            {'smiles': 'CC', 'target': 2.0},
            {'smiles': 'CCC', 'target': 3.0}
        ]

    def test_model_initialization(self):
        """Test if the model components are initialized correctly."""
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.optimizer)
        self.assertEqual(self.model.input_size, 1024)

    def test_featurization_shape(self):
        """Test if SMILES are converted to tensors of correct shape."""
        smiles_list = ['C', 'CC']
        features = self.model._featurize(smiles_list)
        self.assertTrue(torch.is_tensor(features))
        self.assertEqual(features.shape, (2, 1024))

    def test_training_loop(self):
        """Test if the training loop runs without errors."""
        try:
            self.model.train(self.training_data, epochs=2)
        except Exception as e:
            self.fail(f"Training raised an exception: {e}")

    def test_prediction_single(self):
        """Test prediction for a single SMILES string."""
        result = self.model.predict('C')
        self.assertIn('predictions', result)
        self.assertIsInstance(result['predictions'], list)
        self.assertEqual(len(result['predictions']), 1)
        self.assertIsInstance(result['predictions'][0], float)

    def test_prediction_batch(self):
        """Test prediction for a list of SMILES strings."""
        smiles_list = ['C', 'CC', 'CCC']
        result = self.model.predict(smiles_list)
        self.assertIn('predictions', result)
        self.assertEqual(len(result['predictions']), 3)

if __name__ == "__main__":
    unittest.main()
