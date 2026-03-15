import torch
import torch.optim as optim
from torch.distributions import Categorical
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.loader import DataLoader
import pandas as pd
import os
import warnings

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

from smiles_generator import SMILESRNN, SMILESVocab, generate_molecules
from train_gnn import AntiviralGNN as HIV_GNN
from model_hiv import AntiviralGNN as Tox_GNN
from data_loader import mol_to_graph

def load_vocab():
    """Rebuilds the exact vocabulary used during the generator's pre-training."""
    print("Loading vocabulary...")
    df = pd.read_csv("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv")
    active_smiles = df[df['HIV_active'] == 1]['smiles'].dropna().tolist()
    return SMILESVocab(active_smiles)

def get_reward(smiles, hiv_model, tox_model, device):
    """Uses both GNNs to score the generated molecule for high HIV activity and low toxicity."""
    mol = Chem.MolFromSmiles(smiles)
    # Severe penalty for invalid chemical structures
    if mol is None:
        return 0.0, 0.0, 0.0
    
    data = mol_to_graph(mol)
    if data is None or data.x.numel() == 0:
        return 0.0, 0.0, 0.0
    
    loader = DataLoader([data], batch_size=1)
    batch = next(iter(loader)).to(device)
    
    with torch.no_grad():
        # HIV Prediction (LLM Model)
        try:
            edge_attr = getattr(batch, 'edge_attr', None)
            out_hiv = hiv_model(batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr)
        except TypeError:
            out_hiv = hiv_model(batch.x, batch.edge_index, batch.batch)
        prob_hiv = torch.sigmoid(out_hiv).item()
        
        # Toxicity Prediction (Baseline Model)
        out_tox = tox_model(batch)
        prob_tox = torch.sigmoid(out_tox).item()
    
    # Multi-Objective Reward: High HIV activity minus Toxicity penalty
    reward = max(0.0, prob_hiv - prob_tox)
    return reward, prob_hiv, prob_tox

def rl_finetune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = load_vocab()
    
    print("\nLoading Pre-trained Generative RNN...")
    rnn = SMILESRNN(vocab.vocab_size).to(device)
    if os.path.exists("smiles_generator.pth"):
        rnn.load_state_dict(torch.load("smiles_generator.pth", map_location=device))
    else:
        print("Warning: smiles_generator.pth not found. Make sure you trained the generator first!")
        return
        
    print("Loading Pre-trained HIV Oracle (Smart Version)...")
    hiv_gnn = HIV_GNN().to(device)
    if os.path.exists("best_gnn_model.pt"):
        checkpoint = torch.load("best_gnn_model.pt", map_location=device, weights_only=True)
        hiv_gnn.load_state_dict(checkpoint)
    hiv_gnn.eval()

    print("Loading Pre-trained Toxicity Oracle...")
    tox_gnn = Tox_GNN(num_features=16, hidden_dim=64, dropout=0.2, num_classes=1).to(device)
    if os.path.exists("tox_model_backup.pth"):
        checkpoint = torch.load("tox_model_backup.pth", map_location=device, weights_only=False)
        if 'state_dict' in checkpoint:
            tox_gnn.load_state_dict(checkpoint['state_dict'])
        else:
            tox_gnn.load_state_dict(checkpoint)
    tox_gnn.eval()
    
    # We use a smaller learning rate for fine-tuning
    optimizer = optim.Adam(rnn.parameters(), lr=5e-5)
    
    epochs = 1000
    batch_size = 32 # number of molecules generated per weight update
    max_len = 100
    entropy_weight = 0.02 # Hyperparameter to encourage exploration
    
    print("\nStarting RL Fine-tuning (REINFORCE algorithm)...")
    for epoch in range(epochs):
        rnn.train()
        batch_loss, batch_reward, unique_valid_mols = 0, 0, 0
        batch_smiles = set()
        
        optimizer.zero_grad()
        for _ in range(batch_size):
            input_seq = torch.tensor([[vocab.char2idx['<sos>']]]).to(device)
            hidden = None
            log_probs, generated_chars, entropies = [], [], []
            
            for _ in range(max_len):
                output, hidden = rnn(input_seq, hidden)
                probs = torch.softmax(output[0, -1, :], dim=0)
                
                # Sample an action (character) from the probability distribution
                m = Categorical(probs)
                action = m.sample()
                log_probs.append(m.log_prob(action))
                entropies.append(m.entropy())
                
                char = vocab.idx2char[action.item()]
                if char == '<eos>': break
                    
                generated_chars.append(char)
                input_seq = torch.tensor([[action.item()]]).to(device)
            
            smiles = ''.join(generated_chars)
            reward, _, _ = get_reward(smiles, hiv_gnn, tox_gnn, device)
            
            # Uniqueness penalty to prevent mode collapse
            if smiles in batch_smiles:
                reward *= 0.1 # Severe penalty for generating a duplicate in the same batch
            elif reward > 0:
                batch_smiles.add(smiles)
                unique_valid_mols += 1
            
            # Calculate Policy Gradient Loss: -log(P(action)) * reward
            entropy_bonus = entropy_weight * sum(entropies)
            loss = -sum(log_probs) * reward - entropy_bonus
            loss.backward() # Accumulate gradients over the batch
            
            batch_loss += loss.item()
            batch_reward += reward
            
        optimizer.step() # Update generator weights
        
        avg_reward = batch_reward / batch_size
        print(f"Epoch {epoch+1:3d}/{epochs} | Avg Reward: {avg_reward:.4f} | Unique Valid: {unique_valid_mols:2d}/{batch_size}")

    print("\n--- Testing RL-Tuned Generator ---")
    new_smiles = generate_molecules(rnn, vocab, num_mols=5, temperature=0.8)
    
    mols_to_draw = []
    draw_labels = []
    
    for s in new_smiles:
        rew, p_hiv, p_tox = get_reward(s, hiv_gnn, tox_gnn, device)
        print(f"{s:<50} | HIV: {p_hiv:.4f} | Tox: {p_tox:.4f} | Net Reward: {rew:.4f}")
        
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            mols_to_draw.append(mol)
            draw_labels.append(f"Prob: {rew:.4f}")
            
    if mols_to_draw:
        img = Draw.MolsToGridImage(mols_to_draw, molsPerRow=min(5, len(mols_to_draw)), subImgSize=(300, 300), legends=draw_labels)
        img.save("generated_hits.png")
        print(f"\nSaved 2D structures to 'generated_hits.png'")
        
    torch.save(rnn.state_dict(), "smiles_generator_rl.pth")
    print("\nSaved RL-finetuned model to 'smiles_generator_rl.pth'")

if __name__ == "__main__":
    rl_finetune()