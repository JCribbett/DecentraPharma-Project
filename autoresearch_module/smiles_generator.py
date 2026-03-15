import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from rdkit import Chem
import os
import warnings

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

# --- 1. Vocabulary Builder ---
class SMILESVocab:
    def __init__(self, smiles_list):
        chars = set(''.join(smiles_list))
        self.char2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
        for char in sorted(list(chars)):
            self.char2idx[char] = len(self.char2idx)
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)

    def encode(self, smiles):
        return [self.char2idx['<sos>']] + [self.char2idx[c] for c in smiles] + [self.char2idx['<eos>']]

# --- 2. Dataset ---
class SMILESDataset(Dataset):
    def __init__(self, smiles_list, vocab, max_len=100):
        self.vocab = vocab
        self.max_len = max_len
        self.data = []
        for s in smiles_list:
            if len(s) <= self.max_len - 2:
                encoded = self.vocab.encode(s)
                padded = encoded + [self.vocab.char2idx['<pad>']] * (self.max_len - len(encoded))
                self.data.append(torch.tensor(padded, dtype=torch.long))
    
    def __len__(self): 
        return len(self.data)
        
    def __getitem__(self, idx): 
        # Input is sequence up to last char, Target is sequence shifted by 1
        return self.data[idx][:-1], self.data[idx][1:]

# --- 3. Generative RNN Model ---
class SMILESRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

# --- 4. Generation Function ---
def generate_molecules(model, vocab, num_mols=10, max_len=100, temperature=1.0):
    model.eval()
    molecules = []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for _ in range(num_mols):
            input_seq = torch.tensor([[vocab.char2idx['<sos>']]]).to(device)
            hidden = None
            generated_chars = []
            
            for _ in range(max_len):
                output, hidden = model(input_seq, hidden)
                logits = output[0, -1, :] / temperature
                probs = F.softmax(logits, dim=0)
                next_char_idx = torch.multinomial(probs, 1).item()
                
                if next_char_idx == vocab.char2idx['<eos>']:
                    break
                    
                generated_chars.append(vocab.idx2char[next_char_idx])
                input_seq = torch.tensor([[next_char_idx]]).to(device)
            
            smiles = ''.join(generated_chars)
            molecules.append(smiles)
    return molecules

def main():
    print("Downloading HIV dataset to extract Active compounds...")
    df = pd.read_csv("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv")
    active_smiles = df[df['HIV_active'] == 1]['smiles'].dropna().tolist()
    print(f"Found {len(active_smiles)} active HIV inhibitors for training.")

    vocab = SMILESVocab(active_smiles)
    dataset = SMILESDataset(active_smiles, vocab)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SMILESRNN(vocab.vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    epochs = 10
    print(f"\nTraining Generative Model for {epochs} epochs on {device}...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output, _ = model(x)
            loss = criterion(output.transpose(1, 2), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")

    print("\n--- Generating 10 Brand New Molecules ---")
    new_smiles = generate_molecules(model, vocab, num_mols=10, temperature=0.8)
    
    valid_count = 0
    for i, s in enumerate(new_smiles):
        mol = Chem.MolFromSmiles(s)
        is_valid = "VALID" if mol is not None else "INVALID"
        if mol is not None: valid_count += 1
        print(f"{i+1:2d}. {is_valid:<7} | {s}")
        
    print(f"\n{valid_count}/10 molecules were chemically valid.")
    print("(Note: To increase validity rate, train for 50+ epochs with a larger dataset!)")
    
    # Save the generator
    torch.save(model.state_dict(), "smiles_generator.pth")
    print("Saved generative model to 'smiles_generator.pth'")

if __name__ == "__main__":
    main()