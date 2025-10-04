# models/smiles_generator.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import torch.optim as optim

class SMILESGenerator(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, num_layers=2):
        super(SMILESGenerator, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Character embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # LSTM layers - learns sequence patterns
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.3)
        
        # Output layer - predicts next character
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        # Convert character indices to embeddings
        embedded = self.embedding(x)
        
        # Process through LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Predict next character probabilities
        output = self.fc(lstm_out)
        
        return output, hidden

class AlzheimerDrugGenerator:
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self.model = None
        self.max_length = 80
        
    def build_vocabulary(self, smiles_list):
        """Build character vocabulary from SMILES strings"""
        print("Building character vocabulary...")
        
        # Collect all unique characters from SMILES
        all_chars = set()
        for smiles in smiles_list:
            all_chars.update(list(smiles))
        
        # Add special tokens
        all_chars.add('<START>')  # Start of sequence
        all_chars.add('<END>')    # End of sequence  
        all_chars.add('<PAD>')    # Padding
        
        # Create mapping between characters and indices
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(all_chars))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        print(f"Vocabulary built with {self.vocab_size} characters")
        
    def smiles_to_tensor(self, smiles):
        """Convert SMILES string to tensor of character indices"""
        # Add start and end tokens
        tokens = ['<START>'] + list(smiles) + ['<END>']
        
        # Convert characters to indices
        indices = []
        for token in tokens:
            if token in self.char_to_idx:
                indices.append(self.char_to_idx[token])
            else:
                # Handle unknown characters
                indices.append(self.char_to_idx['<PAD>'])
        
        # Pad or truncate to fixed length
        if len(indices) < self.max_length:
            indices += [self.char_to_idx['<PAD>']] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
            
        return torch.tensor(indices, dtype=torch.long)
    
    def tensor_to_smiles(self, tensor):
        """Convert tensor back to SMILES string"""
        tokens = []
        for idx in tensor:
            if idx.item() in self.idx_to_char:
                char = self.idx_to_char[idx.item()]
                if char == '<END>':
                    break  # Stop at end token
                if char not in ['<START>', '<PAD>']:
                    tokens.append(char)
        return ''.join(tokens)
    
    def train(self, smiles_list, epochs=300, lr=0.001):
        """Train the model on Alzheimer's drug SMILES"""
        print("Starting training process...")
        
        # Build vocabulary from training data
        self.build_vocabulary(smiles_list)
        
        # Initialize model
        self.model = SMILESGenerator(self.vocab_size)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Use CrossEntropyLoss for classification
        criterion = nn.CrossEntropyLoss(ignore_index=self.char_to_idx['<PAD>'])
        
        # Prepare training data
        training_data = [self.smiles_to_tensor(smiles) for smiles in smiles_list]
        training_data = torch.stack(training_data)
        
        print(f"Training on {len(training_data)} molecules for {epochs} epochs...")
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for sequence in training_data:
                # Input: all but last character, Target: all but first character
                inputs = sequence[:-1].unsqueeze(0)
                targets = sequence[1:].unsqueeze(0)
                
                # Forward pass
                optimizer.zero_grad()
                output, _ = self.model(inputs)
                
                # Calculate loss
                loss = criterion(output.view(-1, self.vocab_size), targets.view(-1))
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Print progress every 50 epochs
            if epoch % 50 == 0:
                avg_loss = total_loss / len(training_data)
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
                
                # Generate sample to show progress
                if epoch > 0:
                    samples = self.generate_molecules(2)
                    print(f"  Samples: {samples}")
    
    def generate_molecules(self, num_molecules=10, temperature=0.8):
        """Generate new molecules using the trained model"""
        if self.model is None:
            raise ValueError("Model must be trained before generation!")
        
        self.model.eval()
        generated_molecules = []
        
        print(f"Generating {num_molecules} molecules...")
        
        for i in range(num_molecules):
            # Start with <START> token
            current_token = torch.tensor([[self.char_to_idx['<START>']]], dtype=torch.long)
            hidden = None
            generated_sequence = []
            
            # Generate characters until <END> or max length
            for step in range(self.max_length - 1):
                with torch.no_grad():
                    output, hidden = self.model(current_token, hidden)
                
                # Apply temperature for creativity control
                # Higher temperature = more random, Lower = more conservative
                output = output / temperature
                probabilities = torch.softmax(output[0, -1], dim=0)
                
                # Sample next character from probability distribution
                next_token = torch.multinomial(probabilities, 1)
                next_char = self.idx_to_char[next_token.item()]
                
                # Stop if end token is generated
                if next_char == '<END>':
                    break
                
                # Add character to sequence (ignore special tokens)
                if next_char not in ['<START>', '<PAD>']:
                    generated_sequence.append(next_char)
                
                # Use generated character as next input
                current_token = next_token.unsqueeze(0)
            
            # Convert character sequence to SMILES
            smiles = ''.join(generated_sequence)
            
            # Only keep valid molecules
            if self.is_valid_smiles(smiles):
                generated_molecules.append(smiles)
                print(f"  Generated valid molecule {i+1}: {smiles}")
        
        print(f"Successfully generated {len(generated_molecules)} valid molecules")
        return generated_molecules
    
    def is_valid_smiles(self, smiles):
        """Check if generated SMILES string represents a valid molecule"""
        if not smiles or len(smiles) < 3:
            return False
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def calculate_drug_properties(self, smiles):
        """Calculate key drug properties for evaluation"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        properties = {
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'hydrogen_bond_donors': Descriptors.NumHDonors(mol),
            'hydrogen_bond_acceptors': Descriptors.NumHAcceptors(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'polar_surface_area': Descriptors.TPSA(mol),
            'aromatic_rings': Descriptors.NumAromaticRings(mol),
        }
        
        # Check Lipinski's Rule of Five
        properties['passes_lipinski'] = all([
            properties['molecular_weight'] <= 500,
            properties['logp'] <= 5,
            properties['hydrogen_bond_donors'] <= 5,
            properties['hydrogen_bond_acceptors'] <= 10
        ])
        
        return properties