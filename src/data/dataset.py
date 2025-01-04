import torch
from torch.utils.data import Dataset
import numpy as np

class MoleculeDataset(Dataset):
    def __init__(self, smiles_list, solubility_values, tokenizer):
        """
        Args:
            smiles_list: List of SMILES strings
            solubility_values: List of solubility values
            tokenizer: SMILESTokenizer instance
        """
        self.smiles_list = smiles_list
        self.solubility_values = torch.FloatTensor(solubility_values).squeeze()
        self.tokenizer = tokenizer
        
        # Build vocabulary if not already built
        if len(tokenizer.token_to_id) == 0:
            tokenizer.build_vocabulary(smiles_list)
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        solubility = self.solubility_values[idx]
        
        # Convert SMILES to token IDs
        token_ids = self.tokenizer.tokenize(smiles)
        
        return {
            'input_ids': torch.tensor(token_ids),
            'target': solubility
        }
