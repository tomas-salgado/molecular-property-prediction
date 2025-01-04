import pandas as pd
from rdkit import Chem
from pathlib import Path

def validate_smiles(smiles: str) -> bool: 
    """
    Check if SMILES string is valid
    Returns True if valid, False otherwise
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def load_and_validate_data():
    # Load data
    data_path = "data/delaney-processed.csv"
    df = pd.read_csv(data_path)
    
    # Keep only necessary columns
    processed_df = df[['smiles', 'measured log solubility in mols per litre']]
    
    # Rename for clarity
    processed_df = processed_df.rename(columns={
        'measured log solubility in mols per litre': 'solubility'
    })
    
    # Normalize solubility values
    mean = processed_df['solubility'].mean()
    std = processed_df['solubility'].std()
    processed_df['solubility'] = (processed_df['solubility'] - mean) / std
    
    return processed_df


print(load_and_validate_data())



