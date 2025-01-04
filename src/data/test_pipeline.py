import pandas as pd
from torch.utils.data import DataLoader
from preprocessing import load_and_validate_data
from tokenizer import SMILESTokenizer
from dataset import MoleculeDataset

def test_pipeline():
    # 1. Load and preprocess data
    print("Loading data...")
    df = load_and_validate_data()
    
    # 2. Create tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = SMILESTokenizer()
    
    # 3. Create dataset
    print("\nCreating dataset...")
    dataset = MoleculeDataset(
        smiles_list=df['smiles'].tolist(),
        solubility_values=df['solubility'].tolist(),
        tokenizer=tokenizer
    )
    
    # 4. Create dataloader
    print("\nCreating dataloader...")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 5. Test by looking at one batch
    print("\nTesting batch:")
    batch = next(iter(dataloader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Input shape: {batch['input_ids'].shape}")
    print(f"Target shape: {batch['target'].shape}")
    
    # 6. Decode a few tokens from the first sequence to verify tokenization
    first_seq = batch['input_ids'][0]
    print("\nFirst few tokens of first sequence:")
    tokens = [tokenizer.id_to_token[id.item()] for id in first_seq[:10]]
    print(tokens)

if __name__ == "__main__":
    test_pipeline()
