import torch
from torch.utils.data import DataLoader
from src.data.preprocessing import load_and_validate_data
from src.data.dataset import MoleculeDataset
from src.data.tokenizer import SMILESTokenizer
from src.models.transformer import MoleculeTransformer
from src.training.trainer import MoleculeTrainer
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

def main():
    # Hyperparameters
    model_params = {
        'd_model': 256,        # Embedding dimension
        'nhead': 8,           # Number of attention heads
        'num_layers': 3,      # Number of transformer layers
        'dim_feedforward': 512,# Feedforward network size
        'dropout': 0.1        # Dropout rate
    }
    
    training_params = {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'num_epochs': 10
    }

    # 1. Load and preprocess data
    df = load_and_validate_data()
    
    # 2. Create tokenizer and dataset
    tokenizer = SMILESTokenizer()
    
    # Split data into train and validation
    train_smiles, val_smiles, train_sol, val_sol = train_test_split(
        df['smiles'].tolist(),
        df['solubility'].tolist(),
        test_size=0.2,
        random_state=42
    )
    
    # Create two datasets
    train_dataset = MoleculeDataset(train_smiles, train_sol, tokenizer)
    val_dataset = MoleculeDataset(val_smiles, val_sol, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_params['batch_size'])
    
    # 4. Initialize model with params
    model = MoleculeTransformer(
        vocab_size=len(tokenizer.token_to_id),
        **model_params
    )
    
    # 5. Create trainer with training params
    trainer = MoleculeTrainer(
        model,
        learning_rate=training_params['learning_rate'],
        weight_decay=training_params['weight_decay']
    )
    
    # Create save directory
    save_dir = Path('models/saved')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop with validation
    best_val_loss = float('inf')
    for epoch in range(training_params['num_epochs']):
        # Train
        train_loss = trainer.train_epoch(train_loader)
        
        # Validate
        val_loss = trainer.validate(val_loader)
        
        print(f"Epoch {epoch+1}/{training_params['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / 'best_model.pt')
            print(f"Saved new best model with validation loss: {val_loss:.4f}")
        
        print("-" * 50)

if __name__ == "__main__":
    main() 