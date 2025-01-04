import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pathlib import Path
from src.models.transformer import MoleculeTransformer
from src.data.preprocessing import load_and_validate_data
from src.data.dataset import MoleculeDataset
from src.data.tokenizer import SMILESTokenizer
from sklearn.model_selection import train_test_split

def test_model(model, test_loader, device='cpu'):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input_ids']
            targets = batch['target']
            
            outputs = model(inputs)
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    r2 = r2_score(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
    return {
        'R2': r2,
        'MAE': mae,
        'RMSE': rmse,
        'predictions': predictions,
        'actuals': actuals
    }

def main():
    # Load data
    df = load_and_validate_data()
    
    # Create tokenizer and build vocab with ALL data first
    tokenizer = SMILESTokenizer()
    tokenizer.build_vocabulary(df['smiles'].tolist())  # Build vocab with all data
    
    # Then split the data
    train_smiles, val_smiles, train_sol, val_sol = train_test_split(
        df['smiles'].tolist(),
        df['solubility'].tolist(),
        test_size=0.2,
        random_state=42
    )
    
    # Create validation dataset
    val_dataset = MoleculeDataset(val_smiles, val_sol, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Load model architecture
    model = MoleculeTransformer(
        vocab_size=len(tokenizer.token_to_id),
        d_model=256
    )
    
    # Load best weights
    model.load_state_dict(torch.load('models/saved/best_model.pt'))
    
    # Test
    results = test_model(model, val_loader)
    
    print("\nTest Results:")
    print(f"R² Score: {results['R2']:.4f}")
    print(f"Mean Absolute Error: {results['MAE']:.4f}")
    print(f"Root Mean Squared Error: {results['RMSE']:.4f}")

if __name__ == "__main__":
    main() 