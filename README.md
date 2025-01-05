# Identifying Molecular Solubility Using Transformers
This project implements a transformer-based model for predicting molecular solubility, a key property in drug discovery applications. The model processes SMILES string representations of molecules using self-attention mechanisms to identify patterns that correlate with solubility values. We demonstrate that transformer architectures can effectively predict molecular properties using only SMILES sequence information.

## Technical Implementation

We utilize the ESOL dataset from MoleculeNet, and preprocess each entry using RDKit to validate that each SMILES string is valid. We then tokenize the molecular representations using the tokenizer implemented in "src/data/tokenizer.py", where the SMILES characters are converted into numerical tokens. 

The transformer architecture contains the following sequential components:
1. An embedding layer that maps tokenized inputs to d_model dimensional vectors (d_model=256)
2. Positional encoding to maintain sequential information
3. A transformer encoder stack (3 layers) implementing multi-head self-attention (8 heads) as described in "Attention is All You Need" (Vaswer et al., 2017)
4. Mean pooling across the sequence dimension to create a fixed-size representation
5. A final linear layer that gives a single value for solubility prediction

## Project Structure

```
src/
├── data/        # Data processing and dataset creation
│   ├── dataset.py      
│   ├── preprocessing.py 
│   ├── tokenizer.py    
│   └── test_pipeline.py
├── models/      # Model architecture
│   └── transformer.py
└── training/    # Training and evaluation scripts
    ├── train.py
    ├── test.py
    └── trainer.py
```

## Training

The ESOL dataset was split into an 80-20 train-test split. We utilize an AdamW optmizer with weight decay, and use MSE loss for regression. The training used the following: 
- Optimizer: AdamW with weight decay of 0.01
- Loss function: Mean Squared Error (MSE)
- Epochs: 10
- Batch size: 32
- Learning rate: 1e-4

The model was evaluated after each epoch, and the best performing model (based on validation loss) is saved at "models/saved/best_model.pt". 

## Testing & Results

The testing produced strong results: 
- R² Score: 0.8445
- Mean Absolute Error: 0.2873
- Root Mean Squared Error: 0.4090
showing that the model can predict the solubility of molecules with good accuracy. Undoubtedly, with more serious training these results would only improve.  

This work contributes to the growing evidence that self-attention mechanisms can successfully capture the relationships and properties of molecular structure. These applications of deep learning will have huge impacts on the future of molecular biology and drug discovery. 
