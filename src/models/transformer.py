import torch
import torch.nn as nn
import math

class MoleculeTransformer(nn.Module):
    def __init__(self, 
                 vocab_size,
                 d_model=256,
                 nhead=8,
                 num_layers=3,
                 dim_feedforward=512,
                 dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # 1. Token embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Add transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,          # Same dimension as embeddings
            nhead=nhead,              # Number of attention heads
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Final output layer for solubility prediction
        self.output_layer = nn.Linear(d_model, 1)  # 1 output for regression
        
    def forward(self, x):
        # 1. Convert tokens to embeddings
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # 2. Add positional encoding
        x = self.pos_encoder(x)
        
        # 3. Run through transformer
        transformer_output = self.transformer_encoder(x)
        
        # 4. Use [CLS] token or mean pooling for prediction
        sequence_output = transformer_output.mean(dim=1)  # mean pooling
        
        # 5. Final prediction
        output = self.output_layer(sequence_output)
        return output.view(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
