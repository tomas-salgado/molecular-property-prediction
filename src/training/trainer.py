import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

class MoleculeTrainer():
    def __init__(self, 
                 model, 
                 learning_rate=1e-4, 
                 weight_decay=0.01):
        self.model = model
        self.criterion = nn.MSELoss()
        
        # Updated optimizer to AdamW with weight decay
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)  # Default betas often work well
        )
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            inputs = batch['input_ids']
            targets = batch['target']
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader) 

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input_ids']
                targets = batch['target']
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
        return total_loss / len(dataloader) 