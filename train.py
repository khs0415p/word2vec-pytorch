from models.word2vec import CBOW, SkipGram
from torch.utils.data import DataLoader
from dataset import CBOWDataset, SkipGramDataset
from utils import save_checkpoint
import torch.nn as nn 
import torch.optim as optim
import torch
import time
import wandb

# random 요소 설정.
# scheduler 설정



class Trainer:
    def __init__(self, mode, config, device, tokenizer) -> None:
        self.config = config
        self.model_path = self.config.model_path
        self.device = device
        self.mode = mode
        self.tokenizer = tokenizer
        self.epochs = config.epochs
        
        
        if self.config.model_type == 'cbow':
            self.model = CBOW(self.config).to(self.device)
            
        elif self.config.model_type == 'skip':
            self.model = SkipGram(self.config).to(self.device)
            
        else:
            raise Exception('the model type must be selected from cbow or skip.')
        
        
        if mode == 'train':
            wandb.init(
                        entity='hyunsooo',
                        project="word2vec",
                        name=self.config.model_type,
                        config = {
                            "architecture": "word2vec-cbow",
                            "dataset": "naver-review",
                            "epochs" : self.config.epochs,
                            "model_type": self.config.model_type
                        }
                    )
            
            self.dataloaders = {}
            for name in ['train', 'val']:
                if self.config.model_type == 'cbow':
                    self.dataloaders[name] = DataLoader(CBOWDataset(self.config,
                                                                    self.config.data_path + name, self.tokenizer),
                                                                    batch_size=self.config.batch_size,
                                                                    shuffle = True if name == 'train' else False)
                    
                else:
                    self.dataloaders[name] = DataLoader(SkipGramDataset(self.config,
                                                                    self.config.data_path + name, self.tokenizer),
                                                                    batch_size=self.config.batch_size,
                                                                    shuffle = True if name == 'train' else False)
                    
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.config.pad_id)
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr, betas=(0.9, 0.999))
        
    
    def train(self):
        
        best_val_loss = float("inf")
        for epoch in range(self.epochs):
            start = time.time()
            print("====================================")
            print(f"Epoch : {epoch+1}/{self.epochs}")
            
            for name in ['train', 'val']:
                print(f"State : {name}\n")
                if name == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                
                total_loss = 0
                for i, (x, y) in enumerate(self.dataloaders[name]):
                    x, y = x.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(name == 'train'):
                        output = self.model(x)
                        loss = self.criterion(output, y)
                        total_loss += loss.item()
                        
                        wandb.log({f"{name}_loss": loss.item()})
                        if name == 'train':
                            loss.backward()
                            self.optimizer.step()
                    
                    if i % self.config.iter_point == 0:
                        print(f"Step {i}/{len(self.dataloaders[name])}\nLoss : {loss.item():.4f}\n")
                    
                epoch_loss = total_loss / len(self.dataloaders[name])
                print(f"{name} loss : {epoch_loss:.4f}")
                if name == 'val' and best_val_loss > epoch_loss:
                    best_val_loss = epoch_loss
                    save_checkpoint(self.model_path, self.model, self.optimizer)
                print("====================================")
                    
            end = time.time()
            print(f"Time : {end - start:.4f}")
            print("====================================")
        wandb.finish()