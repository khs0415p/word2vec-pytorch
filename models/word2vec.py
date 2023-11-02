import torch.nn as nn
import torch.nn.functional as F
import torch


class CBOW(nn.Module):
    def __init__(self, config) -> None:
        super(CBOW, self).__init__()
        
        # 2 layer
        self.vocab_size = config.vocab_size
        self.emb_size = config.emb_size
        self.init_range = config.init_range
        
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.output_layer = nn.Linear(self.emb_size, self.vocab_size)
    
    def init_emb(self):
        self.embedding.weight.data.uniform_(-self.init_range, self.init_range)
        
        
    def count_param(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
    def get_embedding(self):
        return self.embedding.weight.data
    
    
    def forward(self, contexts):    
        emb_output = self.embedding(contexts)
        sum_output = torch.sum(emb_output, dim=1)
        output = self.output_layer(sum_output)
        return output
    


class SkipGram(nn.Module):
    def __init__(self, config) -> None:
        super(SkipGram, self).__init__()
        
        self.config = config
        self.vocab_size = config.vocab_size
        self.emb_size = config.emb_size
        
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.output_layer = nn.Linear(self.emb_size, self.vocab_size)


    def init_emb(self):
        self.embedding.weight.data.uniform_(-self.init_range, self.init_range)
        

    def forward(self, contexts):
        # [batch, emb_dim]
        emb_output = self.embedding(contexts)
        # [batch, vocab_size]
        output = self.output_layer(emb_output)
        return output
        