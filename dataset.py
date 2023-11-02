from torch.utils.data import Dataset
import pandas as pd
import pickle
import torch

class CBOWDataset(Dataset):
    def __init__(self, config, data_path, tokenizer) -> None:
        super(CBOWDataset, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.window_size = self.config.window_size
        self.data_path = data_path
        
        self.data = []
        with open(self.data_path, 'rb') as f:
            docs = pickle.load(f)
            for sentence in docs:
                sentence = self.tokenizer.encode_to_ids(sentence)
                if len(sentence) >= self.window_size:
                    for i in range(len(sentence) - self.window_size + 1):
                        self.data.append(sentence[i: i+self.window_size])
        
        
    def __getitem__(self, index):
        if self.window_size == 3:
            return torch.LongTensor([self.data[index][0], self.data[index][2]]), torch.tensor(self.data[index][1])
                
        
        elif self.window_size == 5:
            return torch.LongTensor([self.data[index][0], self.data[index][1], self.data[index][3], self.data[index][4]]), torch.tensor(self.data[index][2])
        
        else:
            raise Exception("window size must be 3 or 5.")
    
    
    def __len__(self):
        return len(self.data)
    

class SkipGramDataset(Dataset):
    def __init__(self, config, data_path, tokenizer) -> None:
        super(SkipGramDataset, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.window_size = self.config.window_size
        self.center_loc = self.window_size >> 1
        self.data_path = data_path
        
        self.data = []
        with open(self.data_path, 'rb') as f:
            docs = pickle.load(f)
            for sentence in docs:
                sentence = self.tokenizer.encode_to_ids(sentence)
                if len(sentence) >= self.window_size:
                    for i in range(len(sentence) - self.window_size + 1):
                        center = sentence[i + self.center_loc]
                        contexts = sentence[i: i+self.center_loc] + sentence[i+self.center_loc+1:i+self.window_size]
                        for context in contexts:
                            self.data.append([center, context])
        
        
    def __getitem__(self, index):
        return torch.tensor(self.data[index][0]), torch.tensor(self.data[index][1])
    
    
    def __len__(self):
        return len(self.data)
    
    
if __name__ == "__main__":
    from spm_tokenizer import Tokenizer
    from config import Config
    from torch.utils.data import DataLoader
    
    config = Config("config.json")
    tokenizer = Tokenizer(config)
    tokenizer.load()
    dataset = CBOWDataset(config, tokenizer)
    dl = DataLoader(dataset, 3)
    for a in dl:
        t, v = a
        print(t)
        print(t.size())
        print(v)
        print(v.size())
        break