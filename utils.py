from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import torch
import random



def make_vocab_file(path, out_path):
    with open(path, 'r') as f, open(out_path, 'w') as of:
        while line := f.readline():
            try:
                _, doc, _ = line.split('\t')
                of.write(doc)
                of.write("\n")
                
            except:
                pass
            
            
def split_data(path, out_path):
    df = pd.read_csv(path, sep='\t')
    data = df['document'].dropna().to_list()
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    
    with open(out_path + 'train', 'wb') as f:
        pickle.dump(train, f)

    with open(out_path + 'val', 'wb') as f:
        pickle.dump(test, f)
    
    
def save_checkpoint(path, model, optimizer):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, path)
    print("Saving model...")
    
    
def set_seed(seed=42):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  random.seed(seed)
    