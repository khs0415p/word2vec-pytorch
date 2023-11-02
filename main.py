from config import Config
from load_data import load_data
from utils import make_vocab_file, split_data, set_seed
from spm_tokenizer import Tokenizer
from train import Trainer
import os
import argparse
import torch



def main(args):
    set_seed()
    mps_avalible = torch.backends.mps.is_available()
    device = torch.device("mps" if args.device == 'mps' and mps_avalible else "cpu")
    config = Config(args.config)
    config.model_path = 'save/' + config.model_name
    
    print("====================================")
    print(f"Device is {device}")
    print(f"Mode is {args.mode}")
    print(f"Model type is {config.model_type}")
    
    if not os.path.exists('save/'):
        os.makedirs('save/', exist_ok=True)
    
    load_data()
    if not os.path.exists(config.data_path + 'train'):
        split_data(config.corpus_load_path, config.data_path)
    
    tokenizer = Tokenizer(config)
    if not os.path.exists(config.corpus_save_path):
        make_vocab_file(config.corpus_load_path, config.corpus_save_path)        
        
    tokenizer.train()
    config.vocab_size = tokenizer.sp.vocab_size()
    print(f"Vocabulary size is {tokenizer.sp.vocab_size()}")    
    print("====================================")
    
    trainer = Trainer(args.mode, config, device, tokenizer)
    if args.mode == "train":
        trainer.train()
        
    else:
        print("test")
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, choices=["cpu", "mps"], default='mps')
    parser.add_argument("-m", "--mode", type=str, choices=["train", "test"], default='train')
    parser.add_argument("-c", '--config', type=str, default="config.json")
    args = parser.parse_args()
    
    main(args)