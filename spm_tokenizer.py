import sentencepiece as spm
import os


class Tokenizer:
    def __init__(self, config) -> None:
        self.config = config
        self.path = self.config.corpus_save_path
        self.prefix = self.config.corpus_prefix
        self.vocab_size = self.config.vocab_size
        self.model_path = self.prefix + '.model'
        
        if not os.path.exists(self.model_path[:self.model_path.rfind('/')]):
            os.makedirs(self.model_path[:self.model_path.rfind('/')], exist_ok=True)
    
    
    def load(self):
        sp = spm.SentencePieceProcessor()
        sp.Load(self.model_path)
        self.sp = sp


    def train(self):
        
        if os.path.exists(self.model_path):
            self.load()
            return
        
        spm.SentencePieceTrainer.Train(
            f'--input={self.path} --model_prefix={self.prefix} --vocab_size={self.vocab_size}'
            '--model_type=bpe' +
            ' --character_coverage=1.0' +
            ' --shuffle_input_sentence=true' +
            ' --max_sentence_length=999999' +
            ' --pad_id=0 --pad_piece=[PAD]' +
            ' --unk_id=1 --unk_piece=[UNK]' +
            ' --bos_id=2 --bos_piece=[BOS]' +
            ' --eos_id=3 --eos_piece=[EOS]'
            )
        
        self.load()
        
    
    def vocab_size(self):
        return self.sp.vocab_size()
    
    
    def tokenize(self, sequence):
        return self.sp.EncodeAsPieces(sequence)
    
    
    def encode_to_ids(self, sequence):
        return self.sp.EncodeAsIds(sequence)
    
    
    def decode(self, sequence):
        return self.sp.Decode(sequence)
        
        
if __name__ == "__main__":
    tokenizer = Tokenizer('data/ratings.txt')
    print(tokenizer.wtoi)
    print(tokenizer.vocab_size)
    