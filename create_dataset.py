import torch
from torch.utils.data import Dataset
from typing import Iterable, List, Sequence
from tokenizer import to_chat_text



class TextDataset(Dataset):
    def __init__(self,
                 texts,
                 tokenizer,
                 *,
                 seq_len : int = 1024,
                 add_eos : bool = True,
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.seq_len = seq_len
                
        # texts = [to_chat_text(text) for text in texts]
        
        if add_eos and tokenizer.eos_token:
            texts = [t + tokenizer.eos_token for t in texts]
            
        ids : List[List[int]] = tokenizer(
            list(texts), add_special_tokens=False, return_attention_mask=False)["input_ids"]

        flat_ids = [token for seq in ids for token in seq]
        total_len = (len(flat_ids) // seq_len) * seq_len
        
        if total_len == 0 :
            raise ValueError("Corpus is shorter than a single seq_len window")
        
        flat_ids = flat_ids[:total_len]
        self.input_ids = torch.tensor(flat_ids,dtype=torch.long).view(-1,seq_len)
        
    def __len__(self) -> int:
        return self.input_ids.size(0)
    
    def __getitem__(self, index):
        
        x = self.input_ids[index]
        return {"input_ids" :x , "labels" : x.clone()}