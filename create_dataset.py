import torch
from torch.utils.data import Dataset
from typing import Iterable, List, Sequence
from tokenizer import to_chat_text



class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, *, seq_len=1024, stride=1, add_eos=True):
        samples = []
        for t in texts:
            if add_eos and tokenizer.eos_token:
                t += tokenizer.eos_token
            ids = tokenizer(t, add_special_tokens=False)["input_ids"]

            # sliding window
            for start in range(0, len(ids) - stride, stride):
                window = ids[start : start + seq_len]
                if len(window) < seq_len:
                    break                           # last short chunk is dropped
                samples.append(torch.tensor(window, dtype=torch.long))

        self.input_ids = torch.stack(samples)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        x = self.input_ids[idx]
        return {"input_ids": x, "labels": x.clone()}