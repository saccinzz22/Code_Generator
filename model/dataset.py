import json
import torch # type: ignore
from torch.utils.data import Dataset # type: ignore

class CodingDataset(Dataset):
    def __init__(self, path, src_max_len, tgt_max_len, pad_idx=0):
        self.data=[]
        self.src_max_len=src_max_len
        self.tgt_max_len=tgt_max_len
        self.pad_idx=pad_idx
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        sample=self.data[idx]
        src=sample["input_ids"].copy()
        tgt=sample["target_ids"].copy()
        decoder_input=tgt[:-1]
        labels=tgt[1:]
        
        src=src+[self.pad_idx]*max(0, self.src_max_len - len(src))
        decoder_input=decoder_input+[self.pad_idx]*max(0, self.tgt_max_len - len(decoder_input))
        labels=labels+[self.pad_idx]*max(0, self.tgt_max_len - len(labels))
        
        src=src[:self.src_max_len]
        decoder_input=decoder_input[:self.tgt_max_len]
        labels=labels[:self.tgt_max_len]
        return {
            "src": torch.tensor(src, dtype=torch.long),
            "decoder_input": torch.tensor(decoder_input, dtype=torch.long), 
            "labels": torch.tensor(labels, dtype=torch.long)}