from typing import List, Dict
import torch
from torch.utils.data import Dataset

class PairedCAADataset(torch.utils.data.Dataset):
    def __init__(self, data, mode="chosen"):
        self.data = data
        self.mode = mode  # "chosen" or "rejected"

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        # Return a dictionary so collate_fn can access these separately
        return {
            "prompt": ex["prompt"],
            "response": ex[self.mode]  # dynamically fetches 'chosen' or 'rejected'
        }
    
class MultipleOptionDataset(Dataset):
    def __init__(self, tokenizer, prompts: List[List[str]], questions: List[str], labels: List[str]):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompts = prompts   
        self.questions = questions 
        self.labels = labels
        
        self.eos_token = self.tokenizer.eos_token if self.tokenizer.eos_token else "<end_of_turn>"

    def __getitem__(self, index: int):
        context_str = self.questions[index]
       
        tokenized_question = self.tokenizer(
            context_str, 
            return_tensors='pt', 
            add_special_tokens=False
        )
        question_len = tokenized_question.input_ids.shape[1]

        tokenized_row_ids = []
        tokenized_row_mask = []

        
        directions = []
        for i, option in enumerate(self.prompts[index]):
            full_text = f"{context_str}{option}{self.eos_token}"
            
            tok = self.tokenizer(
                full_text, 
                return_tensors='pt', 
                add_special_tokens=False
            )
            
            tokenized_row_ids.append(tok.input_ids.squeeze(0))
            tokenized_row_mask.append(tok.attention_mask.squeeze(0))
            
            # Map index 0 to Direction 1, and index 1 to Direction -1
            # (Adjust this math if you actually want 0 and 1!)
            dir_value = 1 if i == 0 else -1 
            directions.append(dir_value)

        decoded = [self.tokenizer.convert_ids_to_tokens(ids) for ids in tokenized_row_ids]
        return {
            "question_length": question_len,
            "input_ids": tokenized_row_ids,
            "attention_mask": tokenized_row_mask,
            "label": self.labels[index],
            'd':directions,
            "decode": decoded
        }
    
    def __len__(self) -> int:
        return len(self.prompts)

    
class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, i):
        return self.prompts[i]
    
class SteerbenchDataset(Dataset):
    def __init__(self, tokenizer, prompts: List[List[str]], questions: List[str], labels: List[str], communities: List[str]):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompts = prompts   
        self.questions = questions 
        self.communities = communities
        self.labels = labels
        
        self.eos_token = self.tokenizer.eos_token if self.tokenizer.eos_token else "<end_of_turn>"

    def __getitem__(self, index: int):
        context_str = self.questions[index]
       
        tokenized_question = self.tokenizer(
            context_str, 
            return_tensors='pt', 
            add_special_tokens=False
        )
        question_len = tokenized_question.input_ids.shape[1]

        tokenized_row_ids = []
        tokenized_row_mask = []

        
        for i, option in enumerate(self.prompts[index]):
            full_text = f"{context_str}{option}{self.eos_token}"
            
            tok = self.tokenizer(
                full_text, 
                return_tensors='pt', 
                add_special_tokens=False
            )
            
            tokenized_row_ids.append(tok.input_ids.squeeze(0))
            tokenized_row_mask.append(tok.attention_mask.squeeze(0))
            
        
            
        return {
            "question_length": question_len,
            "input_ids": tokenized_row_ids,
            "attention_mask": tokenized_row_mask,
            "label": self.labels[index],
            "communities":self.communities[index]
        }
    
    def __len__(self) -> int:
        return len(self.prompts)
