
import random
import numpy as np
import torch
from typing import Dict,Tuple
from datasets import load_dataset
import os
from types import SimpleNamespace
from utils.prompts import SYSTEM_PROMPT

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}"
    )

def get_data(tokenizer, num_proc=1, behavior='power-seeking', train=True, generation_prompt:bool = True):
    file_path = f"./data/{behavior}/{'train' if train else 'test'}.csv"
    dataset = load_dataset("csv", data_files=file_path, split='train')
    original_columns = dataset.column_names
    
    dataset = dataset.filter(
        lambda x: x["question"] is not None 
        and x["matching"] is not None 
        and x["not_matching"] is not None
    )
    
    def return_prompt_and_responses(samples):
        prompts = []
        for question in samples["question"]:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=generation_prompt
            )
            prompts.append(prompt)
            
        return {
            "prompt": prompts,
            "chosen": [s + tokenizer.eos_token for s in samples["matching"]],
            "rejected": [s + tokenizer.eos_token for s in samples["not_matching"]],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

def get_eval_data(tokenizer, behavior, system_prompt=SYSTEM_PROMPT, generation_prompt:bool = True):
    path = f"./data/{behavior}/test_infer.csv"
    if not os.path.exists(path):
         raise FileNotFoundError(f"Data file not found: {path}")
         
    dataset = load_dataset("csv", data_files=path, split='train')
    
    questions = [] 
    prompts = []  
    labels = []    
    
    for row in dataset:
        if row['question'] is not None:  
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row['question']},
            ]
            full_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=generation_prompt
            )
            
            questions.append(full_prompt)
            
            current_options = [row[col] for col in ['A','B','C','D'] if col in row and row[col] is not None]
            prompts.append(current_options)
            labels.append(row['matching'])

    return{
        'questions':questions,
        'prompts':prompts,
        'labels':labels,
    }

def get_sterbench_data(tokenizer, behavior, system_prompt=SYSTEM_PROMPT, generation_prompt:bool = True):
    path = f"./data/{behavior}/test_infer.csv"
    if not os.path.exists(path):
         raise FileNotFoundError(f"Data file not found: {path}")
         
    dataset = load_dataset("csv", data_files=path, split='train')
    
    questions = [] 
    prompts = []  
    labels = []    
    communities=[]
    for row in dataset:
        if row['question'] is not None:  
            messages = [
                {"role": "system", "content": 'Answer the given Question'},
                {"role": "user", "content": row['question']},
            ]
            full_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=generation_prompt
            )
            
            questions.append(full_prompt)
            
            current_options = [row[col] for col in ['A','B','C','D'] if col in row and row[col] is not None]
            prompts.append(current_options)
            labels.append(row['matching'])
            communities.append(row['communities'])

    return{
        'questions':questions,
        'prompts':prompts,
        'labels':labels,
        'communities':communities,
    }

def batch_logps(logits: torch.Tensor, ids: torch.Tensor, pad_id: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.shape[:-1] != ids.shape:
        raise ValueError("Logits and ids must have the same shape. (batch,sequence_length,dim)")

    ids = ids.clone()
    ids = ids[:, 1:].contiguous()
    logits = logits[:, :-1, :].contiguous()

    loss_mask = None
    if pad_id is not None:
        loss_mask = ids != pad_id
        ids[ids == pad_id] = 0
        
    token_logps = torch.gather(logits.log_softmax(-1), dim=-1, index=ids.unsqueeze(-1)).squeeze(-1)
    return token_logps, loss_mask



