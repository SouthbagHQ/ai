import os
import torch
import numpy as np
import json
import sys

def export(model_dir, pt_file):
    sys.path.insert(0, model_dir)
    from model import GPT
    from config import GPTConfig
    from tokenizer import CharTokenizer
    
    print(f"Exporting {model_dir}...")
    
    # Reconstruct tokenizer to get the exact vocab
    with open(os.path.join(model_dir, 'input.txt'), 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer = CharTokenizer(text)
    
    # Save vocab
    vocab_data = {
        'stoi': tokenizer.stoi,
        'itos': tokenizer.itos,
        'vocab_size': tokenizer.vocab_size
    }
    with open(os.path.join(model_dir, 'vocab.json'), 'w') as f:
        json.dump(vocab_data, f)
        
    config = GPTConfig(vocab_size=tokenizer.vocab_size)
    model = GPT(config)
    
    pt_path = os.path.join(model_dir, pt_file)
    if os.path.exists(pt_path):
        model.load_state_dict(torch.load(pt_path, map_location='cpu'))
    else:
        print(f"Warning: {pt_path} not found. Exporting randomly initialized weights.")
        
    # Export weights to dict
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().numpy()
        
    np.savez(os.path.join(model_dir, 'model.npz'), **weights)
    print(f"Exported {model_dir} to NumPy and vocab.json")
    sys.path.pop(0)

export('k1', 'tiny_gpt.pt')
export('c1', 'tiny_gpt.bin')
