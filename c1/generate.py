"""
Autoregressive text generation using the trained (or randomly initialized) GPT model.
"""

import torch
from torch.nn import functional as F
from model import GPT
from config import GPTConfig
from tokenizer import CharTokenizer

def generate_text(model: GPT, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None, greedy: bool = False):
    """
    Generates new tokens autoregressively.
    
    Args:
        model: The GPT model instance.
        idx: Initial context as a tensor of shape (Batch, Time).
        max_new_tokens: Number of tokens to generate.
        temperature: Controls randomness (lower = more deterministic, higher = more random).
        top_k: If set, only sample from the top-k most likely tokens.
        greedy: If True, always picks the single most likely token (ignores temperature and top_k).
        
    Returns:
        Tensor containing the original context plus generated tokens.
    """
    model.eval() # Set model to evaluation mode
    
    for _ in range(max_new_tokens):
        # Crop the context to the maximum block size (context_length)
        idx_cond = idx[:, -model.config.context_length:]
        
        # Get the predictions from the model
        with torch.no_grad():
            logits, _ = model(idx_cond)
        
        # Focus only on the last time step prediction
        logits = logits[:, -1, :] # (B, vocab_size)
        
        if greedy:
            # Greedy decoding: simply take the argmax
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # Apply temperature scaling
            logits = logits / temperature
            
            # Optionally apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            
            # Sample from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
    return idx

if __name__ == "__main__":
    # Example usage (with an untrained model)
    print("Initializing dummy config and model...")
    config = GPTConfig()
    
    # We need a dummy vocabulary just for the example to work
    # In a real scenario, this would be built from a training dataset
    dummy_text = "abcdefghijklmnopqrstuvwxyz \n.,!?" 
    tokenizer = CharTokenizer(dummy_text)
    
    # Overwrite vocab size in config to match our dummy tokenizer
    config.vocab_size = tokenizer.vocab_size
    # Initialize model
    model = GPT(config)
    
    # Start generation with a single space token
    context = "hello"
    # Encode context to tensor, if character is not in dummy_text, it'll be ignored.
    encoded_context = tokenizer.encode(context)
    if not encoded_context:
        # Fallback if empty
        encoded_context = [tokenizer.encode(" ")[0]]
        
    context_tensor = torch.tensor([encoded_context], dtype=torch.long)
    
    print(f"Generating 100 tokens with greedy=False (model is randomly initialized, output will be gibberish)...")
    out_tokens = generate_text(model, context_tensor, max_new_tokens=100, temperature=0.8, top_k=10)
    
    generated_text = tokenizer.decode(out_tokens[0].tolist())
    print("\n--- Output ---")
    print(generated_text)
    print("--------------")
