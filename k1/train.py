"""
Training script for the Southbag-K1-Frontier model.
"""

import os
import math
import urllib.request
import torch
from config import GPTConfig
from tokenizer import CharTokenizer
from model import GPT

# --- Hyperparameters for training ---
batch_size = 64 # How many independent sequences will we process in parallel?
max_iters = 2000
eval_interval = 200
learning_rate = 3e-4
min_lr = 3e-5         # Floor for cosine LR decay
warmup_iters = 100    # Linear warmup steps
weight_decay = 1e-1   # AdamW weight decay (regularization)
grad_clip = 1.0       # Gradient clipping norm
eval_iters = 50
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu' # Use CPU by default for the skeleton, or auto-detect
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

# 1. Download dataset if it doesn't exist
data_path = 'input.txt'
if not os.path.exists(data_path):
    print("Downloading tiny shakespeare dataset...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, data_path)

# 2. Load data and initialize tokenizer
with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Length of dataset in characters: {len(text)}")
tokenizer = CharTokenizer(text)
print(f"Vocabulary size: {tokenizer.vocab_size}")

# Encode the entire dataset and split into train/val
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # 90% for training, 10% for validation
train_data = data[:n]
val_data = data[n:]

# 3. Initialize model
config = GPTConfig(vocab_size=tokenizer.vocab_size)
model = GPT(config)
model.to(device)

print(f"Model initialized with {(sum(p.numel() for p in model.parameters())/1e6):.2f}M parameters.")

# 4. Data batching function
def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    # Random starting indices for the examples in the batch
    ix = torch.randint(len(data) - config.context_length, (batch_size,))
    x = torch.stack([data[i:i+config.context_length] for i in ix])
    y = torch.stack([data[i+1:i+config.context_length+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Function to estimate loss without gradients
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # Set to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train() # Set back to training mode
    return out

# 5. Create PyTorch optimizer (AdamW with weight decay for regularization)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning rate schedule: linear warmup then cosine decay to min_lr
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# 6. Training loop
print("Starting training...")
best_val_loss = float('inf')
for iter in range(max_iters):

    # Set learning rate for this iteration
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.6f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model(xb, yb)
    
    # Backpropagation
    optimizer.zero_grad(set_to_none=True) # Clear previous gradients
    loss.backward()                       # Compute gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) # Clip exploding gradients
    optimizer.step()                      # Update weights

print("Training complete!")

# 7. Generate some text with the trained model
from generate import generate_text

print("\n--- Generating Text ---")
# Start with a newline character
context = torch.tensor([tokenizer.encode("\n")], dtype=torch.long, device=device)
generated_tokens = generate_text(model, context, max_new_tokens=500)
print(tokenizer.decode(generated_tokens[0].tolist()))
print("-----------------------")

# Save full checkpoint (weights + vocab + config) so chat.py can reload it
checkpoint = {
    'model_state': model.state_dict(),
    'stoi': tokenizer.stoi,
    'itos': tokenizer.itos,
    'config': config,
}
torch.save(checkpoint, 'model.bin')
print("\nModel saved to model.bin")
