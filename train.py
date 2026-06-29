"""
Training script for the Tiny GPT model.
"""

import os
import urllib.request
import torch
from config import GPTConfig
from tokenizer import CharTokenizer
from model import GPT

# --- Hyperparameters for training ---
batch_size = 32 # How many independent sequences will we process in parallel?
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
eval_iters = 200
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

# 5. Create PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 6. Training loop
print("Starting training...")
for iter in range(max_iters):

    # Every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model(xb, yb)
    
    # Backpropagation
    optimizer.zero_grad(set_to_none=True) # Clear previous gradients
    loss.backward()                       # Compute gradients
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

# Optionally save the model
torch.save(model.state_dict(), 'tiny_gpt.pt')
print("\nModel saved to tiny_gpt.pt")
