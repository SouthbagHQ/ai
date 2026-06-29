import torch
from config import GPTConfig
from tokenizer import CharTokenizer
from model import GPT

# Create dataset: the word "cheese " repeated 500 times
text = "cheese " * 500
tokenizer = CharTokenizer(text)
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

# Train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

config = GPTConfig(vocab_size=tokenizer.vocab_size)
model = GPT(config)

batch_size = 32
max_iters = 1000
learning_rate = 1e-3

# Auto-detect device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - config.context_length, (batch_size,))
    x = torch.stack([data_split[i:i+config.context_length] for i in ix])
    y = torch.stack([data_split[i+1:i+config.context_length+1] for i in ix])
    return x.to(device), y.to(device)

print("Starting training on 'cheese'...")
for iter in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if iter % 200 == 0:
        print(f"Step {iter}: Loss {loss.item():.4f}")

print(f"Final Loss: {loss.item():.4f}")

# Export as binary PyTorch state dict
out_path = 'tiny_gpt.bin'
torch.save(model.state_dict(), out_path)
print(f"Model exported to {out_path}")

# Test generation to prove it works
from generate import generate_text
print("\n--- Generating Text ---")
context = torch.tensor([tokenizer.encode("c")], dtype=torch.long, device=device)
generated_tokens = generate_text(model, context, max_new_tokens=50)
print(tokenizer.decode(generated_tokens[0].tolist()))
print("-----------------------")
