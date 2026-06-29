import numpy as np
import json
import sys

def softmax(x):
    shift_x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shift_x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm(x, weight, bias, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    out = (x - mean) / np.sqrt(var + eps)
    return out * weight + bias

class NumpyTokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, 'r') as f:
            data = json.load(f)
        self.stoi = data['stoi']
        self.itos = data['itos']
        self.vocab_size = data['vocab_size']
        
    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]
        
    def decode(self, l):
        return ''.join([self.itos[str(i)] for i in l if str(i) in self.itos])

class NumpyGPT:
    def __init__(self, npz_path, n_layer=4, n_head=4, n_embd=128):
        self.weights = np.load(npz_path)
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        
    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.weights['token_embedding_table.weight'][idx]
        pos_emb = self.weights['position_embedding_table.weight'][:T]
        x = tok_emb + pos_emb
        
        for i in range(self.n_layer):
            x_ln1 = layer_norm(x, self.weights[f'blocks.{i}.ln1.weight'], self.weights[f'blocks.{i}.ln1.bias'])
            
            sa_out = []
            for h in range(self.n_head):
                q_w = self.weights[f'blocks.{i}.sa.heads.{h}.query.weight'].T
                k_w = self.weights[f'blocks.{i}.sa.heads.{h}.key.weight'].T
                v_w = self.weights[f'blocks.{i}.sa.heads.{h}.value.weight'].T
                
                q = x_ln1 @ q_w
                k = x_ln1 @ k_w
                v = x_ln1 @ v_w
                
                wei = q @ k.transpose(0, 2, 1) * (q_w.shape[-1] ** -0.5)
                tril = np.tril(np.ones((T, T)))
                wei = np.where(tril == 0, -np.inf, wei)
                wei = softmax(wei)
                sa_out.append(wei @ v)
                
            sa_out = np.concatenate(sa_out, axis=-1)
            proj_w = self.weights[f'blocks.{i}.sa.proj.weight'].T
            proj_b = self.weights[f'blocks.{i}.sa.proj.bias']
            x = x + (sa_out @ proj_w + proj_b)
            
            x_ln2 = layer_norm(x, self.weights[f'blocks.{i}.ln2.weight'], self.weights[f'blocks.{i}.ln2.bias'])
            ff_w1 = self.weights[f'blocks.{i}.ffwd.net.0.weight'].T
            ff_b1 = self.weights[f'blocks.{i}.ffwd.net.0.bias']
            ff_w2 = self.weights[f'blocks.{i}.ffwd.net.2.weight'].T
            ff_b2 = self.weights[f'blocks.{i}.ffwd.net.2.bias']
            
            ff = np.maximum(0, x_ln2 @ ff_w1 + ff_b1)
            ff = ff @ ff_w2 + ff_b2
            x = x + ff
            
        x = layer_norm(x, self.weights['ln_f.weight'], self.weights['ln_f.bias'])
        logits = x @ self.weights['lm_head.weight'].T + self.weights['lm_head.bias']
        return logits

def generate(model, tokenizer, prompt, max_new_tokens=100, context_length=128, temperature=1.0):
    idx = np.array([tokenizer.encode(prompt)])
    if idx.size == 0:
        idx = np.array([[tokenizer.encode(" ")[0]]])
        
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        logits = model.forward(idx_cond)
        logits = logits[:, -1, :]
        
        # Temperature scaling
        logits = logits / temperature
        probs = softmax(logits)
        
        # Sample
        idx_next = np.random.choice(tokenizer.vocab_size, p=probs[0])
        idx = np.concatenate([idx, [[idx_next]]], axis=1)
        
    return tokenizer.decode(idx[0].tolist())

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python numpy_inference.py <model_dir> <prompt>")
        sys.exit(1)
        
    model_dir = sys.argv[1]
    prompt = sys.argv[2]
    
    print(f"Loading model from {model_dir}...")
    tokenizer = NumpyTokenizer(f"{model_dir}/vocab.json")
    model = NumpyGPT(f"{model_dir}/model.npz")
    
    print(f"Generating for prompt: '{prompt}'...")
    out = generate(model, tokenizer, prompt, max_new_tokens=150, temperature=0.8)
    print("\n--- Output ---")
    print(out)
    print("--------------")
