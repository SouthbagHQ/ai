"""
The GPT architecture implementation from scratch.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from config import GPTConfig

class Head(nn.Module):
    """ One head of self-attention """

    def __init__(self, config: GPTConfig, head_size: int):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        # register_buffer ensures the tril tensor is not treated as a trainable parameter
        self.register_buffer('tril', torch.tril(torch.ones(config.context_length, config.context_length)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # Compute attention scores ("affinities")
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5) 
        
        # Mask out future tokens to ensure autoregressive property (tokens can only attend to past tokens)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Aggregate the values based on attention scores
        v = self.value(x) # (B, T, head_size)
        out = wei @ v     # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """

    def __init__(self, config: GPTConfig):
        super().__init__()
        head_size = config.n_embd // config.n_head
        # ModuleList holds our multiple attention heads
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Concatenate results from all heads along the feature dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Apply projection to mix the head outputs
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd), # Expand representation (standard is 4x)
            nn.ReLU(),                                   # Activation function
            nn.Linear(4 * config.n_embd, config.n_embd), # Project back to embedding dimension
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication (attention) followed by computation (feed-forward) """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        # Apply layer normalization BEFORE the sub-layers (Pre-LN architecture)
        # Residual connections (x + ...) help with training deep networks by providing a clear gradient path
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    """ The full GPT Language Model """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Token embedding table
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        # Positional embedding table encodes the position of each token in the sequence
        self.position_embedding_table = nn.Embedding(config.context_length, config.n_embd)
        
        # Stack of Transformer blocks
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd) 
        # Final linear layer to project back to vocabulary size
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        
        # Create positional embeddings for sequence length T
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, n_embd)
        
        # Add token and position embeddings together
        x = tok_emb + pos_emb # (B, T, n_embd)
        
        # Pass through transformer blocks
        x = self.blocks(x) # (B, T, n_embd)
        
        # Apply final layer norm
        x = self.ln_f(x) # (B, T, n_embd)
        
        # Get logits (predictions for next token)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            # PyTorch cross_entropy expects (Batch * Time, Channels)
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(logits_reshaped, targets_reshaped)

        return logits, loss
