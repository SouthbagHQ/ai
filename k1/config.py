"""
Configuration for the Southbag-K1-Frontier model.
"""

from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 65      # Size of the vocabulary (e.g., number of unique characters)
    context_length: int = 256 # Maximum sequence length (block size)
    n_embd: int = 384         # Embedding dimension
    n_head: int = 6           # Number of attention heads
    n_layer: int = 6          # Number of transformer blocks
    dropout: float = 0.2      # Dropout probability
