"""
Configuration for the Southbag-C1 model.
"""

from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 65      # Size of the vocabulary (e.g., number of unique characters)
    context_length: int = 128 # Maximum sequence length (block size)
    n_embd: int = 128         # Embedding dimension
    n_head: int = 4           # Number of attention heads
    n_layer: int = 4          # Number of transformer blocks
    dropout: float = 0.1      # Dropout probability
