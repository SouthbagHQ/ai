"""
A simple character-level tokenizer.
"""

class CharTokenizer:
    def __init__(self, data: str):
        """
        Initializes the tokenizer by finding all unique characters in the dataset.
        
        Args:
            data (str): A sample of text to build the vocabulary from.
        """
        # Find all unique characters in the provided data
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        
        # Create mappings from character to integer and vice versa
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
    def encode(self, s: str) -> list[int]:
        """
        Encodes a string into a list of integers.
        """
        # If a character is unseen, we just skip or we could have a special <UNK> token.
        # For simplicity in this basic version, we ignore unknown characters.
        return [self.stoi[c] for c in s if c in self.stoi]
        
    def decode(self, l: list[int]) -> str:
        """
        Decodes a list of integers back into a string.
        """
        return ''.join([self.itos[i] for i in l if i in self.itos])
