"""Tokenizer wrapper for the JAX Transformer LLM."""

from typing import List, Union
import numpy as np


class SimpleTokenizer:
    """
    A simple character-level tokenizer as a fallback.
    For production, replace with SentencePiece or a pre-trained tokenizer.
    """

    def __init__(self, vocab_size: int = 50257):
        """Initialize tokenizer."""
        self.vocab_size = vocab_size
        # Create a simple char-level vocab (extend with special tokens)
        self.char_to_id = {chr(i): i for i in range(256)}
        self.id_to_char = {i: chr(i) for i in range(256)}
        # Add special tokens
        self.char_to_id["<PAD>"] = 0
        self.char_to_id["<UNK>"] = 1
        self.char_to_id["<BOS>"] = 2
        self.char_to_id["<EOS>"] = 3
        self.id_to_char[0] = "<PAD>"
        self.id_to_char[1] = "<UNK>"
        self.id_to_char[2] = "<BOS>"
        self.id_to_char[3] = "<EOS>"

    def encode(self, text: str, max_length: int = None) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            max_length: Optional max length (pads or truncates)
            
        Returns:
            List of token IDs
        """
        tokens = [self.char_to_id.get(c, self.char_to_id["<UNK>"]) for c in text]
        
        if max_length is not None:
            if len(tokens) < max_length:
                tokens += [self.char_to_id["<PAD>"]] * (max_length - len(tokens))
            else:
                tokens = tokens[:max_length]
        
        return tokens

    def decode(self, token_ids: Union[List[int], np.ndarray]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List or array of token IDs
            
        Returns:
            Decoded text string
        """
        text = ""
        for token_id in token_ids:
            char = self.id_to_char.get(int(token_id), "<UNK>")
            if char not in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
                text += char
        return text
