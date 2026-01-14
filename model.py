"""JAX Transformer model using Flax."""

from typing import Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn
from config import (
    NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, MAX_SEQ_LEN,
    VOCAB_SIZE, DROPOUT_RATE, DTYPE
)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module."""
    
    d_model: int = D_MODEL
    num_heads: int = NUM_HEADS
    dropout_rate: float = DROPOUT_RATE

    def setup(self):
        """Initialize linear layers."""
        self.d_k = self.d_model // self.num_heads
        self.W_q = nn.Dense(self.d_model)
        self.W_k = nn.Dense(self.d_model)
        self.W_v = nn.Dense(self.d_model)
        self.W_o = nn.Dense(self.d_model)
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray = None,
                 training: bool = False) -> jnp.ndarray:
        """
        Compute multi-head attention.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask
            training: Whether in training mode
            
        Returns:
            Attention output [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.W_q(x)  # [batch, seq_len, d_model]
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape for multi-head: [batch, seq_len, num_heads, d_k] -> [batch, num_heads, seq_len, d_k]
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(jnp.array(self.d_k, dtype=jnp.float32))
        
        # Apply causal mask (for decoder)
        if mask is not None:
            scores = scores + mask
        else:
            # Create causal mask
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len))) - 1
            causal_mask = causal_mask * -1e9
            scores = scores + causal_mask[jnp.newaxis, jnp.newaxis, :, :]
        
        attn_weights = nn.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights, deterministic=not training)
        
        # Apply attention to values
        context = jnp.matmul(attn_weights, V)  # [batch, num_heads, seq_len, d_k]
        
        # Concatenate heads: [batch, seq_len, d_model]
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        output = self.W_o(context)
        return output


class FeedForward(nn.Module):
    """Feed-forward network."""
    
    d_model: int = D_MODEL
    d_ff: int = D_FF
    dropout_rate: float = DROPOUT_RATE

    def setup(self):
        """Initialize linear layers."""
        self.W_1 = nn.Dense(self.d_ff)
        self.W_2 = nn.Dense(self.d_model)
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Feed-forward transformation with ReLU activation.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            training: Whether in training mode
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        x = self.W_1(x)
        x = nn.relu(x)
        x = self.dropout(x, deterministic=not training)
        x = self.W_2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block (attention + FFN + layer norms)."""
    
    d_model: int = D_MODEL
    num_heads: int = NUM_HEADS
    d_ff: int = D_FF
    dropout_rate: float = DROPOUT_RATE

    def setup(self):
        """Initialize attention, FFN, and layer norms."""
        self.attn = MultiHeadAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        )
        self.ffn = FeedForward(
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout_rate=self.dropout_rate
        )
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Apply transformer block with residual connections.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            training: Whether in training mode
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Self-attention + residual + layer norm
        attn_out = self.attn(self.ln1(x), training=training)
        x = x + self.dropout(attn_out, deterministic=not training)
        
        # FFN + residual + layer norm
        ffn_out = self.ffn(self.ln2(x), training=training)
        x = x + self.dropout(ffn_out, deterministic=not training)
        
        return x


class Transformer(nn.Module):
    """Decoder-only Transformer LLM."""
    
    num_layers: int = NUM_LAYERS
    d_model: int = D_MODEL
    num_heads: int = NUM_HEADS
    d_ff: int = D_FF
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = MAX_SEQ_LEN
    dropout_rate: float = DROPOUT_RATE

    def setup(self):
        """Initialize embeddings and transformer blocks."""
        self.token_embed = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)
        self.pos_embed = nn.Embed(num_embeddings=self.max_seq_len, features=self.d_model)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        self.blocks = [
            TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate
            )
            for _ in range(self.num_layers)
        ]
        
        self.ln_final = nn.LayerNorm()
        self.lm_head = nn.Dense(self.vocab_size)

    def __call__(self, input_ids: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            training: Whether in training mode
            
        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        seq_len = input_ids.shape[1]
        pos_ids = jnp.arange(seq_len)[jnp.newaxis, :]
        
        # Embedding
        x = self.token_embed(input_ids) + self.pos_embed(pos_ids)
        x = self.dropout(x, deterministic=not training)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, training=training)
        
        # Output layer norm and projection
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        return logits
