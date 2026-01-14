"""Inference script for the JAX Transformer LLM."""

import jax
import jax.numpy as jnp
import argparse
from typing import Tuple
import numpy as np

from model import Transformer
from tokenizer import SimpleTokenizer
from config import (
    NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, VOCAB_SIZE, MAX_SEQ_LEN,
    SEED, SEQ_LEN
)


def initialize_model(rng: jax.random.PRNGKey) -> Tuple:
    """
    Initialize model with random weights.
    
    Args:
        rng: Random number generator key
        
    Returns:
        Model instance and parameters
    """
    model = Transformer(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN
    )
    
    # Initialize with dummy input
    dummy_input = jnp.ones((1, SEQ_LEN), dtype=jnp.int32)
    params = model.init(rng, dummy_input, training=False)
    
    return model, params


def generate_greedy(
    model: Transformer,
    params: dict,
    prompt_ids: jnp.ndarray,
    max_new_tokens: int = 50,
    rng: jax.random.PRNGKey = None
) -> jnp.ndarray:
    """
    Generate text using greedy decoding.
    
    Args:
        model: Transformer model
        params: Model parameters
        prompt_ids: Initial prompt token IDs [1, prompt_len]
        max_new_tokens: Number of tokens to generate
        rng: Random key (for future sampling extensions)
        
    Returns:
        Generated token IDs [1, prompt_len + max_new_tokens]
    """
    generated = prompt_ids
    
    for _ in range(max_new_tokens):
        # Get current sequence (truncate if too long)
        current_seq = generated[:, -SEQ_LEN:]
        
        # Forward pass
        logits = model.apply(params, current_seq, training=False)
        
        # Get last token logits
        next_logits = logits[:, -1, :]
        
        # Greedy decoding: take argmax
        next_token = jnp.argmax(next_logits, axis=-1, keepdims=True)
        
        # Append to sequence
        generated = jnp.concatenate([generated, next_token], axis=1)
    
    return generated


def generate_top_k(
    model: Transformer,
    params: dict,
    prompt_ids: jnp.ndarray,
    max_new_tokens: int = 50,
    k: int = 5,
    rng: jax.random.PRNGKey = None
) -> jnp.ndarray:
    """
    Generate text using top-k sampling.
    
    Args:
        model: Transformer model
        params: Model parameters
        prompt_ids: Initial prompt token IDs [1, prompt_len]
        max_new_tokens: Number of tokens to generate
        k: Number of top candidates to sample from
        rng: Random key for sampling
        
    Returns:
        Generated token IDs [1, prompt_len + max_new_tokens]
    """
    if rng is None:
        rng = jax.random.PRNGKey(SEED)
    
    generated = prompt_ids
    
    for i in range(max_new_tokens):
        rng, subkey = jax.random.split(rng)
        
        # Get current sequence
        current_seq = generated[:, -SEQ_LEN:]
        
        # Forward pass
        logits = model.apply(params, current_seq, training=False)
        next_logits = logits[:, -1, :].squeeze(0)
        
        # Top-k filtering
        top_k_logits, top_k_indices = jax.lax.top_k(next_logits, k=min(k, VOCAB_SIZE))
        
        # Softmax and sample
        probs = jax.nn.softmax(top_k_logits)
        sampled_idx = jax.random.choice(subkey, k, p=probs)
        next_token = top_k_indices[sampled_idx]
        
        # Append to sequence
        generated = jnp.concatenate([generated, next_token[jnp.newaxis, jnp.newaxis]], axis=1)
    
    return generated


def main():
    """Main inference script."""
    parser = argparse.ArgumentParser(description="Generate text with JAX Transformer LLM")
    parser.add_argument("--prompt", type=str, default="Hello world", help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--method", type=str, default="greedy", choices=["greedy", "top_k"], help="Generation method")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    
    args = parser.parse_args()
    
    try:
        print("Initializing JAX Transformer LLM...")
        print(f"Device: {jax.devices()}")
        
        # Initialize model
        rng = jax.random.PRNGKey(args.seed)
        model, params = initialize_model(rng)
        print(f"Model initialized with {sum(x.size for x in jax.tree_util.tree_leaves(params))} parameters")
        
        # Initialize tokenizer
        tokenizer = SimpleTokenizer(vocab_size=VOCAB_SIZE)
        
        # Encode prompt
        prompt_ids = jnp.array([tokenizer.encode(args.prompt, max_length=None)])
        print(f"Prompt tokens: {prompt_ids.shape}")
        
        # Generate
        print(f"Generating {args.max_tokens} tokens using {args.method}...")
        if args.method == "greedy":
            generated_ids = generate_greedy(model, params, prompt_ids, max_new_tokens=args.max_tokens)
        else:
            generated_ids = generate_top_k(model, params, prompt_ids, max_new_tokens=args.max_tokens)
        
        # Decode and print
        generated_text = tokenizer.decode(generated_ids[0])
        print("\n=== Generated Text ===")
        print(generated_text)
        print("=" * 22)
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
