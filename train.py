"""Training script for the JAX Transformer LLM."""

import jax
import jax.numpy as jnp
import optax
import argparse
from typing import Tuple, Dict
import numpy as np

from model import Transformer
from tokenizer import SimpleTokenizer
from config import (
    NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, VOCAB_SIZE, MAX_SEQ_LEN,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, SEQ_LEN, SEED
)


def initialize_model_and_optimizer(rng: jax.random.PRNGKey) -> Tuple:
    """
    Initialize model and optimizer.
    
    Args:
        rng: Random number generator key
        
    Returns:
        Model, params, optimizer, opt_state
    """
    model = Transformer(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN
    )
    
    # Initialize parameters
    dummy_input = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
    params = model.init(rng, dummy_input, training=True)
    
    # Initialize optimizer (Adam)
    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = optimizer.init(params)
    
    return model, params, optimizer, opt_state


def loss_fn(
    params: dict,
    model: Transformer,
    input_ids: jnp.ndarray,
    target_ids: jnp.ndarray,
    rng: jax.random.PRNGKey
) -> jnp.ndarray:
    """
    Compute cross-entropy loss.
    
    Args:
        params: Model parameters
        model: Transformer model
        input_ids: Input token IDs [batch, seq_len]
        target_ids: Target token IDs [batch, seq_len]
        rng: Random key for dropout
        
    Returns:
        Scalar loss
    """
    logits = model.apply(params, input_ids, training=True, rngs={"dropout": rng})
    
    # Shift for language modeling: predict next token
    logits = logits[:, :-1, :]  # [batch, seq_len-1, vocab_size]
    target_ids = target_ids[:, 1:]  # [batch, seq_len-1]
    
    # Flatten for cross-entropy
    logits_flat = logits.reshape(-1, VOCAB_SIZE)
    target_flat = target_ids.reshape(-1)
    
    # Cross-entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, target_flat)
    return jnp.mean(loss)


def train_step(
    state: Dict,
    batch: jnp.ndarray,
    rng: jax.random.PRNGKey
) -> Tuple[Dict, jnp.ndarray]:
    """
    Single training step.
    
    Args:
        state: Training state (params, opt_state, model, optimizer)
        batch: Batch of token IDs [batch, seq_len]
        rng: Random key
        
    Returns:
        Updated state and loss
    """
    params, opt_state, model, optimizer = state["params"], state["opt_state"], state["model"], state["optimizer"]
    
    # Input and target (shifted by 1 for autoregressive)
    input_ids = batch
    target_ids = batch
    
    # Compute gradients
    loss, grads = jax.value_and_grad(loss_fn)(params, model, input_ids, target_ids, rng)
    
    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    state["params"] = params
    state["opt_state"] = opt_state
    
    return state, loss


def generate_dummy_batch(batch_size: int, seq_len: int) -> jnp.ndarray:
    """
    Generate a dummy batch for demonstration.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        
    Returns:
        Batch of random token IDs [batch_size, seq_len]
    """
    return jnp.array(np.random.randint(4, VOCAB_SIZE, size=(batch_size, seq_len)))


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train JAX Transformer LLM")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--steps", type=int, default=100, help="Steps per epoch (for dummy data)")
    
    args = parser.parse_args()
    
    try:
        print("Initializing JAX Transformer LLM for training...")
        print(f"Device: {jax.devices()}")
        
        # Initialize
        rng = jax.random.PRNGKey(args.seed)
        model, params, optimizer, opt_state = initialize_model_and_optimizer(rng)
        
        num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"Model initialized with {num_params} parameters")
        
        # Training state
        state = {
            "params": params,
            "opt_state": opt_state,
            "model": model,
            "optimizer": optimizer
        }
        
        # Training loop (using dummy data for demonstration)
        print("\nStarting training (with dummy data)...")
        print("NOTE: Replace generate_dummy_batch() with real data from your dataset (e.g., Hugging Face datasets)\n")
        
        for epoch in range(args.epochs):
            total_loss = 0.0
            
            for step in range(args.steps):
                # Generate dummy batch
                batch = generate_dummy_batch(args.batch_size, SEQ_LEN)
                
                # Split RNG for dropout
                rng, subkey = jax.random.split(rng)
                
                # Training step
                state, loss = train_step(state, batch, subkey)
                total_loss += loss
                
                if (step + 1) % 20 == 0:
                    avg_loss = total_loss / (step + 1)
                    print(f"Epoch {epoch + 1}/{args.epochs}, Step {step + 1}/{args.steps}, Loss: {avg_loss:.4f}")
            
            print(f"Epoch {epoch + 1}/{args.epochs} completed, Avg Loss: {total_loss / args.steps:.4f}\n")
        
        print("Training completed!")
        print("\nNext steps:")
        print("1. Replace generate_dummy_batch() with real data loading")
        print("2. Add validation loop")
        print("3. Implement checkpoint saving with jax.tree_util")
        print("4. Add learning rate scheduling with optax")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
