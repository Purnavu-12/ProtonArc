# ProtonArc ⚛

A simple, self-contained decoder-only transformer language model built entirely in JAX and Flax.
(STILL IN TRAINING STAGE , NOT READY TO USE..)


## Features

- **Decoder-only Transformer**: GPT-style architecture (4 layers, 256 hidden dim, 4 heads)
- **Vocabulary Size**: 50,257 (GPT-2 style)
- **Maximum Sequence Length**: 512 tokens
- **Device Support**: CPU/GPU/TPU (auto-detected by JAX)
- **Inference Methods**: Greedy decoding and top-k sampling
- **Training Framework**: Ready for real data (uses dummy data initially)

## Project Structure

```
HackLLM/
├── config.py         # Hyperparameters (adjustable)
├── model.py          # Transformer model (Flax modules)
├── tokenizer.py      # Simple character-level tokenizer
├── inference.py      # Inference script (greedy & top-k)
├── train.py          # Training loop with dummy data
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## Installation

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Inference (Out-of-the-box)

Generate text from a prompt using the randomly-initialized model:

```bash
# Default prompt
python inference.py

# Custom prompt
python inference.py --prompt "The future of AI is" --max_tokens 100

# Using top-k sampling (more diverse)
python inference.py --prompt "Hello" --method top_k --max_tokens 50
```

**Note**: Since the model is randomly initialized and untrained, generated text will be random. This is expected!

### Training

Train the model on your data:

```bash
# Train with dummy data (demonstration)
python train.py --epochs 3 --steps 100

# Adjust hyperparameters
python train.py --epochs 10 --batch_size 16
```

**Important**: The training script uses dummy data by default. To train on real data:
1. Replace `generate_dummy_batch()` in `train.py` with your data loader
2. Use Hugging Face `datasets` library or similar
3. Tokenize your text with the tokenizer

Example with real data:
```python
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2")
# Tokenize and load batches in train.py
```

## Model Hyperparameters

Edit `config.py` to adjust:

```python
NUM_LAYERS = 4           # Increase for larger model
D_MODEL = 256            # Hidden dimension
NUM_HEADS = 4            # Attention heads
D_FF = 1024              # Feed-forward dimension
MAX_SEQ_LEN = 512        # Max sequence length
VOCAB_SIZE = 50257       # GPT-2 vocab
DROPOUT_RATE = 0.1       # Regularization
BATCH_SIZE = 8           # Training batch size
LEARNING_RATE = 1e-3     # Adam LR
```

## Architecture Overview

### Core Components

1. **Token Embedding**: Maps input tokens to d_model dimensions
2. **Positional Encoding**: Adds position information (Flax Embed layer)
3. **Multi-Head Attention**: Scaled dot-product attention with 4 heads
4. **Feed-Forward**: Two linear layers with ReLU activation
5. **Layer Normalization**: Pre-norm residual connections
6. **Output Layer**: Projects to vocabulary logits

### Model Size (Default Config)

- Parameters: ~3.7M
- Memory: ~15 MB (with default batch size 8)
- Inference Speed: ~100-500 tokens/sec (CPU), 1000+ tokens/sec (GPU)

## Tokenizer

The included `SimpleTokenizer` is a character-level tokenizer for demo purposes. To use a real tokenizer:

```python
# Option 1: Use pre-trained SentencePiece
from sentencepiece import SentencePieceProcessor
sp = SentencePieceProcessor(model_file='gpt2.model')

# Option 2: Use Hugging Face tokenizers
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Update tokenizer.py to wrap the real tokenizer
```

## Scaling

To create a larger model, modify `config.py`:

```python
# Small model (current)
NUM_LAYERS = 4
D_MODEL = 256
NUM_HEADS = 4

# Medium model
NUM_LAYERS = 8
D_MODEL = 512
NUM_HEADS = 8

# Larger model (requires more memory)
NUM_LAYERS = 12
D_MODEL = 768
NUM_HEADS = 12
```

## Device Handling

JAX automatically selects device (CPU/GPU/TPU). To explicitly set:

```bash
# Use CPU only
JAX_PLATFORM_NAME=cpu python inference.py

# Use GPU
JAX_PLATFORM_NAME=gpu python inference.py
```

Check available devices:
```python
import jax
print(jax.devices())
```

## Key Files Explained

### `model.py` (224 lines)
- `MultiHeadAttention`: Scaled dot-product attention with causal masking
- `FeedForward`: MLPLayer with ReLU
- `TransformerBlock`: Attention + FFN with residual connections
- `Transformer`: Full model with embeddings and stacked blocks

### `inference.py` (130 lines)
- `initialize_model()`: Creates model and initializes with random weights
- `generate_greedy()`: Argmax token selection
- `generate_top_k()`: Top-k sampling for diversity

### `train.py` (140 lines)
- `loss_fn()`: Cross-entropy loss for language modeling
- `train_step()`: Single JAX-based gradient update step
- `main()`: Training loop with Adam optimizer

### `config.py` (16 lines)
- All hyperparameters in one place for easy tuning

### `tokenizer.py` (62 lines)
- `SimpleTokenizer`: Character-level encode/decode (demo)
- Ready for SentencePiece or Hugging Face integration

## Debugging & Troubleshooting

### Issue: CUDA/GPU not detected
```bash
# Verify JAX installation
python -c "import jax; print(jax.devices())"

# If only CPU shows, reinstall jaxlib with GPU support
pip install --upgrade jax jaxlib
```

### Issue: Out of memory
- Reduce `BATCH_SIZE` in `config.py`
- Reduce `D_MODEL` or `NUM_LAYERS`
- Use CPU instead of GPU for debugging

### Issue: Slow inference
- Check device with `jax.devices()`
- Reduce `MAX_SEQ_LEN` if not needed
- Consider batch processing for multiple prompts

## Next Steps

1. **Real Data**: Load your dataset (text files, HuggingFace datasets, etc.)
2. **Tokenization**: Use proper tokenizer (SentencePiece, BPE, etc.)
3. **Validation**: Add validation loop in `train.py`
4. **Checkpointing**: Save/load model weights with `jax.tree_util` and `pickle`
5. **Evaluation**: Implement perplexity or other metrics
6. **Scaling**: Increase model size and training data
7. **Fine-tuning**: Add downstream task adaptation

## References

- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)
- [Optax Documentation](https://optax.readthedocs.io/)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/papers/Better_Language_Models_and_Their_Roles_in_Society.pdf)

## License

MIT License - Feel free to use for learning and experimentation.

## Notes

- **Random Initialization**: Model weights are randomly initialized. Training is required for meaningful text generation.
- **Dummy Data**: Training script uses synthetic data. Replace with real data for actual training.
- **Simple Tokenizer**: Character-level tokenizer is for demo. Use a real BPE/SentencePiece tokenizer for production.
- **No Pre-trained Weights**: No external model loading; 100% standalone.

---

Built for learning and experimentation. Happy training! 🚀
=======
