"""Configuration for the JAX Transformer LLM."""

# Model hyperparameters
NUM_LAYERS = 4
D_MODEL = 256
NUM_HEADS = 4
D_FF = 1024
MAX_SEQ_LEN = 512
VOCAB_SIZE = 50257  # GPT-2 tokenizer vocab size
DROPOUT_RATE = 0.1

# Training hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_EPOCHS = 3
SEQ_LEN = 128

# Device & dtype
DTYPE = "float32"

# Random seed
SEED = 42
