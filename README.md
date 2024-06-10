GPT Using Pytorch 

### Configuration
The `Config` class contains the hyperparameters and configurations for the model:
- `block_size`: Length of input sequences.
- `n_embd`: Embedding dimension.
- `n_layer`: Number of transformer layers.
- `n_head`: Number of attention heads.
- `dropout`: Dropout rate.
- `learning_rate`: Learning rate for the optimizer.
- `max_grad_norm`: Maximum gradient norm for clipping.
- `epochs`: Number of training epochs.
- `batch_size`: Batch size for training.
- `max_new_tokens`: Maximum number of tokens to generate.
- `temperature`: Temperature for sampling.
- `top_k`: Number of top logits to consider for sampling.

### TokenEmbedding
The `TokenEmbedding` class maps tokens (characters) to their corresponding embeddings.
- `vocab_size`: Size of the vocabulary.
- `n_embd`: Embedding dimension.
- The `forward` method returns the token embeddings for the input tensor `x`.

### PositionalEmbedding
The `PositionalEmbedding` class provides positional embeddings to encode the position of tokens in a sequence.
- `block_size`: Maximum sequence length.
- `n_embd`: Embedding dimension.
- The `forward` method creates and returns positional embeddings.

### MultiHeadAttention
The `MultiHeadAttention` class implements the multi-head attention mechanism.
- `n_embd`: Embedding dimension.
- `n_head`: Number of attention heads.
- `dropout`: Dropout rate.
- The `forward` method computes the attention scores and applies them to the value vectors, returning the projected output.

### FeedForward
The `FeedForward` class implements a feed-forward neural network with one hidden layer.
- `n_embd`: Embedding dimension.
- `dropout`: Dropout rate.
- The `forward` method applies a linear transformation followed by GELU activation, dropout, and another linear transformation.

### Block
The `Block` class represents a transformer block that contains a multi-head attention layer followed by a feed-forward network.
- `n_embd`: Embedding dimension.
- `n_head`: Number of attention heads.
- `dropout`: Dropout rate.
- The `forward` method applies layer normalization, attention, and feed-forward operations in sequence.

### GPT
The `GPT` class is the main model that combines token and positional embeddings, multiple transformer blocks, and a final linear layer for generating logits.
- `vocab_size`: Size of the vocabulary.
- `config`: Configuration object.
- The `forward` method passes the input through the token and positional embeddings, transformer blocks, and final linear layer to obtain logits.
- The `generate` method generates new tokens based on the input indices using sampling with temperature and top-k filtering.

### TextDataset
The `TextDataset` class handles the dataset, converting text to indices and preparing input-output pairs for training.
- `text`: Input text data.
- `block_size`: Length of input sequences.
- The `__len__` method returns the number of possible input-output pairs.
- The `__getitem__` method returns a pair of input and target sequences.

### create_mask
The `create_mask` function creates a lower triangular mask for causal self-attention.
- `size`: Size of the mask.
- The mask ensures that the model can only attend to previous and current tokens.

### train
The `train` function trains the GPT model on the provided dataset.
- `model`: GPT model.
- `dataset`: TextDataset object.
- `config`: Configuration object.
- It initializes the data loader, optimizer, and loss function, then trains the model for the specified number of epochs, printing the loss at regular intervals. 

### Example Usage   

1. It sets the device to CUDA if available.
2. Reads text from a file.
3. Creates a TextDataset object.
4. Initializes the GPT model with the vocabulary size and configuration.
5. Trains the model on the dataset.
6. Generates new text based on an input string and prints the input and generated text.


