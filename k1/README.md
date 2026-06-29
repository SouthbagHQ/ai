# Southbag-K1-Frontier

Southbag-K1-Frontier is Southbag's frontier banking agent — a from-scratch GPT (Generative Pre-trained Transformer) architecture implemented in PyTorch. The companion model is **Southbag-C1**. This repo focuses on clarity and showing how the models work under the hood.

## Project Structure

* `config.py`: Contains the `GPTConfig` dataclass storing hyperparameters (context length, embedding size, etc.).
* `tokenizer.py`: Implements a simple character-level tokenizer (`CharTokenizer`) to convert strings to integers and back.
* `model.py`: The core Transformer architecture implementation (Self-Attention, Feed-Forward, Blocks, and the GPT class).
* `generate.py`: Contains the autoregressive text generation function, supporting temperature, top-k sampling, and greedy decoding.

## Core Concepts Explained

### 1. Token Embeddings
Before text can be processed by a neural network, it must be converted into numbers. The tokenizer converts characters into integer IDs. A Token Embedding is a lookup table that maps each integer ID to a dense vector (a list of floating-point numbers) of size `n_embd`. This vector represents the "meaning" of the token.

### 2. Positional Embeddings
Transformers process all tokens in a sequence simultaneously, meaning they have no inherent concept of order (unlike RNNs). To fix this, we add a Positional Embedding to each Token Embedding. This is another lookup table that provides a unique vector for each position (1st token, 2nd token, etc.), allowing the model to understand where each token is located in the sequence.

### 3. Self-Attention
Self-attention is the core mechanism of the Transformer. It allows tokens to "look" at other tokens in the sequence to gather context. 
* Every token emits three vectors: a **Query** (what am I looking for?), a **Key** (what do I contain?), and a **Value** (what is my actual content?).
* Attention scores are calculated by taking the dot product of a token's Query with all past tokens' Keys. Higher scores mean more relevance.
* We apply a "mask" (the `tril` lower triangular matrix) to ensure tokens can only attend to past tokens (autoregressive property), preventing them from "cheating" by looking at future tokens.
* The scores are turned into probabilities using Softmax, and we use these probabilities to compute a weighted sum of the past tokens' Values.

### 4. Feed-Forward Network
After self-attention allows tokens to communicate, the Feed-Forward network allows them to "think" independently. It is a standard multi-layer perceptron (two linear layers with a ReLU activation in between) applied to each token's representation individually. This expands the feature space, applies a non-linearity, and projects it back.

### 5. Residual Connections
As neural networks get deeper, they suffer from the "vanishing gradient" problem, making them hard to train. Residual connections (or skip connections) solve this by adding the input of a layer directly to its output (`x = x + layer(x)`). This provides an unimpeded "highway" for gradients to flow backwards during backpropagation.

### 6. The Forward Pass
1. Input integers (tokens) are mapped to Token Embeddings and Positional Embeddings, then summed together.
2. The sequence passes through multiple Transformer Blocks. In each block:
   * Layer Normalization stabilizes learning.
   * Multi-Head Self-Attention allows tokens to communicate.
   * Residual connections bypass the attention layer.
   * Layer Normalization is applied again.
   * The Feed-Forward network processes each token individually.
   * Another residual connection bypasses the feed-forward layer.
3. Finally, a linear layer (the LM head) projects the final representations back into the vocabulary size, producing logits (raw prediction scores for the next token).

### 7. Text Generation (Autoregressive)
To generate text, we provide an initial context (e.g., "hello").
1. We pass the context through the model to get the logits for the next token.
2. We focus only on the prediction for the very last token in the sequence.
3. We convert logits to probabilities and sample the next token (using techniques like temperature scaling or top-k to control randomness).
4. We append the new token to our context and repeat the process, predicting one token at a time.

## Running the Code

Train the model, then generate:

```bash
pip install -r requirements.txt
python train.py
```

Southbag-K1-Frontier is trained on the Southbag persona dataset; Southbag-C1 is a small companion model.
