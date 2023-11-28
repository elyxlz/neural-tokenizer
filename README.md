# neural-tokenizer
High compression text tokenizers via VQAEs for efficient and democratic language modeling.

Language models struggle with semantic modeling due to high frequency details in typical tokens.

# Loading
```python
from neural_tokenizer import NeuralTokenizer, NeuralTokenizerConfig

# Random initialization
model = NeuralTokenizer(NeuralTokenizerConfig())

# Push to huggingface hub
model = model.push_to_hub("elyxlz/neural-tokenizer-v1")

# Load pretrained model
model = NeuralTokenizer.from_pretrained("elyxlz/neural-tokenizer-v1")
```


# Usage
```python
text = ["Hello world!"]
tokens = model.tokenize(text)
print(tokens)
# [[0, 1235, 1236, 1]]

recon = model.detokenize(tokens)
print(recon)
# ['Hello world!']
```