# Loading
```
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
text = ["Hello", "World :)"]
tokens = model.encode(text)
print(tokens.data)
# [[0, 1235, d1236, 1], [0, 1237, 1238, 1239, 1240, 1]]

recon = model.decode(tokens)
print(recon)
# ["Hello", "World :)"]
```

# TODO
- [ ] Dataloader with HF datasets
- [ ] Add training 
- [ ] AR and discrete diffusion decoders
- [ ] Variational + continuous bottleneck
- [ ] Replace SparseTensor abstract class with torch nested tensors once they mature