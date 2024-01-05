# neural-tokenizer
High compression text tokenizers via VQAEs for efficient and democratic language modeling.

Language models struggle with semantic modeling due to high frequency details in tokens from typical tokenizers, employing stronger textual compression via neural tokenizers may solve alleviate this problem.


# Usage
```python
from neural_tokenizer import NeuralTokenizer

# Load pretrained model
model = NeuralTokenizer.from_pretrained("elyxlz/neural-tokenizer-v1")

text = ["Hello", "World :)"]
tokens = model.encode(text)
print(tokens.data)
# [[0, 1235, 1236, 1], [0, 1237, 1238, 1239, 1240, 1]]

recon = model.decode(tokens)
print(recon)
# ["Hello", "World :)"]

loss = model.forward(text, max_len=2048)
# 5.56...
```

# Training
Install train dependencies
```zsh
pip install -e '.[train]'
```

Setup accelerate config
```sh
accelerate config
```

Create a config file like the one in `configs/demo_run.py`

Then run the training
```sh
accelerate launch train.py demo_run
```

# TODO
- [x] Dataloader with HF datasets
- [x] Add training 
- [ ] Implement varlen windowed flash attn
- [ ] Validate idea with a simple experiment
- [ ] GAN training
- [ ] Variational + continuous bottleneck
