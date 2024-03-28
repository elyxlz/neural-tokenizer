from neural_tokenizer import (
    NeuralTokenizer,
    NeuralTokenizerConfig,
)

model = NeuralTokenizer(NeuralTokenizerConfig())

# print num params in M
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
print(f"num params: {num_params}M")

some_strings = [
    "hi",
    "worldasdkjhasdkjhaskjdhaskjdhkasjdhkajsdhkajsdhkajsd",
    "this",
    "is",
    "a",
    "test",
]

# lossless
char_tokens, mask = model.char_tokenize(some_strings)
recon_lossless = model.char_detokenize(char_tokens, mask)

z, mask = model.encode(some_strings)

recon = model.decode(z, mask)

loss = model.forward(some_strings)
