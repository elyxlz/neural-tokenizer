from neural_tokenizer import NeuralTokenizer, NeuralTokenizerConfig

model = NeuralTokenizer(NeuralTokenizerConfig(
    hidden_sizes = [32, 32, 64, 64],
    latent_size = 128,
    mlp_factor = 2,
))

# print num params in M
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
print(f"num params: {num_params}M")

some_strings = ["hello", "world", "this", "is", "a", "test"]

z = model.encode(some_strings)

recon = model.decode(z)

import pdb; pdb.set_trace()