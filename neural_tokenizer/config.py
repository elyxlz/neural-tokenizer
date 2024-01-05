from transformers import PretrainedConfig


class NeuralTokenizerConfig(PretrainedConfig):
    model_type = "neural_tokenizer"

    def __init__(
        self,
        char_vocab_size: int = 256,
        quantizer_levels: list[int] = [8, 8, 5, 3],  # 960 vocab size
        num_heads: int = 2,
        hidden_sizes: list[int] = [4, 8, 16, 32],
        latent_size: int = 48,
        factors: list[int] = [2, 2, 2, 2],
        mlp_factor: int = 1,
        window_size: int = 32,
        embed_pos: bool = False,
        norm: bool = False,
        zero_out: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.char_vocab_size = char_vocab_size
        self.quantizer_levels = quantizer_levels
        self.num_heads = num_heads
        self.hidden_sizes = hidden_sizes
        self.latent_size = latent_size
        self.mlp_factor = mlp_factor
        self.factors = factors
        self.window_size = window_size
        self.embed_pos = embed_pos
        self.norm = norm
        self.zero_out = zero_out

        self.num_layers = len(self.hidden_sizes)
