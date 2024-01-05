from neural_tokenizer import (
    NeuralTokenizer,
    NeuralTokenizerConfig,
    Trainer,
    TrainConfig,
)

model = NeuralTokenizer(NeuralTokenizerConfig(
    char_vocab_size=256,
))

trainer = Trainer(
    model=model,
    train_config=TrainConfig(
    name='neural-tokenizer-demo',
))

