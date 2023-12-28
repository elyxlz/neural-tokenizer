from setuptools import setup, find_packages

setup(
    name="neural-tokenizer",
    version="0.0.1",
    author="Elio Pascarelli",
    author_email="elio@pascarelli.com",
    description="VQAEs for Text",
    url="",  # Add the URL of your package here
    classifiers=[
        "Framework :: Pytorch",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",
        "einops",
        "python-dotenv",
        # "flash_attn",
    ],
    extras_require={"train": ["accelerate", "wandb"]},
    license="MIT",
)
