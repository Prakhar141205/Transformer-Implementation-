# Transformer From Scratch 
## inspired from Paper (Attenion is all you need)

A PyTorch implementation of the Transformer architecture built from scratch for educational purposes.

## Overview

This project aims to provide a deeper understanding of Transformer architectures by implementing the core building blocks manually instead of relying on high-level libraries.

Implemented components include:

* Multi-Head Attention
* Scaled Dot-Product Attention
* Feed Forward Network (FFN)
* Encoder Block
* Decoder Block
* Layer Normalization
* Residual Connections
* Attention Masking

## Project Structure

```text
.
├── attention.py
├── encoder.py
├── decoder.py
├── transformer.py
├── positional_encoding.py
└── README.md
```

## Architecture

```text
Input Embeddings
        │
        ▼
Positional Encoding
        │
        ▼
Encoder
        │
        ▼
Encoder Output
        │
        ▼
Decoder
        │
        ▼
Linear Layer
        │
        ▼
Output
```

## Tech Stack

* Python
* PyTorch

## Educational Purpose

> This repository is intended solely for learning and educational purposes.

The goal of this project is to understand how Transformer models work internally by implementing their components from scratch.

AI-assisted tools were used to help generate portions of the documentation, comments, and explanations. The project should be viewed as a learning exercise rather than a production-ready implementation.


## References

* Attention Is All You Need (Vaswani et al., 2017)
* PyTorch Documentation

## License

This project is released for educational purposes.
