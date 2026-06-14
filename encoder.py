import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Encoder(nn.Module):
  """
    Single Transformer Encoder Block.

    Components:
        - Multi-Head Self-Attention
        - Feed Forward Network (FFN)
        - Residual Connections
        - Layer Normalization
        - Dropout

    Args:
        embedd_dim (int):
            Dimension of token embeddings and hidden representations.
            Often referred to as d_model.

        n_heads (int):
            Number of attention heads used in
            multi-head self-attention.

        ff_dim (int):
            Hidden dimension of the Feed Forward Network.

        bias (bool, optional):
            Whether linear layers use bias terms.

            Default:
                False
    """
  def __init__(self, embedd_dim, n_heads, ff_dim, bias=False):
    super().__init__()


    self.mha = MultiHeadAttention(embedd_dim, n_heads, drop_out = 0.1, bias=bias)
    self.ff = FNN(embedd_dim, bias=bias)

    self.norm1 = nn.LayerNorm(embedd_dim)
    self.norm2 = nn.LayerNorm(embedd_dim)

    self.dropout = nn.Dropout(0.1)

  def forward(self, x, mask=None):
    """
    Forward pass of a Transformer Encoder Block.

    Args:
        x (Tensor):
            Input token embeddings.

            Shape:
                (batch_size, seq_len, embedd_dim)

            Example:
                (32, 128, 512)

        mask (Tensor, optional):
            Padding mask used to ignore
            padded tokens during attention.

            Shape:
                (batch_size, 1, 1, seq_len)

            Default:
                None

    Returns:
        Tensor:
            Contextualized token representations
            produced by the encoder block.

            Shape:
                (batch_size, seq_len, embedd_dim)

    Processing Steps:
        1. Multi-Head Self-Attention
           Q = K = V = x

        2. Residual Connection + LayerNorm

        3. Feed Forward Network (FFN)

        4. Residual Connection + LayerNorm

        5. Return encoder representations
    """
    x = x + self.dropout(self.mha(x, x, x, mask))

    ff_out = self.FF(x)

    x = x + self.dropout(self.norm2(ff_out))

    return x
    