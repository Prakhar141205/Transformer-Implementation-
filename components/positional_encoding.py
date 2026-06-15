import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class PositionalEncoding(nn.Module):
  """
    Implements sinusoidal positional encodings for Transformer models.

    Positional encodings are added to token embeddings to provide
    information about the position of tokens within a sequence.
  """
  def __init__(self, embedd_dim, max_len=5000):
  
    super().__init__()

    self.pos = torch.arange(0, max_len)
    self.theta = 1 / ((10000 ** torch.arange(0, embedd_dim, 2)) / embedd_dim)

    positional_encodings = torch.zeros(max_len, embedd_dim)
    positional_encodings[:, 0::2] = torch.sin(self.pos[:, None] / self.theta)
    positional_encodings[:, 1::2] = torch.cos(self.pos[:, None] / self.theta)

    self.register_buffer("positional_encoding", positional_encodings) 

  def forward(self, x):
    """
    Adds positional encodings to the input embeddings.

    Args:
        x: Input tensor of shape
           (batch_size, seq_len, embedd_dim).

    Returns:
        Tensor of the same shape with positional
        encodings added.
    """
    batch_size, seq_len, embed_size = x.shape
    return x + self.pe[:seq_len, :]