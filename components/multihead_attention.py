import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module used in Transformer architectures.

    Projects inputs into query, key, and value representations,
    computes scaled dot-product attention across multiple heads,
    and combines the resulting context vectors.
    """
    def __init__(self, embedd_dim, n_heads, bias=False ) -> None:
       
      super().__init__()

      if(embedd_dim % n_heads != 0):
          raise ValueError(
              f"embedding dimension = {embedd_dim} should be divisible by number of heads = {n_heads}"
          )

      self.embedd_dim = embedd_dim
      self.n_heads = n_heads
      self.head_dim = embedd_dim // n_heads

      self.query_vec = nn.Linear(embedd_dim, embedd_dim, bias=bias)
      self.key_vec = nn.Linear(embedd_dim, embedd_dim, bias=bias)
      self.value_vec = nn.Linear(embedd_dim, embedd_dim, bias=bias)

      self.out = nn.Linear(embedd_dim, embedd_dim, bias=bias)

    def forward(self, query, key, value, mask=None):
      query = self.query_vec(query)
      key = self.key_vec(key)
      value = self.value_vec(value)

      # input dim => (batch_size, seq_len, out_dim)

      # (batch_size, seq_len, out_dim) => (batch, n_heads, seq_len, out_dim)
      query = rearrange(query, "b n (h d) -> b h n d", h=self.n_heads)
      key = rearrange(key, "b n (h d) -> b h n d", h=self.n_heads)
      value = rearrange(value, "b n (h d) -> b h n d", h=self.n_heads)

      ## calculate attention score
      attention_score = (torch.matmul(query, key.transpose(-2, -1))) / (key.shape[-1] ** 0.5)

      if mask is not None:
        attention_score = attention_score.masked_fill(mask == 0, float("-inf"))

      attention_score = F.softmax(attention_score, dim=-1)

      context_vec = torch.matmul(attention_score, value)

      context_vec = rearrange(context_vec, "b h n d -> b n (h d)")

      output = self.out(context_vec)

      return output