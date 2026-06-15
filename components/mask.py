import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def get_mask(src, tgt):
  """
    Creates source and target attention masks for the Transformer.

    Args:
        src: Source token ids of shape (batch_size, src_seq_len).
        tgt: Target token ids of shape (batch_size, tgt_seq_len).

    Returns:
        src_mask: Masks padding tokens in the source sequence.
        final_tgt_mask: Combines padding and causal masking for
                        autoregressive decoder self-attention.
    """

  src_mask = (src != 0) [:, None, None, :]  # (batch_size, seq_len) => (batch_size, n_heads, query_len, key_len)
  tgt_mask = (tgt != 0) [:, None, :, None]

  seq_len = tgt_mask.shape[-2]

  causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()  # causal_mack (look ahead mask ) it is making the decoder autoregressive
  final_tgt_mask = causal_mask & tgt_mask

  return src_mask, final_tgt_mask

  

