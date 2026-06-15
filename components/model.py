import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


from components.decoder import Decoder
from components.encoder import Encoder
from components.positional_encoding import PositionalEncoding
from components.multihead_attention import MultiHeadAttention
from components.mask import get_mask

class Transformer(nn.Module):
  """
    Transformer architecture consisting of stacked encoder and decoder blocks.

    The model uses token embeddings, sinusoidal positional encodings,
    multi-head self-attention, cross-attention, and feed-forward networks
    to perform sequence-to-sequence tasks such as machine translation.
    """
  def __init__(
      self,
      src_vocab_size,
      embedd_dim,
      tgt_vocab_size,
      max_seq_len,
      n_heads,
      dropout_ratio,
      bias ,
      n_encoders,
      n_decoders
  ):
    super().__init__()
    self.encoder_embedd = nn.Embedding(src_vocab_size, embedd_dim)
    self.decoder_embedd = nn.Embedding(tgt_vocab_size, embedd_dim)
    self.positional_encoding = PositionalEncoding(embedd_dim, max_seq_len)

    # create multiple encoder decoder layers
    self.encoder = nn.ModuleList([Encoder(embedd_dim, n_heads, dropout_ratio, bias) for _ in range(n_encoders)])
    self.decoder = nn.ModuleList([Decoder(embedd_dim, n_heads, dropout_ratio, bias) for _ in range(n_decoders)])

    self.dropout = nn.Dropout(dropout_ratio)

    self.FF = nn.Linear(embedd_dim, tgt_vocab_size)
    
    
    
  def forward(self, src, tgt):

    """
    Performs a forward pass through the Transformer.

    Args:
        src: Source token ids of shape (batch_size, src_seq_len).
        tgt: Target token ids of shape (batch_size, tgt_seq_len).

    Returns:
        shape = (batch_size, tgt_seq_len, tgt_vocab_size).
    """
    src_mask,tgt_mask = get_mask(src, tgt)

    encoder_embed = self.dropout(self.positional_encoding(self.encoder_embedd(src)))
    decoder_embed = self.dropout(self.positional_encoding(self.decoder_embedd(tgt)))

    encoder_output = encoder_embed
    for encoder_layer in self.encoder:
      encoder_output = encoder_layer(encoder_output, src_mask)

    decoder_output = decoder_embed
    for decoder_layer in self.decoder:
      decoder_output = decoder_layer(decoder_output, encoder_output, src_mask, tgt_mask)

    output = self.FF(decoder_output)

    return output



