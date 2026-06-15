from components.model import Transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


model = Transformer(
    src_vocab_size=100,
    tgt_vocab_size=120,
    embedd_dim=64,
    max_seq_len=50,
    n_heads=8,
    dropout_ratio=0.1,
    bias=True,
    n_encoders=2,
    n_decoders=2
)

print(model)


# check forward pass

# forward pass
batch_size = 4
src_len = 10
tgt_len = 8

src = torch.randint(1, 100, (batch_size, src_len))
tgt = torch.randint(1, 120, (batch_size, tgt_len))

output = model(src, tgt)
print(output.shape)

## check backward pass
target = torch.randint(0, 120, (4, 8))

loss_fn = nn.CrossEntropyLoss()

loss = loss_fn(
    output.reshape(-1, 120),
    target.reshape(-1)
)

loss.backward()

print(loss.item())