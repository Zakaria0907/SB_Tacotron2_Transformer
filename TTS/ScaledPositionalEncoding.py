import torch
import torch.nn as nn
import math
import speechbrain.lobes.models.transformer.Transformer as Transformer


class ScaledPositionalEncoding(nn.Module):
    """Implements the scaled absolute sinusoidal positional encoding function.

    PE(pos, 2i)   = alpha * sin(pos / (10000^(2i/d_model)))
    PE(pos, 2i+1) = alpha * cos(pos / (10000^(2i/d_model)))

    Attributes:
        input_size: Embedding dimension.
        max_len: Max length of the input sequences.
        alpha: Learnable scaling parameter applied to the positional encodings.

    Examples:
        >>> a = torch.rand((8, 120, 512))
        >>> enc = ScaledPositionalEncoding(input_size=a.shape[-1])
        >>> b = enc(a)
        >>> print(b.shape)
        torch.Size([1, 120, 512])
    """

    def __init__(self, input_size, max_len=2500):
        super().__init__()
        if input_size % 2 != 0:
            raise ValueError(
                f"Cannot use sin/cos positional encoding with odd dimensions (got dimensions={input_size})")

        self.max_len = max_len
        self.alpha = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, input_size, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, input_size, 2).float()
            * (-math.log(10000.0) / input_size))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]

        Returns:
            Tensor with the same shape as x, with positional encodings scaled by alpha added to it.
        """
        pos_emb = self.alpha * self.pe[:, :x.size(1)].clone().detach()
        return pos_emb




