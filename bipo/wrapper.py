"""Block wrapper used while BiPO learns its steering vectors.

One trainable vector per wrapped layer, added to every position of that layer's output:

    h <- h + multiplier * v_layer

The trainer flips `multiplier` between +1 and -1 from batch to batch, which is what makes
`-v` a usable negative steering vector at generation time. Generation does not use this
wrapper: train.py writes the vectors in the CAA layout and models/wrapper.py applies them.
"""

from __future__ import annotations

import torch
from torch import nn


class BiPOBlockWrapper(nn.Module):
    """Wrap a decoder block with a trainable steering vector."""

    def __init__(self, block: nn.Module, hidden_dim: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.block = block
        self.multiplier = 1.0
        # fp32 even when the frozen base model runs in bf16: an Adam update at this
        # learning rate is smaller than a bf16 ulp around 1.0 and would be rounded away.
        # Cast down to the activation dtype at injection time instead.
        self.vec = nn.Parameter(torch.zeros(hidden_dim, dtype=dtype))

    def set_multiplier(self, multiplier: float) -> None:
        self.multiplier = multiplier

    def forward(self, hidden_states, *args, **kwargs):
        output = self.block(hidden_states, *args, **kwargs)
        out_tensor = output[0] if isinstance(output, tuple) else output

        steered = out_tensor + (self.multiplier * self.vec).to(out_tensor.dtype)

        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    def __getattr__(self, name):
        # nn.Module.__getattr__ resolves only parameters and submodules; everything else
        # the decoder layer exposes is forwarded to the wrapped block.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.block, name)
