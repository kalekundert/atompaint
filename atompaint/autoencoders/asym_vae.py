from torch import Tensor
from torch.nn import Module
from einops import rearrange

class AsymMeanStd(Module):

    def forward(self, x: Tensor) -> Tensor:
        return rearrange(x, 'b (m c) ... -> b m c ...', m=2)

        






