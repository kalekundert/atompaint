from torch import Tensor
from torch.nn import Module
from escnn.nn import GeometricTensor
from einops import rearrange
from itertools import accumulate
from more_itertools import pairwise

class SymMeanStd(Module):
    # Something that constructs the `mean_std` tensor in a way that's 
    # equivariant.  The specific algorithm I have in mind:
    #
    # - The encoder outputs a tensor that can be split in half, with both 
    #   halves having the same channel types.
    #
    # - Use one half as the means.  This can be done directly, because the 
    #   means themselves are already equivariant.
    #
    # - Use the other half as the standard deviations.
    #
    #   - If the standard deviations are invariant, then they won't break 
    #     equivariance when added to the means.  There may be other ways to do 
    #     this, but this one is simple.
    #
    #   - Calculate the norm of each equivariant vector.  Use that value as the 
    #     standard deviation for each corresponding mean.

    def __init__(self, field_type):
        super().__init__()
        self.in_type = self.out_type = field_type

        reprs = field_type.representations
        if len(reprs) % 2 != 0:
            raise ValueError(f"can't split field type into two matching halves\n• odd number of representations\n• field type: {field_type}")
        i = len(reprs) // 2
        if reprs[:i] != reprs[i:]:
            raise ValueError(f"can't split field type into two matching halves\n• field type: {field_type}")

        self._repr_slices = [
                slice(i,j)
                for i, j in pairwise(
                    accumulate((r.size for r in reprs[i:]), initial=0)
                )
        ]

    def forward(self, x: GeometricTensor) -> Tensor:
        assert x.type == self.in_type

        y = rearrange(x.tensor, 'b (m c) ... -> b m c ...', m=2)

        # We have to create a new tensor for the invariant standard deviations, 
        # otherwise the backwards pass will fail due to in-place updates.
        z = y.clone()

        for ij in self._repr_slices:
            z[:, 1:2, ij] = y[:, 1:2, ij].norm(dim=2, keepdim=True)

        return z

        





