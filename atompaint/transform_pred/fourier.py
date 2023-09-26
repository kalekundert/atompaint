import torch
from escnn.nn import QuotientFourierPointwise, GeometricTensor

# TODO
# ====
# I want to refactor the Fourier transform code in ESCNN.  Currently, the only 
# Fourier transform functionality is in the nonlinearity modules.  There are 
# actually two such modules, `FourierPointwise` and `QuotientFourierPointwise`, 
# and probably >90% of the code for both is identical.
#
# My goal is to support the following:
#
# - Fourier pooling.  Right now the only option for pooling Fourier 
#   representations is average pooling.  Max pooling is often more expressive, 
#   but has to happen in the spatial domain.
#
# - Inverse Fourier transform.  This turned out to be a useful output layer for 
#   my application.
#
# Plans:
#
# - Create a module that just does forward and inverse Fourier transforms.
#
#   - Unfortunately, this can't be an `EquivariantModule`, because the output 
#     isn't a `GeometricTensor`.  I'd also kind of rather it not be a module at 
#     all, since there's two operations its can do.  I think this would require 
#     the caller to explicitly call some sort of registration function, though.
#

class QuotientInverseFourier(QuotientFourierPointwise):

    def forward(self, input: GeometricTensor) -> torch.Tensor:
        assert input.type == self.in_type
        
        shape = input.shape
        x_hat = input.tensor.view(
                shape[0],
                len(self.in_type),
                self.rho.size,
                *shape[2:],
        )
        return torch.einsum('bcf...,gf->bcg...', x_hat, self.A)
