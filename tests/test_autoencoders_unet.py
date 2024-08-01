import torch
import torch.nn as nn

from atompaint.autoencoders.unet import UNet, UNetBlock
from atompaint.field_types import (
        make_fourier_field_types, add_gates, CastToFourierFieldType,
)
from atompaint.nonlinearities import leaky_hard_shrink
from atompaint.pooling import FourierExtremePool3D
from atompaint.upsampling import R3Upsampling
from atompaint.time_embedding import SinusoidalEmbedding, LinearTimeActivation
from atompaint.vendored.escnn_nn_testing import (
        check_equivariance, get_exact_3d_rotations,
)
from escnn.nn import (
        GeometricTensor, FourierPointwise,
        SequentialModule, GatedNonLinearity1,
)
from escnn.gspaces import rot3dOnR3
from torchtest import assert_vars_change
from functools import partial

from test_time_embedding import ModuleWrapper, InputWrapper

# I didn't include an equivariance test here because the model is big enough 
# that its equivariance is too imperfect to test automatically.  However, in 
# experiment #87, I manually confirmed that many different configurations of 
# the U-Net are equivariant.

def test_unet():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    grid = so3.grid(type='thomson_cube', N=4)

    unet = UNet(
            field_types=make_fourier_field_types(
                gspace, 
                channels=[3, 1, 2],
                max_frequencies=[0, 1, 1],
            ),
            block_factory=lambda in_type, out_type: UNetBlock(
                in_type,
                time_activation=LinearTimeActivation(
                    time_dim=16,
                    activation=FourierPointwise(
                        out_type,
                        grid=grid,
                    )
                ),
                out_activation=FourierPointwise(
                    out_type,
                    grid=grid,
                ),
            ),
            block_repeats=2,
            downsample_factory=partial(
                FourierExtremePool3D,
                grid=grid,
                kernel_size=2,
            ),
            upsample_factory=partial(
                R3Upsampling,
                size_expr=lambda x: 2*x + 1,
            ),
            time_embedding=nn.Sequential(
                SinusoidalEmbedding(
                    out_dim=16,
                    min_wavelength=0.1,
                    max_wavelength=100,
                ),
                nn.Linear(16, 16),
                nn.ReLU(),
            ),
    )

    x = GeometricTensor(torch.randn(2, 3, 15, 15, 15), unet.in_type)
    t = torch.randn(2)
    y = torch.randn(2, 3, 15, 15, 15)

    assert_vars_change(
            model=ModuleWrapper(unet),
            loss_fn=nn.MSELoss(),
            optim=torch.optim.Adam(unet.parameters()),
            batch=(InputWrapper(x, t), y),
            device='cpu',
    )

