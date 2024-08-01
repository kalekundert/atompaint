import torch
import torch.nn as nn
import atompaint.time_embedding as ap

from torchtest import assert_vars_change
from escnn.nn import (
        FourierFieldType, GeometricTensor,
        InverseFourierTransform, FourierTransform, FourierPointwise,
)
from escnn.gspaces import rot3dOnR3
from atompaint.vendored.escnn_nn_testing import (
        check_equivariance, get_exact_3d_rotations,
)
from math import sin, cos, tau

class ModuleWrapper(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        x, t = inputs
        y = self.module(x, t)
        return y.tensor if isinstance(y, GeometricTensor) else y

class InputWrapper:

    def __init__(self, *args):
        self.inputs = args

    def __iter__(self):
        yield from self.inputs

    def to(self, device):
        self.inputs = [x.to(device) for x in self.inputs]
        return self

def test_add_time_to_image():
    from torch.nn.init import eye_, constant_

    f = ap.AddTimeToImage(4, 3)

    # I want to make sure that the time embedding is being added to the image 
    # in the way I think it should be.  Unfortunately, I can't think of a way 
    # to do this while being agnostic to the internal structure of the module.  
    # The following code initializes the module parameters such that the linear 
    # layers change their inputs in predictable ways.

    for name, p in f.named_parameters():
        if name.endswith('weight'):
            eye_(p)
        else:
            constant_(p, 0)

    x = torch.ones(2, 3, 2, 2, 2)
    t = torch.tensor([
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
    ])
    y = f(x, t)

    # Time only affects channel dimension; all voxels within a channel should 
    # have the same value.
    expected = torch.tensor([
        [
            [[[1+4, 1+4], [1+4, 1+4]], [[1+4, 1+4], [1+4, 1+4]]],
            [[[  2,   2], [  2,   2]], [[  2,   2], [  2,   2]]],
            [[[  3,   3], [  3,   3]], [[  3,   3], [  3,   3]]],
        ], [
            [[[5+8, 5+8], [5+8, 5+8]], [[5+8, 5+8], [5+8, 5+8]]],
            [[[  6,   6], [  6,   6]], [[  6,   6], [  6,   6]]],
            [[[  7,   7], [  7,   7]], [[  7,   7], [  7,   7]]],
        ],
    ], dtype=y.dtype)

    torch.testing.assert_close(y, expected)

def test_sinusoidal_positional_embedding():
    emb = ap.SinusoidalEmbedding(4, max_wavelength=8)
    t = torch.arange(9)
    t_emb = emb(t)

    expected = torch.tensor([
            [sin(0),        sin(0),       cos(0),       cos(0)],
            [sin(1*tau/4),  sin(1*tau/8), cos(1*tau/4), cos(1*tau/8)],
            [sin(2*tau/4),  sin(2*tau/8), cos(2*tau/4), cos(2*tau/8)],
            [sin(3*tau/4),  sin(3*tau/8), cos(3*tau/4), cos(3*tau/8)],
            [sin(0),        sin(4*tau/8), cos(0),       cos(4*tau/8)],
            [sin(1*tau/4),  sin(5*tau/8), cos(1*tau/4), cos(5*tau/8)],
            [sin(2*tau/4),  sin(6*tau/8), cos(2*tau/4), cos(6*tau/8)],
            [sin(3*tau/4),  sin(7*tau/8), cos(3*tau/4), cos(7*tau/8)],
            [sin(0),        sin(0),       cos(0),       cos(0)],
    ], dtype=t_emb.dtype)

    torch.testing.assert_close(t_emb, expected)

def test_fourier_time_activation():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    in_type = FourierFieldType(gspace, 2, so3.bl_irreps(1))
    grid = so3.grid(type='thomson_cube', N=4)

    time_act = ap.FourierTimeActivation(
            in_type=in_type,
            grid=grid,
            time_dim=16,
    )

    x = GeometricTensor(torch.randn(2, 20, 3, 3, 3), in_type)
    t = torch.randn(2, 16)
    y = torch.randn(2, 20, 3, 3, 3)

    assert_vars_change(
            model=ModuleWrapper(time_act),
            loss_fn=nn.MSELoss(),
            optim=torch.optim.Adam(time_act.parameters()),
            batch=(InputWrapper(x, t), y),
            device='cpu',
    )

def test_fourier_time_activation_equivariance():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    in_type = FourierFieldType(gspace, 2, so3.bl_irreps(1))
    grid = so3.grid(type='thomson_cube', N=4)

    time_act = ap.FourierTimeActivation(
            in_type=in_type,
            grid=grid,
            time_dim=16,
    )
    t = torch.randn(2, 16)

    check_equivariance(
            lambda x: time_act(x, t),
            in_tensor=(2, 20, 3, 3, 3),
            in_type=time_act.in_type,
            out_shape=(2, 20, 3, 3, 3),
            out_type=time_act.out_type,
            group_elements=get_exact_3d_rotations(so3),
    )

def test_linear_time_activation():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    in_type = FourierFieldType(gspace, 2, so3.bl_irreps(1))
    grid = so3.grid(type='thomson_cube', N=4)

    time_act = ap.LinearTimeActivation(
            time_dim=16,
            activation=FourierPointwise(
                in_type=in_type,
                grid=grid,
            ),
    )

    x = GeometricTensor(torch.randn(2, 20, 3, 3, 3), in_type)
    t = torch.randn(2, 16)
    y = torch.randn(2, 20, 3, 3, 3)

    assert_vars_change(
            model=ModuleWrapper(time_act),
            loss_fn=nn.MSELoss(),
            optim=torch.optim.Adam(time_act.parameters()),
            batch=(InputWrapper(x, t), y),
            device='cpu',
    )

def test_linear_time_activation_equivariance():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    in_type = FourierFieldType(gspace, 2, so3.bl_irreps(1))
    grid = so3.grid(type='thomson_cube', N=4)

    time_act = ap.LinearTimeActivation(
            time_dim=16,
            activation=FourierPointwise(
                in_type=in_type,
                grid=grid,
            ),
    )
    t = torch.randn(2, 16)

    check_equivariance(
            lambda x: time_act(x, t),
            in_tensor=(2, 20, 3, 3, 3),
            in_type=time_act.in_type,
            out_shape=(2, 20, 3, 3, 3),
            out_type=time_act.out_type,
            group_elements=get_exact_3d_rotations(so3),
    )

def test_gated_time_activation():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    in_type = FourierFieldType(gspace, 2, so3.bl_irreps(1), unpack=True)

    time_act = ap.GatedTimeActivation(in_type, time_dim=16)

    x = GeometricTensor(torch.randn(2, 20, 3, 3, 3), in_type)
    t = torch.randn(2, 16)
    y = torch.randn(2, 20, 3, 3, 3)

    assert_vars_change(
            model=ModuleWrapper(time_act),
            loss_fn=nn.MSELoss(),
            optim=torch.optim.Adam(time_act.parameters()),
            batch=(InputWrapper(x, t), y),
            device='cpu',
    )

def test_gated_time_activation_equivariance():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    in_type = FourierFieldType(gspace, 2, so3.bl_irreps(1), unpack=True)

    time_act = ap.GatedTimeActivation(in_type, time_dim=16)
    t = torch.randn(2, 16)

    check_equivariance(
            lambda x: time_act(x, t),
            in_tensor=(2, 20, 3, 3, 3),
            in_type=time_act.in_type,
            out_shape=(2, 20, 3, 3, 3),
            out_type=time_act.out_type,
            group_elements=get_exact_3d_rotations(so3),
    )


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    emb = SinusoidalEmbedding(512)
    t = torch.arange(128)
    t_emb = emb(t)

    plt.imshow(t_emb.numpy())
    plt.show()


