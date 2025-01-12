import torch
import torch.nn as nn
import atompaint.conditioning as ap

from escnn.nn import (
        FourierFieldType, GeometricTensor,
        FourierPointwise,
)
from escnn.gspaces import rot3dOnR3
from atompaint.vendored.escnn_nn_testing import (
        check_equivariance, get_exact_3d_rotations,
)
from torchtest import assert_vars_change
from pytest import raises
from math import sin, cos, tau

class ModuleWrapper(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input_wrapper):
        x = self.module(*input_wrapper.args, **input_wrapper.kwargs)
        return x.tensor if isinstance(x, GeometricTensor) else x

class InputWrapper:

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def to(self, device):
        self.args = [
                x.to(device) if isinstance(x, torch.Tensor) else x
                for x in self.args
        ]
        self.kwargs = {
                k: x.to(device) if isinstance(x, torch.Tensor) else x
                for k, x in self.kwargs.items()
        }
        return self

class NoOpModel(nn.Module):

    def forward(self, x, y):
        return x, y

def test_conditioned_model():
    cond_model = ap.ConditionedModel(
            NoOpModel(),
            noise_embedding=nn.Identity(),
    )

    x_in = torch.randn(2, 3, 4, 4, 4)
    y_in = torch.randn(2, 3)

    with raises(ValueError):
        cond_model(x_in, y_in, label=y_in)
    with raises(ValueError):
        cond_model(x_in, y_in, x_self_cond=x_in)

    x_out, y_out = cond_model(x_in, y_in)

    torch.testing.assert_close(x_out, x_in)
    torch.testing.assert_close(y_out, y_in)

def test_confitioned_model_label():
    cond_model = ap.ConditionedModel(
            NoOpModel(),
            noise_embedding=nn.Identity(),
            label_embedding=nn.Identity(),
    )

    x_in = torch.randn(2, 3, 4, 4, 4)
    noise_in = torch.randn(2, 3)
    label_in = torch.randn(2, 3)

    with raises(ValueError):
        cond_model(x_in, noise_in)
    with raises(ValueError):
        cond_model(x_in, noise_in, x_self_cond=x_in)

    x_out, y_out = cond_model(x_in, noise_in, label=label_in)

    torch.testing.assert_close(x_out, x_in)
    torch.testing.assert_close(y_out, noise_in + label_in)

def test_confitioned_model_self_cond():
    cond_model = ap.ConditionedModel(
            NoOpModel(),
            noise_embedding=nn.Identity(),
            allow_self_cond=True,
    )

    x_in = torch.randn(2, 3, 4, 4, 4)
    y_in = torch.randn(2, 3)

    x_out, y_out = cond_model(x_in, y_in, x_self_cond=x_in)

    torch.testing.assert_close(x_out, torch.cat([x_in, x_in], dim=1))
    torch.testing.assert_close(y_out, y_in)

def test_add_condition_to_image():
    from torch.nn.init import eye_, constant_

    f = ap.AddConditionToImage(
            cond_dim=4,
            channel_dim=3,
            affine=True,
    )

    # I want to make sure that the condition embedding is being added to the 
    # image in the way I think it should be.  Unfortunately, I can't think of a 
    # way to do this while being agnostic to the internal structure of the 
    # module.  The following code initializes the module parameters such that 
    # the linear layers change their inputs in predictable ways.

    for name, p in f.named_parameters():
        if name.endswith('weight'):
            eye_(p)
        else:
            constant_(p, 0)

    x = torch.ones(2, 3, 2, 2, 2)
    y = torch.tensor([
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
    ])
    xy = f(x, y)

    # The condition only affects channel dimension; all voxels within a channel 
    # should have the same value.
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

    torch.testing.assert_close(xy, expected)

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

def test_fourier_conditioned_activation():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    in_type = FourierFieldType(gspace, 2, so3.bl_irreps(1))
    grid = so3.grid(type='thomson_cube', N=4)

    cond_act = ap.FourierConditionedActivation(
            in_type=in_type,
            grid=grid,
            cond_dim=16,
    )

    x = GeometricTensor(torch.randn(2, 20, 3, 3, 3), in_type)
    y = torch.randn(2, 16)
    xy = torch.randn(2, 20, 3, 3, 3)

    assert_vars_change(
            model=ModuleWrapper(cond_act),
            loss_fn=nn.MSELoss(),
            optim=torch.optim.Adam(cond_act.parameters()),
            batch=(InputWrapper(x, y), xy),
            device='cpu',
    )

    check_equivariance(
            lambda x: cond_act(x, y),
            in_tensor=(2, 20, 3, 3, 3),
            in_type=cond_act.in_type,
            out_shape=(2, 20, 3, 3, 3),
            out_type=cond_act.out_type,
            group_elements=get_exact_3d_rotations(so3),
            atol=1e-5,
    )

def test_linear_conditioned_activation():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    in_type = FourierFieldType(gspace, 2, so3.bl_irreps(1))
    grid = so3.grid(type='thomson_cube', N=4)

    cond_act = ap.LinearConditionedActivation(
            cond_dim=16,
            activation=FourierPointwise(
                in_type=in_type,
                grid=grid,
            ),
    )

    x = GeometricTensor(torch.randn(2, 20, 3, 3, 3), in_type)
    y = torch.randn(2, 16)
    xy = torch.randn(2, 20, 3, 3, 3)

    assert_vars_change(
            model=ModuleWrapper(cond_act),
            loss_fn=nn.MSELoss(),
            optim=torch.optim.Adam(cond_act.parameters()),
            batch=(InputWrapper(x, y), xy),
            device='cpu',
    )

    check_equivariance(
            lambda x: cond_act(x, y),
            in_tensor=(2, 20, 3, 3, 3),
            in_type=cond_act.in_type,
            out_shape=(2, 20, 3, 3, 3),
            out_type=cond_act.out_type,
            group_elements=get_exact_3d_rotations(so3),
    )

def test_gated_conditioned_activation():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    in_type = FourierFieldType(gspace, 2, so3.bl_irreps(1), unpack=True)

    cond_act = ap.GatedConditionedActivation(in_type, cond_dim=16)

    x = GeometricTensor(torch.randn(2, 20, 3, 3, 3), in_type)
    y = torch.randn(2, 16)
    xy = torch.randn(2, 20, 3, 3, 3)

    assert_vars_change(
            model=ModuleWrapper(cond_act),
            loss_fn=nn.MSELoss(),
            optim=torch.optim.Adam(cond_act.parameters()),
            batch=(InputWrapper(x, y), xy),
            device='cpu',
    )

    check_equivariance(
            lambda x: cond_act(x, y),
            in_tensor=(2, 20, 3, 3, 3),
            in_type=cond_act.in_type,
            out_shape=(2, 20, 3, 3, 3),
            out_type=cond_act.out_type,
            group_elements=get_exact_3d_rotations(so3),
    )


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    emb = ap.SinusoidalEmbedding(
            out_dim=128,
            min_wavelength=1e-1,
            max_wavelength=1e2,
    )
    t = 1.2 * torch.randn(100) - 1.2
    t = torch.sort(t)[0]
    t_emb = emb(t)

    plt.imshow(t_emb.numpy())
    plt.show()


