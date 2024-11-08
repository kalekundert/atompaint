import torch
import parametrize_from_file as pff

from atompaint.classifiers.neighbor_loc import (
        SymViewPairEncoder, SymViewPairClassifier, 
        make_neighbor_loc_model, make_fourier_mlp_field_types

)
from atompaint.encoders import SymEncoder
from atompaint.layers import (
        sym_conv_bn_fourier_layer, sym_conv_layer, sym_linear_fourier_layer, 
)
from atompaint.field_types import make_fourier_field_types
from atompaint.vendored.escnn_nn_testing import get_exact_3d_rotations
from escnn.nn import FourierFieldType, GeometricTensor
from escnn.gspaces import rot3dOnR3, no_base_space
from escnn.group import so3_group
from functools import partial
from utils import *

with_py = pff.Namespace()
with_ap = pff.Namespace('from atompaint.classifiers.neighbor_loc import *')

def classifier_equivariance(*, require_grids=None):

    def schema(params):
        grid = params.pop('grid')

        if require_grids and grid not in require_grids:
            params['marks'] = 'skip'

        params['inputs'] = partial(
                load,
                grid,
                params.pop('origins'),
                params.pop('g_rot_vec'),
                params.pop('g_permut'),
        )
        return params

    def load(grid_name, origins_str, g_rot_vec_str, g_permut_str):
        g_rot_vec = vector(g_rot_vec_str)
        g_permut = integers(g_permut_str)

        so3 = so3_group()
        grid = so3.sphere_grid(grid_name)
        g = so3.element(g_rot_vec, 'EV')
        
        return so3, grid_name, grid, g, g_permut

    return schema


def test_view_pair_encoder_equivariance():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    ift_grid = so3.grid('thomson_cube', N=4)

    cnn = SymEncoder(
            in_channels=1,
            field_types=make_fourier_field_types(
                gspace=gspace, 
                channels=[1, 1, 1],
                max_frequencies=2,
            ),
            block_factories=partial(
                sym_conv_bn_fourier_layer,
                ift_grid=ift_grid,
            ),
            tail_factory=sym_conv_layer,
    )
    encoder = SymViewPairEncoder(cnn)

    in_type = encoder.in_type
    so3 = in_type.fibergroup

    x = torch.randn(1, 2, 1, 7, 7, 7)

    for g in get_exact_3d_rotations(so3):
        f_x = encoder(x)
        gf_x = f_x.transform(g)

        # The input has too many dimensions to transform, due to the addition 
        # of a dimension to distinguish between the two views (this is the 
        # second dimension).  Here we work around that by temporarily removing 
        # the first dimension, so that the "view" dimension will be treated as 
        # the minibatch dimension.  This works because there's only 1 
        # minibatch; if there were more we'd have to do some fancier reshaping.
        gx = in_type.transform(x[0], g).reshape(*x.shape)
        f_gx = encoder(gx)

        assert f_x.shape == (1, 70)
        assert f_gx.shape == (1, 70)
        assert gf_x.shape == (1, 70)

        torch.testing.assert_close(
                gf_x.tensor,
                f_gx.tensor,
                atol=1e-4,
                rtol=1.3e-6,
        )

@pff.parametrize(
        key='test_classifier_equivariance',
        schema=classifier_equivariance(),
)
def test_view_pair_classifier_equivariance(inputs):
    so3, _, grid, g, g_permut = inputs()

    gspace = no_base_space(so3)
    max_freq = 2
    in_type = FourierFieldType(
            gspace=gspace,
            channels=2,
            bl_irreps=so3.bl_irreps(max_freq),
    )
    mlp = SymViewPairClassifier(
            layer_types=make_fourier_mlp_field_types(
                in_type=in_type,
                channels=1,
                max_frequencies=max_freq,
            ),
            layer_factory=partial(
                sym_linear_fourier_layer,
                ift_grid=so3.grid('thomson_cube', 4),
            ),
            logits_max_freq=max_freq,
            logits_grid=grid,
    )

    x = torch.randn(2, 70)
    x = GeometricTensor(x, in_type)

    f_x = mlp.forward(x)
    gf_x = f_x[:, g_permut]

    gx = x.transform(g)
    f_gx = mlp.forward(gx)

    torch.testing.assert_close(gf_x, f_gx)

@pff.parametrize(
        schema=pff.cast(
            model=with_py.eval,
            in_shape=with_py.eval,
            out_shape=with_py.eval,
        ),
)
@pff.parametrize(
        key='test_classifier_equivariance',

        # We currently have the "cube" grid hard-coded into the predictors, so 
        # we can only handle test cases with this exact grid.
        schema=[
            classifier_equivariance(require_grids=['cube']),
        ],
)
def test_model_equivariance(model, in_shape, out_shape, inputs):
    model = make_neighbor_loc_model(**model)
    _, _, _, g, g_permut = inputs()

    b, v, *img_shape = in_shape

    # If there's only one training example, the batch normalization will be 
    # unable to calculate variances.
    assert b > 1
    assert v == 2

    # The input has an extra "view" dimension that it's incompatible with the 
    # transformation functions provided by escnn.  We work around this by 
    # putting the views in the batch dimension and reshaping after the 
    # transformation.
    x0 = torch.randn(b * v, *img_shape)
    x = x0.reshape(*in_shape)

    f_x = model(x)
    gf_x = f_x[:, g_permut]

    assert f_x.shape == out_shape

    gx = model.in_type.transform(x0, g).reshape(*x.shape)
    f_gx = model(gx)

    assert f_gx.shape == out_shape

    torch.testing.assert_close(gf_x, f_gx)
