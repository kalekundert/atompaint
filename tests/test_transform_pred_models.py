import atompaint.transform_pred.models as ap
import torch
import pytest
import parametrize_from_file as pff

from atompaint.transform_pred.models import (
        ViewPairEncoder, ViewPairClassifier,
        make_fourier_classifier_field_types, make_linear_fourier_layer,
)
from atompaint.encoders.cnn import FourierCnn
from atompaint.vendored.escnn_nn_testing import get_exact_3d_rotations
from escnn.nn import FieldType, FourierFieldType, GeometricTensor
from escnn.gspaces import no_base_space
from escnn.group import SO3, so3_group
from scipy.spatial.transform import Rotation
from math import radians
from functools import partial
from utils import *

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
        origins = coords(origins_str)
        g_rot_vec = vector(g_rot_vec_str)
        g_permut = integers(g_permut_str)

        so3 = so3_group()
        grid = so3.sphere_grid(grid_name)
        g = so3.element(g_rot_vec, 'EV')
        
        return so3, grid_name, grid, g, g_permut

    return schema


def test_view_pair_encoder_equivariance():
    cnn = FourierCnn(
            channels=[1, 1, 1, 1],
            conv_field_of_view=3,
            conv_stride=1,
            conv_padding=0,
            frequencies=2,
    )
    encoder = ViewPairEncoder(cnn)

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

        torch.testing.assert_close(gf_x.tensor, f_gx.tensor)

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
    mlp = ViewPairClassifier(
            layer_types=make_fourier_classifier_field_types(
                in_type=in_type,
                channels=1,
                max_frequencies=max_freq,
            ),
            layer_factory=partial(
                make_linear_fourier_layer,
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
