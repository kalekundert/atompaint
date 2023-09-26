import atompaint.transform_pred.models as ap
import torch
import pytest
import parametrize_from_file as pff

from test_datasets_coords import coords, vector
from test_transform_pred_datasets_classification import check_origins
from atompaint.transform_pred.datasets.classification import make_view_frames_ab
from escnn.nn import FieldType, GeometricTensor
from escnn.gspaces import no_base_space
from escnn.group import SO3, so3_group
from scipy.spatial.transform import Rotation
from math import radians
from functools import partial
from utils import *

with_ap = pff.Namespace('from atompaint.transform_pred import *')

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
        
        # Make sure the grid points are what they're supposed to be.  This 
        # depends on the internal implementation of `so3.sphere_grid()`, and if 
        # it changes for any reason, the `g_permut` test parameter will have to 
        # be updated manually.
        check_origins(make_view_frames_ab(grid, 1), origins)

        return so3, grid_name, grid, g, g_permut

    return schema


@pff.parametrize(
        key='test_classifier_equivariance',
        schema=classifier_equivariance(require_grids=['cube']),
)
def test_transform_pred_equivariance(inputs):
    _, _, _, g, g_permut = inputs()

    transform_pred = ap.TransformationPredictor(
            frequencies=1,
            conv_channels=[1, 1, 1, 1],
            conv_field_of_view=3,
            conv_stride=2,
            mlp_channels=[1],
    )

    # The input has an extra "view" dimension that it's incompatible with the 
    # transformation functions provided by escnn.  We work around this by 
    # putting the regions in the batch dimension and reshaping after the 
    # transformation.
    img_shape = 1, 15, 15, 15
    x0 = torch.randn(2, *img_shape)
    x = x0.reshape(1, 2, *img_shape)

    f_x = transform_pred(x)
    gf_x = f_x[:, g_permut]

    gx = transform_pred.in_type.transform(x0, g).reshape(1, 2, *img_shape)
    f_gx = transform_pred(gx)

    torch.testing.assert_close(gf_x, f_gx)

@pff.parametrize(
        key='test_classifier_equivariance',
        schema=classifier_equivariance(),
)
def test_view_classifier_mlp_equivariance(inputs):
    so3, _, grid, g, g_permut = inputs()

    gspace = no_base_space(so3)
    irreps = so3.bl_irreps(1)
    fourier_repr = so3.spectral_regular_representation(*irreps)
    so3_fields = [
            FieldType(gspace, [fourier_repr]),
    ]
    layer_factory = lambda in_type, out_type: [Linear(in_type, out_type)]

    mlp = ap.ViewClassifierMlp(
            so3_fields=so3_fields,
            layer_factory=layer_factory,
            fourier_irreps=irreps,
            fourier_grid=grid,
    )

    x = torch.randn(1, so3_fields[0].size)
    x = GeometricTensor(x, so3_fields[0])

    f_x = mlp.forward(x)
    gf_x = f_x[:, g_permut]

    gx = x.transform(g)
    f_gx = mlp.forward(gx)

    torch.testing.assert_close(gf_x, f_gx)
