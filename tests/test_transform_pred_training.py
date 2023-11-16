import torch
import parametrize_from_file as pff

from test_transform_pred_models import classifier_equivariance
from atompaint.transform_pred.training import ResNetPredictorModule
from atompaint.vendored.escnn_nn_testing import (
        check_equivariance, get_exact_3d_rotations,
)

with_py = pff.Namespace()
with_ap = pff.Namespace('from atompaint.transform_pred.training import *')

@pff.parametrize(
        schema=pff.cast(
            model=with_ap.eval(defer=True),
            in_shape=with_py.eval,
            out_shape=with_py.eval,
        ),
)
@pff.parametrize(
        path='test_transform_pred_models.nt',
        key='test_classifier_equivariance',

        # We currently have the "cube" grid hard-coded into the predictors, so 
        # we can only handle test cases with this exact grid.
        schema=classifier_equivariance(require_grids=['cube']),
)
def test_predictor_equivariance(model, in_shape, out_shape, inputs):
    model = model().model
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
