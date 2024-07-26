import torch
import atompaint.diffusion.time_embedding as ap

from torchtest import assert_vars_change
from math import sin, cos, tau

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    emb = SinusoidalEmbedding(512)
    t = torch.arange(128)
    t_emb = emb(t)

    plt.imshow(t_emb.numpy())
    plt.show()


