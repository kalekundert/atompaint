import atompaint.eval as ap
import pytest

from torch import tensor
from torch.testing import assert_close

@pytest.mark.parametrize(
    'x, b', [
        (tensor([1., 2., 3.]), 1),
        (tensor([1., 2., 3.]), 2),
        (tensor([1., 2., 3.]), 3),
        (tensor([1., 2., 3.]), 4),

        (tensor([[1., 2.], [3., 4.], [5., 6.]]), 1),
        (tensor([[1., 2.], [3., 4.], [5., 6.]]), 2),
        (tensor([[1., 2.], [3., 4.], [5., 6.]]), 3),
        (tensor([[1., 2.], [3., 4.], [5., 6.]]), 4),
    ],
)
def test_with_max_batch_size(x, b):

    def f(x):
        return x + 1

    g = ap.with_max_batch_size(f, max_batch_size=b)

    assert_close(f(x), g(x))


