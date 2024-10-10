import pytest
from atompaint.utils import require_nested_list, partial_grid, rows, cols
from utils import *

@pff.parametrize(
        schema=[
            pff.cast(
                x=with_py.eval,
                y=with_py.eval,
                rows=int,
                cols=int,
            ),
            pff.defaults(rows=None, cols=None),
            pff.error_or('y'),
        ],
)
def test_require_nested_list(x, y, rows, cols, error):
    with error:
        assert require_nested_list(x, rows=rows, cols=cols) == y

def test_partial_grid_no_args():
    g = partial_grid()(dict)

    assert len(g) == 1
    assert len(g[0]) == 1
    assert g[0][0]() == {}

def test_partial_grid_shape():
    g = partial_grid(2, 3)(dict)

    assert len(g) == 2
    assert len(g[0]) == 3
    assert len(g[1]) == 3
    assert g[0][0]() == {}
    assert g[0][1]() == {}
    assert g[0][2]() == {}
    assert g[1][0]() == {}
    assert g[1][1]() == {}
    assert g[1][2]() == {}

def test_partial_grid_rows():
    g = partial_grid(cols=2)(dict, a=rows(1, 2), b=rows(3,4), c=5)

    assert len(g) == 2
    assert len(g[0]) == 2
    assert len(g[1]) == 2
    assert g[0][0]() == dict(a=1, b=3, c=5)
    assert g[0][1]() == dict(a=1, b=3, c=5)
    assert g[1][0]() == dict(a=2, b=4, c=5)
    assert g[1][1]() == dict(a=2, b=4, c=5)

def test_partial_grid_cols():
    g = partial_grid(rows=2)(dict, a=cols(1, 2), b=cols(3, 4), c=5)

    assert len(g) == 2
    assert len(g[0]) == 2
    assert len(g[1]) == 2
    assert g[0][0]() == dict(a=1, b=3, c=5)
    assert g[0][1]() == dict(a=2, b=4, c=5)
    assert g[1][0]() == dict(a=1, b=3, c=5)
    assert g[1][1]() == dict(a=2, b=4, c=5)

def test_partial_grid_rows_cols():
    g = partial_grid()(dict, a=rows(1, 2), b=cols(3, 4), c=5)

    assert len(g) == 2
    assert len(g[0]) == 2
    assert len(g[1]) == 2
    assert g[0][0]() == dict(a=1, b=3, c=5)
    assert g[0][1]() == dict(a=1, b=4, c=5)
    assert g[1][0]() == dict(a=2, b=3, c=5)
    assert g[1][1]() == dict(a=2, b=4, c=5)

def test_partial_grid_inconsistency_shape_row():
    with pytest.raises(ValueError, match="expected 'a' to have 2 values, but got 3"):
        partial_grid(rows=2)(dict, a=rows(1, 2, 3))

def test_partial_grid_inconsistency_row_row():
    with pytest.raises(ValueError, match="expected 'b' to have 2 values, but got 3"):
        partial_grid()(dict, a=rows(1, 2), b=rows(3, 4, 5))

