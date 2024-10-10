import re
from collections.abc import Sequence
from escnn.nn import FieldType, GeometricTensor
from escnn.gspaces import no_base_space
from functools import partial

from typing import Optional

def get_scalar(scalar_or_seq, i):
    if isinstance(scalar_or_seq, Sequence):
        return scalar_or_seq[i]
    else:
        return scalar_or_seq

def identity(x):
    return x

def parse_so3_grid(group, name):
    if name in {'ico', 'cube', 'tetra'}:
        return group.grid(name)

    if m := re.match(r'(thomson|hopf|fibonacci|rand)_(\d+)', name):
        n = int(m.group(2))
        return group.grid(m.group(1), N=n)

    if m := re.match(r'(thomson_cube)_(\d+)', name):
        n = int(m.group(2))
        assert n % 24 == 0
        return group.grid(m.group(1), N=n//24)

    raise ValueError(f"unknown grid: {name}")

def flatten_base_space(geom_tensor):
    # I'd like to add this as a method of the `GeometricTensor` class.
    tensor = geom_tensor.tensor
    field_type = geom_tensor.type
    spatial_dims = field_type.gspace.dimensionality

    assert geom_tensor.coords is None
    assert all(x == 1 for x in tensor.shape[-spatial_dims:])

    new_shape = tensor.shape[:-spatial_dims]
    new_type = FieldType(
            no_base_space(field_type.gspace.fibergroup),
            field_type.representations,
    )

    return GeometricTensor(
            tensor.reshape(new_shape),
            new_type,
    )

def require_nested_list(x, rows=None, cols=None):
    """
    Require that the given input `x` is a list-of-lists, possibly with a 
    specific number of rows and columns.

    The main role of this function is to allow `x` to be specified more 
    succinctly when the same values are used for each row/column.  Here are the 
    rules for how `x` is interpreted:
    
    - If the input is not a list, it will be made into a list-of-lists.  If the 
      numbers of rows/cols is specified, the output  of the given shape.  
    
    - If the input is a list, it must be a list-of-lists.  If a shape is 
      specified, the input must have that shape.  One special case: if the 
      input has only one row and more are expected, that row will be repeated 
      to get the expected number.
    """
    
    if isinstance(x, list):
        if rows and len(x) != rows:
            if len(x) == 1:
                x *= rows
            else:
                raise ValueError(f"expected {rows} rows, but got {len(x)}")

        for row in x:
            if not isinstance(row, list):
                raise ValueError(f"expected list-of-lists, but got list-of: {type(row)}")
            if cols and len(row) != cols:
                raise ValueError(f"expected {cols} columns, but got {len(row)}")

        return x

    else:
        grid = []

        for i in range(rows or 1):
            row = []
            grid.append(row)

            for j in range(cols or 1):
                row.append(x)

        return grid

class partial_grid:
    """
    Create a grid of partial functions, where the arguments to each vary by row 
    and by column.

    To specify that an argument should vary by row or by column, use the `rows` 
    and `cols` containers.  Any other argument will be used for each cell 
    without any interpretation.  To illustrate this, here's an example of a 
    2x2 grid where the `a` argument varies by row, the `b` argument varies by 
    column, and the `c` argument is the same for each cell (despite being at 
    iterable value):

        >>> g = partial_grid()(dict, a=rows(1, 2), b=cols(3, 4), c=(5, 6))
        >>> g[0][0]()
        {'a': 1, 'b': 3, 'c': (5, 6)}
        >>> g[0][1]()
        {'a': 1, 'b': 4, 'c': (5, 6)}
        >>> g[1][0]()
        {'a': 2, 'b': 3, 'c': (5, 6)}
        >>> g[1][1]()
        {'a': 2, 'b': 4, 'c': (5, 6)}

    In the above example, the size of the grid is inferred from the
    `rows(1, 2)` and `cols(3, 4)` arguments.  It's also possible to explicitly 
    the grid size.  This is useful when, for example, you want multiple 
    columns, but don't want any arguments to vary by column:

        >>> g = partial_grid(cols=2)(dict, a=rows(1, 2))
        >>> g[0][0]()
        {'a': 1}
        >>> g[0][1]()
        {'a': 1}
        >>> g[1][0]()
        {'a': 2}
        >>> g[1][1]()
        {'a': 2}
    """

    class rows:
        def __init__(self, *items):
            self.items = items

    class cols:
        def __init__(self, *items):
            self.items = items

    def __init__(self, rows: Optional[int] = None, cols: Optional[int] = None):
        """
        Optionally specify a shape for the grid.

        If a number of rows/columns is specified, then all of the keyword 
        arguments passed to the function---via `__call___()`---must be 
        consistent with that number.  If no number is specified, the shape of 
        the grid will be determined by the keyword arguments.
        """
        self.num_rows = rows
        self.num_cols = cols

    def __call__(self, f, /, **kwargs):
        """
        Create a grid of partial function evaluations.
        
        All arguments to the partial function must be keyword arguments.  
        Arguments that are instances of `partial_grid.rows` will vary by row, 
        and those that are instances `partial_grid.cols` will vary by column.  
        All other arguments will be passed verbatim to each cell.
        """
        grid_kwargs = {}
        row_kwargs = self.num_rows and [{} for _ in range(self.num_rows)]
        col_kwargs = self.num_cols and [{} for _ in range(self.num_cols)]

        def add_kwarg(existing_kwargs, new_k, new_vs):
            if existing_kwargs is None:
                return [{new_k: v} for v in new_vs]

            else:
                if len(new_vs) != len(existing_kwargs):
                    raise ValueError(f"expected {new_k!r} to have {len(existing_kwargs)} values, but got {len(new_vs)}")
                return [
                        kw | {new_k: v}
                        for kw, v in zip(existing_kwargs, new_vs)
                ]

        for k, v in kwargs.items():
            if isinstance(v, partial_grid.rows):
                row_kwargs = add_kwarg(row_kwargs, k, v.items)
            elif isinstance(v, partial_grid.cols):
                col_kwargs = add_kwarg(col_kwargs, k, v.items)
            else:
                grid_kwargs[k] = v

        if row_kwargs is None:
            row_kwargs = [{}]
        if col_kwargs is None:
            col_kwargs = [{}]

        grid = []

        for row_kwargs_i in row_kwargs:
            row = []
            grid.append(row)

            for col_kwargs_j in col_kwargs:
                g = partial(f, **row_kwargs_i, **col_kwargs_j, **grid_kwargs)
                row.append(g)

        return grid

rows = partial_grid.rows
cols = partial_grid.cols
