import re

from escnn.nn import FieldType, GeometricTensor
from escnn.gspaces import no_base_space
from contextlib import contextmanager
from collections.abc import Sequence

def get_scalar(scalar_or_seq, i):
    if isinstance(scalar_or_seq, Sequence):
        return scalar_or_seq[i]
    else:
        return scalar_or_seq

def identity(x):
    return x

@contextmanager
def eval_mode(model):
    start_in_train_mode = model.training

    if start_in_train_mode:
        model.eval()

    try:
        yield

    finally:
        if start_in_train_mode:
            model.train()

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

def partial_ch(f, **kwargs):
    """
    Return a copy of the given function where (i) the given arguments 
    have been pre-applied and (ii) the in/out channel arguments are specified 
    positionally rather than as keywords.

    The purpose of the in/out channel argument behavior is to smooth over a 
    difficulty in using the `torchyield` layer factory functions.  These 
    functions expect all of their arguments to be keywords, but the AtomPaint 
    encoder classes specify input and output channels as positional arguments.  
    As these layer factories are often wrapped in `partial()` anyways, this 
    slightly bizarre version of partial provides a succinct if hacky solution 
    to the problem.
    """
    from functools import wraps

    @wraps(f)
    def wrapper(in_channels, out_channels, **kwargs_wrapper):
        return f(
                in_channels=in_channels,
                out_channels=out_channels,
                **kwargs_wrapper,
                **kwargs,
        )

    return wrapper
