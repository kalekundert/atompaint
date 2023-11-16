import re
from collections.abc import Sequence
from escnn.nn import FieldType, GeometricTensor
from escnn.gspaces import no_base_space

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

