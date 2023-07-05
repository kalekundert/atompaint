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
