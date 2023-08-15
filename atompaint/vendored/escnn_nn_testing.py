import torch
import numpy as np

from torch import Tensor
from torch.nn import Module
from escnn.nn import FieldType, GeometricTensor
from escnn.group import Group, GroupElement
from typing import Optional

# Is there a way to choose only those group elements to those that don't 
# require interpolation?

# These are probably too tight.
ATOL_DEFAULT = 1e-7
RTOL_DEFAULT = 1e-5

class TestCases:

    def __init__(
            self, *,
            module: Module,
            in_tensor: Tensor | GeometricTensor | tuple[int, int] = None,
            in_type: FieldType = None,
            out_type: FieldType = None,
            group_elements: int | list[GroupElement] = 20,
    ):
        self.module = module

        if in_type is None:
            in_type = module.in_type

        if in_tensor is None:
            in_tensor = ()
        if isinstance(in_tensor, tuple):
            in_tensor = make_random_geometric_tensor(in_type, *in_tensor)

        if out_type is None:
            out_type = module.out_type

        self.in_tensor = in_tensor
        self.in_type = in_type
        self.out_type = out_type

        if out_type:
            assert in_type.gspace is out_type.gspace

        self.gspace = in_type.gspace
        self.group = in_type.gspace.fibergroup

        self.group_elements = _pick_group_elements(self.group, group_elements)

def make_random_geometric_tensor(
        in_type: FieldType,
        minibatch_size: int = 3,
        euclidean_size: int = 10,
) -> GeometricTensor:
    x = torch.randn(
            minibatch_size,
            in_type.size,
            *([euclidean_size] * in_type.gspace.dimensionality),
    )
    return GeometricTensor(x, in_type)

def check_invariance(
        module: Module,
        *,
        in_tensor: Optional[Tensor | GeometricTensor] = None,
        in_type: Optional[FieldType] = None,
        group_elements: int | list[GroupElement] = 20,
        atol: float = ATOL_DEFAULT,
        rtol: float = RTOL_DEFAULT,
):
    cases = TestCases(
            module=module,
            in_tensor=in_tensor,
            in_type=in_type,
            out_type=False,
            group_elements=group_elements,
    )
    _check_transformations(
            _iter_invariance_checks(cases),
            'invariance',
            atol, rtol,
    )

def check_equivariance(
        module: Module,
        *,
        in_tensor: Optional[Tensor | GeometricTensor] = None,
        in_type: Optional[FieldType] = None,
        out_type: Optional[FieldType] = None,
        group_elements: int | list[GroupElement] = 20,
        atol: float = ATOL_DEFAULT,
        rtol: float = RTOL_DEFAULT,
):
    cases = TestCases(
            module=module,
            in_tensor=in_tensor,
            in_type=in_type,
            out_type=out_type,
            group_elements=group_elements,
    )
    _check_transformations(
            _iter_equivariance_checks(cases),
            'equivariance',
            atol, rtol,
    )

def _check_transformations(
        checks_iter,
        err_label: str,
        atol: float,
        rtol: float,
):
    results = []

    for g, y1, y2 in checks_iter:
        result = np.allclose(y1, y2, atol=atol, rtol=rtol)
        
        errs = np.abs(y1 - y2).reshape(-1)
        results.append((g, result, errs.mean()))

        if not result:
            imshow_3d(y1, y2)
            show()
            # raise AssertionError(
            #     f"The error found during {err_label} check with element {g!r} is too high: max = {errs.max()}, mean = {errs.mean()} var = {errs.var()}"
            # )
    
    return results

def _iter_invariance_checks(cases: TestCases):
    module, x = cases.module, cases.in_tensor
    y1 = _numpy_from_tensor(module(x))

    for g in cases.group_elements:
        y2 = _numpy_from_tensor(module(_transform_tensor(g, x, cases.in_type)))
        yield g, y1, y2

def _iter_equivariance_checks(cases: TestCases):
    module, x = cases.module, cases.in_tensor

    for g in cases.group_elements:
        y1 = _numpy_from_tensor(module(_transform_tensor(g, x, cases.in_type)))
        y2 = _numpy_from_tensor(_transform_tensor(g, module(x), cases.out_type))
        yield g, y1, y2

def _pick_group_elements(
        group: Group,
        user_spec: int | list[GroupElement],
):
    # KBK: Should use `Group.testing_elements`.
    if isinstance(user_spec, int):
        if group.continuous:
            for i in range(user_spec):
                yield group.sample()
        else:
            yield from group.elements[:user_spec]
    else:
        for el in user_spec:
            if not isinstance(el, GroupElement):
                raise ValueError(f"expected GroupElement, not {el!r}")
            yield el

def _transform_tensor(
        element: GroupElement,
        tensor: Tensor | GeometricTensor,
        field_type: FieldType,
):
    """
    Apply the given transformation to any type of tensor.
    """
    if isinstance(tensor, GeometricTensor):
        assert tensor.type == field_type
        return tensor.transform(element)
    else:
        return field_type.transform(tensor, element, coords=None)

def _numpy_from_tensor(x):
    """
    Create a numpy array from any type of tensor.
    """
    if isinstance(x, GeometricTensor):
        x = x.tensor
    return x.detach().numpy()
