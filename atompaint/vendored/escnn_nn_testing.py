import torch
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc

from torch import Tensor
from torch.nn import Module
from escnn.nn import FieldType, GeometricTensor
from escnn.group import Group, GroupElement, octa_group
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from math import prod
from itertools import product
from typing import Optional

# These are probably too tight.
ATOL_DEFAULT = 1e-7
RTOL_DEFAULT = 1e-5

class TestCases:

    def __init__(
            self, *,
            module: Module,
            in_tensor: Tensor | GeometricTensor | tuple[int, ...],
            in_type: Optional[FieldType] = None,
            out_type: Optional[FieldType] = None,
            out_shape: Optional[tuple[int, ...]] = None,
            group_elements: int | list[GroupElement] = 20,
            random_seed: int = 0,
    ):
        self.module = module

        if in_type is None:
            in_type = module.in_type

        # We want to be able to test both modules that take any kind of input 
        # (e.g. geometric tensors, normal tensors, custom objects, etc.), so we 
        # try to avoid interpreting the *in_tensor* argument too much.  The one 
        # exception we make is the treat a tuple as the shape of a geometric 
        # tensor to construct, just because this is probably the most common 
        # scenario and it saves the user a good amount of boilerplate.
        #
        # Note that when the user specifies a shape, we require that they 
        # specify all dimensions, even though we already know what the channel 
        # dimension should be.  This is because it's good for tests to be 
        # explicit.  We want to catch the case where the input type isn't of 
        # the size that the user expects it to be.

        if isinstance(in_tensor, tuple):
            in_tensor = GeometricTensor(
                    torch.randn(*in_tensor),
                    in_type,
            )

        if out_type is None:
            out_type = module.out_type

        self.in_tensor = in_tensor
        self.in_type = in_type
        self.out_type = out_type
        self.out_shape = out_shape

        self.group = in_type.gspace.fibergroup
        self.group_elements = _pick_group_elements(self.group, group_elements)

        self.random_seed = random_seed

class TestResult:

    def __init__(self, g, out_eq, out_shape, in_plots, out_plots):
        self.g = g
        self.out_eq = tuple(
                _numpy_from_tensor(x)
                for x in out_eq
        )
        self.out_shape = out_shape
        self.in_plots = [
                (_numpy_from_tensor(x), label)
                for x, label in in_plots
        ]
        self.out_plots = [
                (_numpy_from_tensor(x), label)
                for x, label in out_plots
        ]

def check_invariance(
        module: Module,
        *,
        in_tensor: Tensor | GeometricTensor | tuple[int, ...],
        in_type: Optional[FieldType] = None,
        out_shape: tuple[int, ...] = None,
        group_elements: int | list[GroupElement] = 20,
        atol: float = ATOL_DEFAULT,
        rtol: float = RTOL_DEFAULT,
        plot: bool = False,
        plot_inputs: bool = True,
        **plot_kwargs,
):
    cases = TestCases(
            module=module,
            in_tensor=in_tensor,
            in_type=in_type,
            out_type=False,
            out_shape=out_shape,
            group_elements=group_elements,
    )
    _check_transformations(
            checks_iter=_iter_invariance_checks(cases),
            err_label='invariance',
            atol=atol,
            rtol=rtol,
            plot=plot,
            plot_inputs=plot_inputs,
            **plot_kwargs,
    )

def check_equivariance(
        module: Module,
        *,
        in_tensor: Tensor | GeometricTensor | tuple[int, ...],
        in_type: Optional[FieldType] = None,
        out_type: Optional[FieldType] = None,
        out_shape: tuple[int, ...] = None,
        group_elements: int | list[GroupElement] = 20,
        atol: float = ATOL_DEFAULT,
        rtol: float = RTOL_DEFAULT,
        plot: bool = False,
        plot_inputs: bool = True,
        **plot_kwargs,
):
    cases = TestCases(
            module=module,
            in_tensor=in_tensor,
            in_type=in_type,
            out_type=out_type,
            out_shape=out_shape,
            group_elements=group_elements,
    )
    _check_transformations(
            checks_iter=_iter_equivariance_checks(cases),
            err_label='equivariance',
            atol=atol,
            rtol=rtol,
            plot=plot,
            plot_inputs=plot_inputs,
            **plot_kwargs,
    )

def make_random_geometric_tensor(
        in_type: FieldType,
        minibatch_size: int = 3,
        spatial_size: int = 10,
) -> GeometricTensor:
    x = torch.randn(
            minibatch_size,
            in_type.size,
            *([spatial_size] * in_type.gspace.dimensionality),
    )
    return GeometricTensor(x, in_type)

def get_exact_3d_rotations(group: Group) -> list[GroupElement]:
    """
    Return all the rotations that (i) all belong to the given group and (ii) do 
    not require interpolation.  These rotations are good for checking 
    equivariance, because interpolation can be a significant source of error.
    """
    octa = octa_group()
    exact_rots = []

    for octa_element in octa.elements:
        value = octa_element.value
        param = octa_element.param

        try:
            exact_rot = group.element(value, param)
        except ValueError:
            continue

        exact_rots.append(exact_rot)

    assert len(exact_rots) > 1
    return exact_rots

def imshow_nd(xs, *, fig=None, row_labels=[], norm_groups=[], max_batches=0, max_channels=0, cmap=cc.cm.coolwarm):
    """
    Plot tensors with batch, channel, and any number of spatial dimensions.
    """

    xs = [_require_width_height_depth(_numpy_from_tensor(x)) for x in xs]

    shape_groups = [x.shape for x in xs]
    norm_groups = norm_groups or shape_groups

    abs_max = {
            i: np.max(np.abs(x))
            for i, x in enumerate(xs)
    }
    xlims = {}
    for i, x in abs_max.items():
        j = norm_groups[i]
        xlims[j] = max(x, xlims.get(j, 0))

    norms = {
            k: Normalize(-v, v)
            for k, v in xlims.items()
    }

    def get_bcd(x):
        b, c, *d = x.shape[:-2]

        if max_batches > 0:
            b = min(b, max_batches)
        if max_channels > 0:
            c = min(c, max_channels)

        return b, c, *d

    def get_title(i, bcd):
        if i == 0:
            return f'batch={bcd[i]}'
        if i == 1:
            return f'channel={bcd[i]}'
        if i == 2 and len(bcd) == 3:
            return f'depth={bcd[i]}'

        return f'spatial[{i-2}]={bcd[i]}'

    num_cols = 1
    for x in xs:
        bcd = get_bcd(x)
        num_cols = max(num_cols, prod(bcd))

    if fig is None:
        plt.close()
        fig = plt.figure(layout='compressed')

    plot_size = 1.5
    fig.set_size_inches(num_cols * plot_size, len(xs) * plot_size)

    gs = GridSpec(
            len(xs),
            num_cols + 1,
            width_ratios=([1] * num_cols) + [1/10],
            figure=fig,
    )

    for i, x in enumerate(xs):
        bcd = get_bcd(x)
        cols = product(*map(range, bcd))

        prev_titles = []

        for j, bcd_j in enumerate(cols):
            ax = fig.add_subplot(gs[i, j])
            img = ax.imshow(
                    x[bcd_j],
                    norm=norms[norm_groups[i]],
                    cmap=cc.cm.coolwarm,
            )
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0 or shape_groups[i] != shape_groups[i-1]:
                titles = [
                        get_title(k, bcd_j)
                        for k in range(len(bcd_j))
                ]
                title = '\n'.join(
                        x
                        for x in titles
                        if x not in prev_titles
                )
                prev_titles = titles
                ax.set_title(title)

            if not any(bcd_j) and row_labels:
                ax.set_ylabel(row_labels[i])

        ax_cb = fig.add_subplot(gs[i, num_cols])
        plt.colorbar(img, cax=ax_cb)

def _check_transformations(
        checks_iter,
        err_label: str,
        atol: float,
        rtol: float,
        plot: bool = False,
        plot_inputs: bool = True,
        **plot_kwargs,
):
    results = []

    for check in checks_iter:
        y1, y2 = check.out_eq
        result = np.allclose(y1, y2, atol=atol, rtol=rtol)

        if check.out_shape is not None:
            assert y1.shape == check.out_shape
            assert y2.shape == check.out_shape
        
        errs = np.abs(y1 - y2).reshape(-1)
        results.append((check.g, result, errs.mean()))

        if plot:
            if plot_inputs:
                plots = check.in_plots + check.out_plots
            else:
                plots = check.out_plots

            xs, row_labels = zip(*plots)
            imshow_nd(
                    xs=xs,
                    row_labels=row_labels,
                    **plot_kwargs,
            )
            plt.show()

        if not result:
            raise AssertionError(
                f"The error found during {err_label} check with element {check.g!r} is too high: max = {errs.max()}, mean = {errs.mean()} var = {errs.var()}"
            )
    
    return results

def _iter_invariance_checks(cases: TestCases):
    f, x = cases.module, cases.in_tensor
    f_x = _forward(f, x, cases.random_seed)

    for g in cases.group_elements:
        gx = _transform_tensor(g, x, cases.in_type)
        f_gx = _forward(f, gx, cases.random_seed)

        yield TestResult(
                g=g,
                out_eq=(f_x, f_gx),
                out_shape=cases.out_shape,
                in_plots=[
                    (x,    r'$x$'),
                    (gx,   r'$g \cdot x$'),
                ],
                out_plots=[
                    (f_x,  r'$f(x)$'),
                    (f_gx, r'$f(g \cdot x)$'),
                ],
        )

def _iter_equivariance_checks(cases: TestCases):
    f, x = cases.module, cases.in_tensor

    for g in cases.group_elements:
        gx = _transform_tensor(g, x, cases.in_type)
        f_x = _forward(f, x, cases.random_seed)
        f_gx = _forward(f, gx, cases.random_seed)
        gf_x = _transform_tensor(g, f_x, cases.out_type)
        yield TestResult(
                g=g,
                out_eq=(f_gx, gf_x),
                out_shape=cases.out_shape,
                in_plots=[
                    (x,    r'$x$'),
                    (gx,   r'$g \cdot x$'),
                ],
                out_plots=[
                    (f_x,  r'$f(x)$'),
                    (f_gx, r'$f(g \cdot x)$'),
                    (gf_x, r'$g \cdot f(x)$'),
                ],
        )

def _pick_group_elements(
        group: Group,
        user_spec: int | list[GroupElement],
):
    # KBK: I could use `Group.testing_elements`.  But I don't like the whole 
    # idea of this attribute.  The class itself shouldn't be responsible for 
    # its own testing.

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

def _forward(f, x, seed):
    # Reset the random seed before each forward pass, so that we don't get 
    # different results from dropout layers and things like that.
    torch.manual_seed(seed)
    return f(x)

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
    if isinstance(x, np.ndarray):
        return x

    x = getattr(x, 'tensor', x)
    return x.detach().numpy()

def _require_width_height_depth(x):
    n = len(x.shape)

    if n < 2:
        raise ValueError(f"tensors must have batch and channel dimensions: {x.shape}")

    if n < 5:
        y = x.reshape(x.shape + (1,) * (5 - n))
        return y

    return x
