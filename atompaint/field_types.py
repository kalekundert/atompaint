import numpy as np

from atompaint.utils import get_scalar
from escnn.nn import EquivariantModule, FieldType, FourierFieldType
from itertools import repeat
from more_itertools import take, flatten, zip_broadcast

from escnn.nn import GeometricTensor
from escnn.gspaces import GSpace
from escnn.group import Representation
from collections.abc import Iterable, Sequence

class CastToFourierFieldType(EquivariantModule):
    """
    Some modules output field types that are compatible with the Fourier 
    transform modules, but that aren't `FourierFieldType` instances.  This 
    module replaces the output field with a `FourierFieldType`, after doing 
    some checks.
    """

    def __init__(self, in_type, out_type):
        super().__init__()

        self.in_type = in_type
        self.out_type = out_type

        # The user is responsible for making sure that the input type of `x` is 
        # genuinely that same as the given output type.  Here we just do a few 
        # sanity checks to catch mistakes.
        assert in_type.size == self.out_type.size
        assert in_type.irreps == self.out_type.irreps
        assert np.allclose(
            in_type.change_of_basis.toarray(), 
            np.eye(in_type.size),
        )

    def forward(self, x: GeometricTensor):
        assert x.type == self.in_type
        x.type = self.out_type
        return x

    def evaluate_output_shape(self, input_shape):
        return input_shape

def make_top_level_field_types(gspace, channels, make_nontrivial_field_types):
    yield from make_trivial_field_type(gspace, channels[0])
    yield from make_nontrivial_field_types(gspace, channels[1:])

def make_trivial_field_type(gspace, channels):
    yield FieldType(gspace, channels * [gspace.trivial_repr])

def make_trivial_field_types(gspace, channels):
    for channels_i in channels:
        yield FieldType(gspace, channels_i * [gspace.trivial_repr])

def make_fourier_field_types(gspace, channels, max_frequencies, **kwargs):
    group = gspace.fibergroup

    for i, channels_i in enumerate(channels):
        max_freq = get_scalar(max_frequencies, i)
        bl_irreps = group.bl_irreps(max_freq)
        yield FourierFieldType(gspace, channels_i, bl_irreps, **kwargs)

def make_fourier_field_type(gspace, channels, max_frequency, **kwargs):
    bl_irreps = gspace.fibergroup.bl_irreps(max_frequency)
    yield FourierFieldType(gspace, channels, bl_irreps, **kwargs)

def make_polynomial_field_types(gspace, channels, terms):
    for i, channels_i in enumerate(channels):
        terms_i = get_scalar(terms, i)
        assert terms_i > 0

        rho = take(terms_i, iter_polynomial_representations(gspace.fibergroup))
        yield FieldType(gspace, channels_i * list(rho))

def make_exact_polynomial_field_types(
        gspace: GSpace,
        channels: Iterable[int],
        terms: int | Iterable[int],
        gated: bool = False,
        strict: bool = True,
):
    for channels_i, terms_i in zip_broadcast(channels, terms):
        polynomial_i = list(take(
            terms_i,
            iter_polynomial_representations(gspace.fibergroup),
        ))
        yield make_exact_width_field_type(
                gspace=gspace, 
                channels=channels_i,
                representations=polynomial_i,
                gated=gated,
                strict=strict,
        )

def make_exact_width_field_type(
        gspace: GSpace,
        channels: int,
        representations: Sequence[Representation],
        gated: bool = False,
        strict: bool = True,
):
    """
    Construct a field type by repeatedly combining copies of the given 
    representation until the whole field type has the requested number of 
    channels.

    - By default, it is an error if it is not possible to create a field type 
      with the correct number of channels.

    - After each iteration, the last representation is removed from 
      consideration.  Iteration stops when no representations remain.  It's a 
      good idea to sort the representations from smallest (at the front) to 
      largest (at the back).  This makes it easier to evenly fill the requested 
      number of channels.
    """
    rho_multiplicities = np.zeros(len(representations), dtype=int)
    gate_multiplicity = 0
    channels_remaining = channels
    sizes = [(x.size + gated) for x in representations]

    for j in range(len(representations), 0, -1):
        size = sum(sizes[:j])
        n = channels_remaining // size
        rho_multiplicities[:j] += n
        gate_multiplicity += gated and n * j
        channels_remaining -= n * size

    if channels_remaining and strict:
        raise ValueError(f"can't exactly fill {channels} channels with representations of the following sizes: {sizes}")

    # Take care to keep each term contiguous with all the others of the 
    # same type/dimension.  This allows the gated nonlinearity to run 
    # faster, because it can use slices rather than indices.

    # When using gates, take care to put the gates before the 
    # representations being gated.  This is what the `GatedNonLinearity1` 
    # class expects, so respecting this order here makes things just work.

    gates = repeat(
            gspace.fibergroup.trivial_representation,
            gate_multiplicity,
    )
    rho = flatten(
            n * [term]
            for n, term in zip(rho_multiplicities, representations)
    )
    return FieldType(gspace, [*gates, *rho])

def iter_polynomial_representations(group):
    # Avoid generating the tensor product of the zeroth and first irrep.  The 
    # resulting representation doesn't have any "supported nonlinearities", see 
    # the if-statement at the end of `Group._tensor_product()`.  This is 
    # probably a bug in escnn, but the work-around is easy.

    rho_0 = group.irrep(0)
    rho_1 = rho_next = group.irrep(1)

    yield rho_0
    yield rho_1

    while True:
        rho_next = rho_next.tensor(rho_1)
        yield rho_next

def add_gates(in_type):
    gspace = in_type.gspace
    group = in_type.fibergroup
    rho = in_type.representations
    gates = len(rho) * [group.trivial_representation]
    return FieldType(gspace, [*gates, *rho])


