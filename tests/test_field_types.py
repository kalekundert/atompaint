import pytest
import parametrize_from_file as pff

from atompaint.field_types import (
        make_exact_polynomial_field_types, make_exact_width_field_type,
)
from escnn.gspaces import rot3dOnR3

def test_exact_polynomial_field_types():
    # Most of the functionality is tested below, so here we just do a 
    # spot-check on a reasonable use-case.

    gspace = rot3dOnR3()
    field_types = list(make_exact_polynomial_field_types(
        gspace,
        channels=[16, 32],
        terms=3,
        gated=True,
    ))

    def get_sizes(field_type):
        return [x.size for x in field_type.representations]

    assert get_sizes(field_types[0]) == 4 * [1] + 1 * [3] + 1 * [9]
    assert get_sizes(field_types[1]) == 8 * [1] + 2 * [3] + 2 * [9]
    
@pff.parametrize(
        schema=[
            pff.defaults(gated='False', strict_err='none'),
            pff.cast(channels=int, gated=eval, strict_err=pff.error),
        ],
)
@pytest.mark.parametrize('strict', [True, False])
def test_make_exact_width_field_type(channels, rho, gated, strict, strict_err, expected):
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    with_psi = pff.Namespace(
            gspace=gspace,
            so3=so3,
            psi=so3.irrep,
    )

    rho = with_psi.eval(rho)
    expected = with_psi.eval(expected)

    if not strict:
        strict_err = pff.error('none')

    with strict_err:
        field_type = make_exact_width_field_type(
                gspace,
                channels=channels,
                representations=rho,
                gated=gated,
                strict=strict,
        )
        assert_representations_match(field_type, expected)


def assert_representations_match(field_type, expected_representations):
    from pprint import pformat

    __traceback_hide__ = True  # noqa: F841

    error = AssertionError(f"""\
field types are not identical:
actual:
{pformat(field_type.representations)}
expected:
{pformat(expected_representations)}
""")

    if len(field_type.representations) != len(expected_representations):
        raise error

    for a, b in zip(field_type.representations, expected_representations):
        if a is not b:
            raise error
