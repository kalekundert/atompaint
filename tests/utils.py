import parametrize_from_file as pff

from pytest import approx
from escnn.group import Octahedral
from functools import cache

with_py = pff.Namespace()
with_math = pff.Namespace('from math import *')
with_ap = pff.Namespace('import atompaint as ap')

def get_exact_rotations(group):
    """
    Return all the rotations that (i) all belong to the given group and (ii) do 
    not require interpolation.  These rotations are good for checking 
    equivariance, because interpolation can be a significant source of error.
    """
    octa = get_octahedral_group()
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
        
@cache
def get_octahedral_group():
    """
    Return the octahedral group.

    This function is cached because it takes a few seconds to instantiate an 
    octahedral group object each time.  The object returned by this function 
    should never be modified (since the same object is reused on each call).
    """
    return Octahedral()





