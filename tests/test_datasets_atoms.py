import atompaint.datasets.atoms as apda
import parametrize_from_file as pff
import pandas as pd

from test_datasets_coords import frame
from pathlib import Path
from io import StringIO

def atoms(params):
    cols = {
            'element': str,
            'x': float,
            'y': float,
            'z': float,
            'occupancy': float,
    }

    # Fill in the occupancy column if it's not specified.  This is just for 
    # convenience, since occupancy is not relevant to most tests, but it's 
    # still a required column of an atoms data frame.

    io = StringIO(params)
    head = io.readline()
    num_cols = len(head.split())

    io.seek(0)
    df = pd.read_fwf(io, sep=' ', names=list(cols)[:num_cols], dtype=cols)

    if 'occupancy' not in df:
        df['occupancy'] = 1.0

    return df


@pff.parametrize(indirect=['tmp_files'])
def test_load_pisces(tmp_files, tags):
    df = apda.load_pisces(tmp_files / 'pisces.txt')
    assert list(df['tag']) == tags

@pff.parametrize(schema=[pff.cast(path=Path), pff.defaults(expected=None)])
def test_parse_pisces_path(path, expected):
    params = apda.parse_pisces_path(path)

    if expected is None:
        # Just make sure the pattern matched, don't check the contents.
        assert params

    else:
        if 'no_breaks' not in expected:
            expected['no_breaks'] = None
        assert params == expected

@pff.parametrize(indirect=['tmp_files'], schema=pff.cast(expected=atoms))
def test_atoms_from_mmcif(tmp_files, expected):
    atoms = apda.atoms_from_mmcif(tmp_files / 'atoms.cif')
    pd.testing.assert_frame_equal(atoms, expected)

@pff.parametrize(
        schema=pff.cast(atoms_x=atoms, frame_xy=frame, expected_y=atoms),
)
def test_transform_atom_coords(atoms_x, frame_xy, expected_y):
    atoms_y = apda.transform_atom_coords(atoms_x, frame_xy)
    pd.testing.assert_frame_equal(atoms_y, expected_y)

