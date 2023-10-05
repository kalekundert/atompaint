import atompaint.datasets.atoms as apda
import parametrize_from_file as pff
import pandas as pd
import re

from test_datasets_coords import frame
from pathlib import Path
from io import StringIO

def atoms(params):
    dtypes = {
            'i': int,
            'monomer': str,
            'element': str,
            'x': float,
            'y': float,
            'z': float,
            'occupancy': float,
    }
    cols = list(dtypes.keys())[1:]
    col_aliases = {
            'e': 'element',
            'q': 'occupancy',
            'resn': 'monomer',
    }
    dtypes |= {k: dtypes[v] for k, v in col_aliases.items()}

    io = StringIO(params)
    head = io.readline()

    # If there isn't a header row, assume that only the element and coordinate 
    # columns were given.  The rest of the columns are given default values.
    #
    # If there is a header row, create whichever columns it specifies.  Note 
    # that the 'x', 'y', and 'z' columns are always required.

    if {'x', 'y', 'z'} <= set(head.split()):
        fwf_kwargs = {}
    else:
        fwf_kwargs = dict(
                names=['e', 'x', 'y', 'z'],
        )

    io.seek(0)
    df = pd.read_fwf(io, sep=' ', dtype=dtypes, **fwf_kwargs)

    df.rename(
            columns=col_aliases,
            errors='ignore',
            inplace=True,
    )

    if 'i' in df:
        df.set_index('i', drop=True, inplace=True)
        df.rename_axis(None, inplace=True)

    if 'monomer' not in df:
        df['monomer'] = 'ALA'
    if 'element' not in df:
        df['element'] = 'C'
    if 'occupancy' not in df:
        df['occupancy'] = 1.0

    return df[cols]


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

@pff.parametrize(schema=pff.cast(atoms=atoms, expected=atoms))
def test_filter_nonbiological_atoms(atoms, expected):
    actual = apda.filter_nonbiological_atoms(atoms)
    pd.testing.assert_frame_equal(actual, expected)

@pff.parametrize(
        indirect=['tmp_files'],
        schema=[
            pff.cast(expected=atoms),
            pff.defaults(chain=None),
        ],
)
def test_atoms_from_mmcif(tmp_files, chain, expected):
    atoms = apda.atoms_from_mmcif(tmp_files / 'atoms.cif', chain=chain)
    pd.testing.assert_frame_equal(atoms, expected)

@pff.parametrize(
        schema=[
            pff.cast(atoms=atoms),
            pff.defaults(name=''),
        ],
)
def test_mmcif_from_atoms(name, atoms, expected, tmp_path):
    cif_path = tmp_path / 'out.cif'
    apda.mmcif_from_atoms(cif_path, atoms, name=name)

    # The PDBeCIF library adds a bunch of trailing whitespace for some reason.  
    # I don't want to actually require that whitespace to be present (it's not 
    # significant), so instead of adding it to the test case, I just strip it 
    # out before making the comparison.
    cif_text = re.sub(r'\s*$', '', cif_path.read_text(), flags=re.MULTILINE)
    assert cif_text == expected

@pff.parametrize(
        schema=pff.cast(atoms_x=atoms, frame_xy=frame, expected_y=atoms),
)
def test_transform_atom_coords(atoms_x, frame_xy, expected_y):
    atoms_y = apda.transform_atom_coords(atoms_x, frame_xy)
    pd.testing.assert_frame_equal(atoms_y, expected_y)

