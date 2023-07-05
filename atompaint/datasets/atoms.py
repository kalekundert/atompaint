"""
Utilities relating to the "atoms" data structure, which is just a data frame 
with the following columns: element, x, y, z.
"""

import numpy as np
import pandas as pd
import pandera as pa
import re
import os

from .coords import transform_coords, homogenize_coords
from more_itertools import one
from functools import cached_property
from pathlib import Path

from typing import TypeAlias, Optional
from pandera.typing import DataFrame, Series

class AtomSchema(pa.DataFrameModel):
    element: Series[str]
    x: Series[float]
    y: Series[float]
    z: Series[float]
    occupancy: Series[float]

Atoms: TypeAlias = DataFrame[AtomSchema]

def load_pisces(cullpdb_path):
    df = pd.read_fwf(cullpdb_path)
    df['tag'] = 'pisces/' + df['PDBchain']
    return df

def parse_pisces_path(path):
    """
    Attempt to extract as much metadata as possible from the name of a file 
    downloaded from the PISCES server.
    """
    i = '[0-9]+'
    f = fr'{i}\.{i}'
    pisces_pattern = fr"""
            cullpdb_
            pc(?P<max_percent_identity>{f})_
            res(?P<min_resolution_A>{f})-(?P<max_resolution_A>{f})_
            ((?P<no_breaks>noBrks)_)?
            len(?P<min_length>{i})-(?P<max_length>{i})_
            R(?P<max_r_free>{f})_
            (?P<experiments>[a-zA-Z+]+)_
            d(?P<year>\d{{4}})_(?P<month>\d{{2}})_(?P<day>\d{{2}})_
            chains(?P<num_chains>{i})
    """
    if m := re.match(pisces_pattern, path.name, re.VERBOSE):
        return m.groupdict()
    else:
        return {}


def atoms_from_tag(tag: str) -> Atoms:
    form, id = tag.split('/')

    if form == 'pisces':
        id, chain = id[:4], id[-1]
        path = get_pdb_redo_path(id)
        return atoms_from_mmcif(path, chain=chain)
    else:
        raise ValueError(f"unknown tag prefix: {tag}")

def atoms_from_mmcif(path: Path, chain: Optional[str]=None) -> Atoms:
    from pdbecif.mmcif_io import CifFileReader

    if not path.exists():
        raise FileNotFoundError(path)

    cifs = CifFileReader().read(path)
    cif = one(cifs.values())

    # Might make more sense to just pick the highest occupancy conformation to 
    # train on, but occupancy is more true to the underlying data.  We don't 
    # necessarily know which partial occupancy conformations go together.
    df = pd.DataFrame({
            'chain': cif['_atom_site']['label_asym_id'],
            'element': cif['_atom_site']['type_symbol'],
            'x': map(float, cif['_atom_site']['Cartn_x']),
            'y': map(float, cif['_atom_site']['Cartn_y']),
            'z': map(float, cif['_atom_site']['Cartn_z']),
            'occupancy': map(float, cif['_atom_site']['occupancy']),
    })

    if chain is not None:
        df = df[df['chain'] == chain]

    del df['chain']

    return df

def atoms_from_pymol(sele: str, state=-1) -> Atoms:
    from pymol import cmd

    rows = []
    cmd.iterate_state(
            state, sele,
            'rows.append((elem, x, y, z, q))',
            space={'rows': rows},
    )

    return pd.DataFrame(rows, columns=['element', 'x', 'y', 'z', 'occupancy'])

def get_pdb_redo_path(id: str) -> Path:
    id = id.lower()
    root = Path(os.environ['PDB_DIR'])
    return root / id[1:3] / f'{id}_final.cif'


def get_atom_coord(atoms, i):
    # Important to select columns before `loc`: This ensures that the resulting 
    # array is of dtype float rather than object, because all of the selected 
    # rows are float.
    return atoms[['x', 'y', 'z']].loc[i].values

def get_atom_coords(atoms, *, homogeneous=False):
    coords = atoms[['x', 'y', 'z']].values
    return homogenize_coords(coords) if homogeneous else coords

def set_atom_coords(atoms, coords):
    # Only use the first three columns, in case we were given homogeneous 
    # coordinates.
    atoms[['x', 'y', 'z']] = coords[:, 0:3]

def transform_atom_coords(atoms_x, frame_xy, inplace=False):
    coords_x = get_atom_coords(atoms_x, homogeneous=True)
    coords_y = transform_coords(coords_x, frame_xy)

    atoms_y = atoms_x if inplace else atoms_x.copy(deep=True)
    set_atom_coords(atoms_y, coords_y)

    return atoms_y

