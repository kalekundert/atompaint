"""
Utilities relating to the "atoms" data structure, which is just a data frame 
with the following columns: element, x, y, z.
"""

import numpy as np
import pandas as pd
import pandera as pa
import importlib.resources
import re
import os

from .coords import (
        Coord, Coords3, Coords4, Frame, transform_coords, homogenize_coords,
)
from more_itertools import one
from functools import cache, cached_property
from pathlib import Path

from typing import TypeAlias, Optional
from pandera.typing import DataFrame, Series

class AtomSchema(pa.DataFrameModel):
    monomer: Series[str]
    element: Series[str]
    x: Series[float]
    y: Series[float]
    z: Series[float]
    occupancy: Series[float]

Atoms: TypeAlias = DataFrame[AtomSchema]

def load_pisces(cullpdb_path: Path):
    df = pd.read_fwf(cullpdb_path)
    df['tag'] = 'pisces/' + df['PDBchain']
    return df

def parse_pisces_path(path: Path):
    """
    Attempt to extract as much metadata as possible from the name of a file 
    downloaded from the PISCES server.
    """
    i = '[0-9]+'
    f = fr'{i}\.{i}'
    pisces_pattern = fr'''
            cullpdb_
            pc(?P<max_percent_identity>{f})_
            res(?P<min_resolution_A>{f})-(?P<max_resolution_A>{f})_
            ((?P<no_breaks>noBrks)_)?
            len(?P<min_length>{i})-(?P<max_length>{i})_
            R(?P<max_r_free>{f})_
            (?P<experiments>[a-zA-Z+]+)_
            d(?P<year>\d{{4}})_(?P<month>\d{{2}})_(?P<day>\d{{2}})_
            chains(?P<num_chains>{i})
    '''
    if m := re.match(pisces_pattern, path.name, re.VERBOSE):
        return m.groupdict()
    else:
        return {}


@cache
def load_nonbiological_ligands():
    vendor_dir = importlib.resources.files('atompaint.vendored')
    tsv_path = vendor_dir / 'PDB_ligand_quality_composite_score' / 'non-LOI-blocklist.tsv'

    return pd.read_csv(
            tsv_path,
            sep='\t',
            # Can't find column definitions, so I'm just guessing for some of 
            # these (some are self-evident).
            names=['id', 'n', 'het', 'mw', 'formula', 'role', 'notes'],
    )

def filter_nonbiological_atoms(atoms: Atoms) -> Atoms:
    # I'm just checking that the 3-letter ID string matches.  Ideally, I'd also 
    # check that the molecular formula matches, because I'm pretty sure that 
    # these 3-letter IDs aren't guaranteed to be unique between structures.  
    # This information is in the CIF files, but providing this information 
    # would require `atoms` to become a more complicated, opaque object (as 
    # opposed to just a data frame with some standard columns).
    blacklist = load_nonbiological_ligands()
    hits = atoms['monomer'].isin(blacklist['id'])
    return atoms[~hits]


def atoms_from_tag(tag: str) -> Atoms:
    form, id = tag.split('/')

    if form == 'pisces':
        # A number of the PISCES entries have three-letter chains, e.g. AAA or 
        # BBB.  When I look at the actual structures, they just have the normal 
        # chain A or B.  So for that reason, I made the decision to only take 
        # the first letter of the chain and ignore anything later.  I can 
        # imagine that this would be wrong in some cases, but I think it'll be 
        # right more often that not.
        id, chain = id[:4], id[4]
        path = _get_pdb_redo_path(id, '.feather')
        return atoms_from_feather(path, chain=chain)
    else:
        raise ValueError(f"unknown tag prefix: {tag}")

def atoms_from_mmcif(path: Path, chain: Optional[str | bool]=None) -> Atoms:
    from pdbecif.mmcif_io import CifFileReader

    if not path.exists():
        raise FileNotFoundError(path)

    cifs = CifFileReader().read(path)
    cif = one(cifs.values())

    # Might make more sense to just pick the highest occupancy conformation to 
    # train on, but including each atom in proportion to its occupancy is more 
    # true to the underlying data.  We don't necessarily know which partial 
    # occupancy conformations go together.
    df = pd.DataFrame({
        'chain': cif['_atom_site']['label_asym_id'],
        'monomer': cif['_atom_site']['label_comp_id'],
        'element': cif['_atom_site']['type_symbol'],
        'x': map(float, cif['_atom_site']['Cartn_x']),
        'y': map(float, cif['_atom_site']['Cartn_y']),
        'z': map(float, cif['_atom_site']['Cartn_z']),
        'occupancy': map(float, cif['_atom_site']['occupancy']),
    })

    if isinstance(chain, str):
        df = df[df['chain'] == chain]
        df = df.reset_index(drop=True)

    if chain is not True:
        del df['chain']

    return df

def atoms_from_feather(path: Path, chain: Optional[str]) -> Atoms:
    df = pd.read_feather(path)

    if isinstance(chain, str):
        df = df[df['chain'] == chain]
        df = df.reset_index(drop=True)

    if chain is not True:
        del df['chain']

    return df

def atoms_from_pymol(sele: str, state=-1) -> Atoms:
    from pymol import cmd

    rows = []
    cmd.iterate_state(
            state, sele,
            'rows.append((resn, elem, x, y, z, q))',
            space={'rows': rows},
    )

    return pd.DataFrame(rows, columns=['monomer', 'element', 'x', 'y', 'z', 'occupancy'])

def _get_pdb_redo_path(id: str, suffix='.cif') -> Path:
    id = id.lower()
    root = Path(os.environ['PDB_DIR'])
    return root / id[1:3] / f'{id}_final{suffix}'


def get_atom_coord(atoms: Atoms, i: int) -> Coord:
    # Important to select columns before `loc`: This ensures that the resulting 
    # array is of dtype float rather than object, because all of the selected 
    # rows are float.
    return atoms[['x', 'y', 'z']].loc[i].values

def get_atom_coords(
        atoms: Atoms,
        *,
        homogeneous: bool=False,
) -> Coords3 | Coords4:
    coords = atoms[['x', 'y', 'z']].values
    return homogenize_coords(coords) if homogeneous else coords

def set_atom_coords(atoms: Atoms, coords: Coords3 | Coords4):
    # Only use the first three columns, in case we were given homogeneous 
    # coordinates.
    atoms[['x', 'y', 'z']] = coords[:, 0:3]

def transform_atom_coords(
        atoms_x: Atoms,
        frame_xy: Frame,
        inplace: bool=False,
) -> Atoms:
    coords_x = get_atom_coords(atoms_x, homogeneous=True)
    coords_y = transform_coords(coords_x, frame_xy)

    atoms_y = atoms_x if inplace else atoms_x.copy(deep=True)
    set_atom_coords(atoms_y, coords_y)

    return atoms_y

