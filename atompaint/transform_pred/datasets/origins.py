"""\
Each training example is composed of two nearby "views" of the same structure.  
This module is concerned with determining whether or not a view centered at a 
particular origin should be included in the dataset.

Currently, some of the basic ideas are:

- A view must have a certain number of "biological" atoms nearby.  A biological 
  atom is one that is probably not a crystallization artifact, or solvent.

  Part of the purpose of looking at nearby atoms is to discourage training on 
  surface regions, since we don't want to train on examples that genuinely 
  don't have enough information to make a prediction.  We also don't want the 
  network to get in the habit of guessing based on the orientation of the 
  surface.

- Instead of considering every possible point as a potential origin, it may be 
  helpful to consider a discrete set of such points, e.g. all the atoms in a 
  structure.

In the future, I'd also like to threshold on more factors, e.g.:

- The specific atoms in each view, to avoid redundancy.
- The quality of the structure.
"""

import numpy as np
import pandas as pd
import pandera as pa
import json

from atompaint.datasets.coords import Coord, Coords3
from atompaint.datasets.atoms import (
        get_atom_coords, atoms_from_tag, filter_nonbiological_atoms,
)
from scipy.spatial import KDTree
from pandera.typing import DataFrame, Series
from numpy.typing import NDArray
from dataclasses import dataclass, asdict
from pathlib import Path
from shutil import rmtree
from typing import TypeAlias

@dataclass
class OriginParams:
    radius_A: float
    min_nearby_atoms: int

@dataclass
class OriginFilteringAtoms:
    kd_tree: NDArray
    occupancies: NDArray

class OriginsSchema(pa.DataFrameModel):
    tag: Series[str]
    x: Series[float]
    y: Series[float]
    z: Series[float]

Origins: TypeAlias = DataFrame[OriginsSchema]

def choose_origins_for_tags(tags, origin_params):
    dfs = []
    status = {
            'tags_skipped': [],
            'tags_loaded': [],
    }

    for tag in tags:
        try:
            atoms = atoms_from_tag(tag)
        except FileNotFoundError:
            status['tags_skipped'].append(tag)
            continue

        df = choose_origins_for_atoms(tag, atoms, origin_params)
        dfs.append(df)

        status['tags_loaded'].append(tag)

    df = pd.concat(dfs, ignore_index=True)
    df['tag'] = df['tag'].astype('category')
    return df, status

def choose_origins_for_atoms(tag, atoms, origin_params):
    coords = filter_origin_coords(
            get_atom_coords(atoms),
            origin_params,
            select_origin_filtering_atoms(atoms),
    )
    origins = pd.DataFrame(coords, columns=['x', 'y', 'z'])
    origins['tag'] = tag
    return origins

def select_origin_filtering_atoms(atoms):
    """
    Determine which atoms should be considered when filtering origins.

    The returned data structure contains a lot of the same information as the 
    usual ``atoms`` data frame, but the atom coordinates are stored in a KD 
    tree for faster neighbor lookups.
    """
    atoms = filter_nonbiological_atoms(atoms)
    xyz = get_atom_coords(atoms)
    return OriginFilteringAtoms(
            kd_tree=KDTree(xyz),
            occupancies=atoms['occupancy'].values,
    )

def filter_origin_coords(coords_A, origin_params, filtering_atoms):
    n = _count_nearby_atoms(coords_A, filtering_atoms, origin_params.radius_A)
    return coords_A[n >= origin_params.min_nearby_atoms]

def _count_nearby_atoms(coords_A, filtering_atoms, radius_A):
    """
    Calculate the number of atoms within the given radius of the given coordinates.

    The counts will account for occupancy.
    """
    kd_tree = filtering_atoms.kd_tree
    occupancies = filtering_atoms.occupancies

    # It's ok to call this function with just a single coordinate, but the rest 
    # of the code assumes that the input coordinate array is 2D (e.g. this 
    # affect the shape of the `query_ball_point()` return value), so here we 
    # have to add the second dimension (if necessary).
    coords_A.shape = (1, *coords_A.shape)[-2:]

    # If this is a bottleneck, `query_ball_tree()` might be faster.
    hits = kd_tree.query_ball_point(coords_A, radius_A)

    counts = np.zeros(len(coords_A))
    for i, neighbors in enumerate(hits):
        for j in neighbors:
            counts[i] += occupancies[j]
    return counts

def get_origin_coord(origins: Origins, i: int) -> Coord:
    # Important to select columns before `iloc`: This ensures that the
    # resulting array is of dtype float rather than object, because all of the
    # selected rows are float.
    return origins[['x', 'y', 'z']].iloc[i].to_numpy()

def get_origin_coords(origins: Origins) -> Coords3:
    return origins[['x', 'y', 'z']].to_numpy()

def get_origin_tag(origins: Origins, i: int) -> str:
    return origins.iloc[i]['tag']

class ParquetOriginSampler:
    # Store metadata in a pandas data frame loaded into memory.

    def __init__(self, origins_path):
        self.origins = load_origins(origins_path)
        _, self.params = load_origin_params(origins_path)

        # Without this grouping, getting all the origins from a certain tag
        # would require iterating over every origin, and would be a performance
        # bottleneck.  Note that it's slightly faster to create a dictionary
        # here, instead of relying on the `groupby` object.  But this approach
        # uses half as much memory (see expt #224), and I think that's more
        # likely to matter than the speed difference.
        self.origins_by_tag = self.origins.groupby('tag')

    def sample(self, rng):
        from .utils import sample_origin
        origin_a, tag = sample_origin(rng, self.origins)
        origins_b = self.origins_by_tag.get_group(tag)
        return tag, origin_a, origins_b, atoms_from_tag(tag)

    def teardown(self):
        pass

class SqliteOriginSampler:
    """
    Load metadata from an SQLite database

    This is both faster than, and 
    """

    select_origin_a = 'SELECT tag_id, x, y, z FROM origins WHERE rowid=?'
    select_tag = 'SELECT tag FROM tags WHERE id=?'
    select_origins_b = 'SELECT x, y, z FROM origins WHERE tag_id=?'

    def __init__(self, origins_path, copy_to_tmp=True):
        import sqlite3, tempfile, shutil

        origins_path = Path(origins_path)
        db_path = origins_path / 'origins.db'

        if copy_to_tmp:
            self.tmp = tempfile.NamedTemporaryFile(
                    prefix='atompaint_',
                    delete=True,
            )
            shutil.copy2(db_path, self.tmp.name)
            db_path = self.tmp.name

        self.db = sqlite3.connect(db_path)

        # Assume that no rows are ever deleted from the database.
        count_origins = 'SELECT MAX(rowid) FROM origins'
        self.num_origins = self.db.execute(count_origins).fetchone()[0]

        _, self.params = load_origin_params(origins_path)

    def sample(self, rng):
        cur = self.db.cursor()
        i = rng.integers(self.num_origins, dtype=int) + 1

        tag_id, *origin_a = cur.execute(self.select_origin_a, (i,)).fetchone()
        tag, = cur.execute(self.select_tag, (tag_id,)).fetchone()
        origins_b = cur.execute(self.select_origins_b, (tag_id,)).fetchall()

        origin_a = np.array(origin_a)
        origins_b = pd.DataFrame(origins_b, columns=['x', 'y', 'z'])

        # Necessary because I reuse the `sample_origin()` function on 
        # `origins_b`.  Really, `origins_b` should just be the coordinates and 
        # not the tag, since we've already picked the tag at this point.  It 
        # should probably also be a numpy array.
        origins_b['tag'] = tag
        atoms = atoms_from_tag(tag)

        return tag, origin_a, origins_b, atoms

    def teardown(self):
        try:
            self.tmp.close()
        except AttributeError:
            pass

def load_origins(path: Path):
    dfs = [
            pd.read_parquet(p).drop(columns='weight', errors='ignore')
            for p in sorted(path.glob('origins.parquet*'))
    ]
    df = pd.concat(dfs, ignore_index=True)
    df['tag'] = df['tag'].astype('category')
    return df

def load_origin_params(path: Path):
    with open(path / 'params.json') as f:
        params = json.load(f)

    tags = params.pop('tags')
    origin_params = OriginParams(**params)

    return tags, origin_params

def save_origins(path: Path, df, status, suffix=None):
    if suffix:
        worker_id, num_workers = suffix
        suffix = f'.{worker_id:0{len(str(num_workers - 1))}}'
    else:
        suffix = ''

    df.to_parquet(path / f'origins.parquet{suffix}')
    
    with open(path / f'status.json{suffix}', 'w') as f:
        json.dump(status, f)

def save_origin_params(path: Path, tags, origin_params, force=False):
    if path.exists():
        if force or not any(path.glob('origins.parquet*')):
            rmtree(path)
        else:
            raise FileExistsError(path)

    path.mkdir()

    params = {
            'tags': list(tags),
            **asdict(origin_params),
    }
    with open(path / 'params.json', 'w') as f:
        json.dump(params, f)

def consolidate_origins(path: Path, dry_run: bool=False):
    df = load_origins(path)
    status = {'tags_skipped': [], 'tags_loaded': []}

    for p in path.glob('status.json*'):
        with open(p) as f:
            status_i = json.load(f)

        status['tags_loaded'] += status_i['tags_loaded']
        status['tags_skipped'] += status_i['tags_skipped']

    if not dry_run:
        save_origins(path, df, status)

        for p in path.glob('origins.parquet.*'):
            p.unlink()
        for p in path.glob('status.json.*'):
            p.unlink()

    return df, status


