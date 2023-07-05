"""\
Precalculate a set of origins to use for training transformation prediction
models.

Usage:
    ap_choose_origins <pisces_path> [-o <path>] [-f] [-n <neighbors>]
        [-r <radius>]

Arguments:
    <pisces_path>
        A path to a file specifying a subset of the PDB, as generated by the 
        PISCES web server.

Options:
    -o --output-path <path>     [default: origins_max_identity_{max_percent_identity}_max_resolution_{max_resolution_A}_min_neighbors_{min_neighbors}_radius_{radius_A}]

    -f --force
        If the output path already exists, overwrite it.  Otherwise, the 
        program will abort.
        
    -n --min-neighbors <int>    [default: 10]
        How many atoms must be in the vicinity of an atom (defined by the 
        radius below) in order for that atom to be considered as an origin.  
        The default is calculated by counting the number of atoms necessary to 
        fill half the space in question, assuming that each atom has a radius 
        of 0.7 Å.  Note that the most dense sphere packing fills 3/4 of the 
        available space, and that the empirical radii of C, N, and O are 
        between 0.6 and 0.7Å.  For the default radius of 3Å, this works out to 
        40 atoms.

        https://en.wikipedia.org/wiki/Sphere_packing
        https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements

    -r --radius <angstroms>     [default: 5]
        The area in which to count neighboring atoms, in units of Angstroms.  
        See above for more detail.

Environment variables:
    PDB_DIR
        The path to the location of the actual PDB files...
"""

import docopt
from .neighbor_count import OriginParams, choose_origins, save_origins
from ..atoms import load_pisces, parse_pisces_path
from pathlib import Path
from functools import partial
from shutil import rmtree
from tqdm import tqdm

def main():
    args = docopt.docopt(__doc__)

    pisces_path = Path(args['<pisces_path>'])
    origin_params = OriginParams(
            radius_A=float(args['--radius']),
            min_neighbors=int(args['--min-neighbors']),
    )
    all_params = {
            'pisces_path': str(pisces_path),
            **parse_pisces_path(pisces_path),
            'radius_A': origin_params.radius_A,
            'min_neighbors': origin_params.min_neighbors,
    }
    origins_path = Path(args['--output-path'].format(**all_params))

    if origins_path.exists():
        if args['--force']:
            rmtree(origins_path)
        else:
            print(f"Output path already exists: {origins_path}")
            print("Aborting!  Use `-f` to overwrite.")
            raise SystemExit

    pisces_df = load_pisces(pisces_path)
    origins = choose_origins(
            pisces_df['tag'][:100],
            origin_params,
            progress_bar=partial(tqdm, total=len(pisces_df)),
            meta=(meta := {}),
    )

    print(f"Loaded {len(meta['tags_loaded'])} structures.")
    print(f"Skipped {len(meta['tags_skipped'])} structures.")

    save_origins(origins_path, origins, all_params)

