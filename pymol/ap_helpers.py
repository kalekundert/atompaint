import pymol
import numpy as np
import mixbox
import re

from pymol import cmd
from pymol.cgo import (
        ALPHA, BEGIN, COLOR, CONE, CYLINDER, END, LINES, NORMAL, TRIANGLE_FAN,
        VERTEX,
)
from pymol.wizard import Wizard
from itertools import product
from collections import Counter, defaultdict

from atompaint.datasets.atoms import atoms_from_pymol, get_pdb_redo_path
from atompaint.datasets.coords import (
        invert_coord_frame, get_origin, transform_coords,
)
from atompaint.datasets.voxelize import (
        ImageParams, Grid, image_from_atoms, get_element_channel,
        get_voxel_center_coords,
)
from atompaint.transform_pred.datasets.utils import sample_origin
from atompaint.transform_pred.datasets.recording import (
        init_recording, load_recording, load_frames_ab, load_img_params,
        iter_training_examples, record_training_example, drop_training_example,
        has_training_example,
)
from atompaint.transform_pred.datasets.origins import (
        OriginParams, choose_origins_for_atoms,
)
from atompaint.transform_pred.datasets.regression import (
        ViewPairParams, sample_view_pair, calc_min_distance_between_origins,
)

def ap_voxelize(
        sele='all',
        length_voxels=10,
        resolution_A=1,
        center_sele=None,
        channels='C,N,O',
        element_radii_A=None,
        state=-1,
        obj_name='voxels',
):
    atoms = atoms_from_pymol(sele, state)
    length_voxels = int(length_voxels)
    resolution_A = float(resolution_A)
    center_A = cmd.centerofmass(center_sele or sele, state)
    channels = parse_channels(channels)
    channel_colors = pick_channel_colors(sele, channels)
    element_radii_A = parse_element_radii_A(element_radii_A, resolution_A)
    state = int(state)

    img_params = ImageParams(
            grid=Grid(
                length_voxels=length_voxels,
                resolution_A=resolution_A,
                center_A=center_A,
            ),
            channels=channels,
            element_radii_A=element_radii_A,
    )
    render_view(
            obj_names=dict(
                voxels=obj_name,
            ),
            atoms_x=atoms,
            img_params=img_params,
            channel_colors=channel_colors,
    )

pymol.cmd.extend('ap_voxelize', ap_voxelize)
cmd.auto_arg[0]['ap_voxelize'] = cmd.auto_arg[0]['zoom']

def ap_view_pair(
        sele='all',
        length_voxels=10,
        resolution_A=1,
        origin_a=None,
        channels='C,N,O',
        element_radii_A=None,
        min_nearby_atoms=25,
        nearby_radius_A=5,
        max_dist_A=2,
        random_seed=0,
        state=-1,
):
    atoms = atoms_from_pymol(sele, state)
    length_voxels = int(length_voxels)
    resolution_A = float(resolution_A)
    channels = parse_channels(channels)
    channel_colors = pick_channel_colors(sele, channels)
    element_radii_A = parse_element_radii_A(element_radii_A, resolution_A)
    min_nearby_atoms = int(min_nearby_atoms)
    nearby_radius_A = float(nearby_radius_A)
    max_dist_A = float(max_dist_A)
    rng = np.random.default_rng(int(random_seed))
    state = int(state)

    origin_params = OriginParams(
            min_nearby_atoms=min_nearby_atoms,
            radius_A=nearby_radius_A,
    )
    img_params = ImageParams(
            grid=Grid(
                length_voxels=length_voxels,
                resolution_A=resolution_A,
            ),
            channels=channels,
            element_radii_A=element_radii_A,
    )
    min_dist_A = calc_min_distance_between_origins(img_params)
    view_pair_params = ViewPairParams(
            min_dist_A=min_dist_A,
            max_dist_A=min_dist_A + max_dist_A,
    )

    origins = choose_origins_for_atoms(sele, atoms, origin_params)

    if origin_a is None:
        origin_a, _ = sample_origin(rng, origins)
    else:
        origin_a = get_coord(origin_a)

    view_pair = sample_view_pair(
            rng, atoms, origin_a, origins, view_pair_params)

    render_view(
            obj_names=dict(
                voxels='view_a',
                axes='frame_a',
            ),
            atoms_x=view_pair.atoms_a,
            img_params=img_params,
            channel_colors=channel_colors,
            frame_xi=view_pair.frame_ai,
            state=state,
    )
    render_view(
            obj_names=dict(
                voxels='view_b',
                axes='frame_b',
            ),
            atoms_x=view_pair.atoms_b,
            img_params=img_params,
            channel_colors=channel_colors,
            frame_xi=view_pair.frame_bi,
            state=state,
    )

pymol.cmd.extend('ap_view_pair', ap_view_pair)
cmd.auto_arg[0]['ap_view_pair'] = cmd.auto_arg[0]['zoom']

def ap_origin_weights(
        sele='all',
        min_nearby_atoms=10,
        nearby_radius_A=5,
        random_seed=0,
        palette='blue_yellow',
        state=-1,
):
    atoms = atoms_from_pymol(sele, state)
    min_nearby_atoms = int(min_nearby_atoms)
    nearby_radius_A = float(nearby_radius_A)
    state = int(state)

    origin_params = OriginParams(
            min_nearby_atoms=min_nearby_atoms,
            radius_A=nearby_radius_A,
    )
    origins = choose_origins_for_atoms(sele, atoms, origin_params)

    space = dict(
            b_factors=np.zeros(len(atoms)),
            weights=origins['weight'],
    )

    cmd.iterate(sele, 'b_factors[index-1] = b', space=space)
    cmd.alter(sele, 'b = weights.get(index-1, 0)', space=space)
    cmd.spectrum('b', palette, sele)
    cmd.alter(sele, 'b = b_factors[index-1]', space=space)

pymol.cmd.extend('ap_origin_weights', ap_origin_weights)
cmd.auto_arg[0]['ap_origin_weights'] = cmd.auto_arg[0]['zoom']

class TrainingExamples(Wizard):
    # TODO: It'd be nice to be able to show the actual prediction made by the 
    # model, for each training example.

    def __init__(self, recording_path, validation_path=None):
        super().__init__()

        db = load_recording(recording_path)
        self.training_examples = iter_training_examples(db)
        self.frames_ab = load_frames_ab(db)
        self.img_params = load_img_params(db)

        self.validation_path = validation_path
        if validation_path is not None:
            self.validation_db = init_recording(validation_path, self.frames_ab)

        self.curr_pdb_obj = None
        self.curr_training_example = None

        self.show_next_training_example()

    def get_panel(self):
        panel = [
                [1, "AtomPaint Training Examples", ''],
        ]

        if self.validation_path:
            if has_training_example(self.validation_db, self.curr_input_ab):
                button = [2, f"Remove from: \\090{self.validation_path}", 'cmd.get_wizard().remove_from_manual_validation_set()']
            else:
                button = [2, f"Add to: \\090{self.validation_path}", 'cmd.get_wizard().add_to_manual_validation_set()']

            panel.append(button)

        panel += [
                [2, "Next", 'cmd.get_wizard().show_next_training_example()'],
                [2, "Done", 'cmd.set_wizard()'],
        ]

        return panel

    def get_prompt(self):
        if self.curr_training_example:
            return [f"Seed: {self.curr_seed}"]

    def do_key(self, key, x, y, mod):
        # This is <Ctrl-Space>; see `wt_vs_mut` for details.
        if (key, mod) == (0, 2):
            self.show_next_training_example()
        else:
            return 0

        cmd.refresh_wizard()
        return 1

    def get_event_mask(self):
        return Wizard.event_mask_key

    def show_next_training_example(self):
        cmd.delete(self.curr_pdb_obj)

        try:
            self.curr_training_example = next(self.training_examples)
        except StopIteration:
            cmd.set_wizard()
            return

        seed, tag, frame_ia, b, images_ab = self.curr_training_example

        self.curr_pdb_obj = load_tag(tag)
        cmd.util.cbc('elem C')

        frame_ai = invert_coord_frame(frame_ia)
        frame_ab = self.frames_ab[b]
        frame_ib = frame_ab @ frame_ia
        frame_bi = invert_coord_frame(frame_ib)

        select_view(
                name='sele_a',
                sele=self.curr_pdb_obj,
                grid=self.img_params.grid,
                frame_ix=frame_ia,
        )
        select_view(
                name='sele_b',
                sele=self.curr_pdb_obj,
                grid=self.img_params.grid,
                frame_ix=frame_ib,
        )

        render_image(
                img=images_ab[0],
                grid=self.img_params.grid,
                outline=(1, 1, 0),
                frame_xi=frame_ai,
                channel_colors=pick_channel_colors(
                    'sele_a',
                    self.img_params.channels,
                ),
                obj_names=dict(
                    voxels='voxels_a',
                    outline='outline_a',
                    axes='axes',
                ),
        )
        render_image(
                img=images_ab[1],
                grid=self.img_params.grid,
                outline=(0.4, 0.4, 0),
                frame_xi=frame_bi,
                channel_colors=pick_channel_colors(
                    'sele_b',
                    self.img_params.channels,
                ),
                obj_names=dict(
                    voxels='voxels_b',
                    outline='outline_b',
                    axes='axes',
                ),
        )

        cmd.show('sticks', 'byres (sele_a or sele_b)')
        cmd.zoom('sele_a or sele_b', buffer=10)
        cmd.center('sele_a')

    def add_to_manual_validation_set(self):
        record_training_example(self.validation_db, *self.curr_training_example)
        cmd.refresh_wizard()

    def remove_from_manual_validation_set(self):
        drop_training_example(self.validation_db, self.curr_input_ab)
        cmd.refresh_wizard()

    @property
    def curr_seed(self):
        return self.curr_training_example[0]

    @property
    def curr_input_ab(self):
        return self.curr_training_example[4]

def ap_training_examples(recording_path, validation_path=None):
    wizard = TrainingExamples(recording_path, validation_path)
    cmd.set_wizard(wizard)

pymol.cmd.extend('ap_training_examples', ap_training_examples)

def render_view(
        *,
        obj_names,
        atoms_x,
        img_params,
        channel_colors,
        axes=False,
        frame_xi=None,
        state=-1,
):
    img = image_from_atoms(atoms_x, img_params)
    render_image(
            obj_names=obj_names,
            img=img,
            grid=img_params.grid,
            channel_colors=channel_colors,
            axes=axes,
            frame_xi=frame_xi,
            state=state,
    )

def render_image(
        *,
        obj_names,
        img,
        grid,
        channel_colors,
        axes=False,
        outline=False,
        frame_xi=None,
        state=-1,
):
    view = cmd.get_view()

    # Important to render the axes before the voxels.  I don't know why, but if 
    # the voxels are rendered first, PyMOL regards them as opaque (regardless 
    # of the `transparency_mode` setting.
    if axes:
        ax = cgo_axes()
        cmd.delete(obj_names['axes'])
        cmd.load_cgo(ax, obj_names['axes'])

    if outline:
        edges = cgo_cube_edges(grid.center_A, grid.length_A, outline)
        cmd.delete(obj_names['outline'])
        cmd.load_cgo(edges, obj_names['outline'])

    if img is not None:
        # If `transparency_mode` is disabled (which is the default), CGOs will 
        # be opaque no matter what.
        cmd.set('transparency_mode', 1)

        voxels = cgo_voxels(img, grid, channel_colors)
        cmd.delete(obj_names['voxels'])
        cmd.load_cgo(voxels, obj_names['voxels'])

    if frame_xi is not None:
        for obj in obj_names.values():
            frame_1d = list(frame_xi.flat)
            cmd.set_object_ttt(obj, frame_1d, state)

    cmd.set_view(view)

def select_view(name, sele, grid, frame_ix):
    n = cmd.count_atoms(sele)
    coords_i = np.zeros((n, 4))
    cmd.iterate_state(
            state=1,
            selection=sele,
            expression='coords_i[index-1] = (x, y, z, 1)',
            space=locals(),
    )

    coords_x = transform_coords(coords_i, frame_ix)
    
    half_len = grid.length_A / 2
    within_grid = np.logical_and(
            coords_x >= -half_len,
            coords_x <= half_len,
    ).all(axis=1)

    cmd.alter(sele, 'b = within_grid[index-1]', space=locals())
    cmd.select(name, 'b = 1')

def parse_channels(channels_str):
    return channels_str.split(',') + ['.*']

def parse_element_radii_A(element_radii_A, resolution_A):
    if element_radii_A is None:
        return resolution_A / 2
    else:
        return float(element_radii_A)

def pick_channel_colors(sele, channels):
    color_counts = defaultdict(Counter)
    get_channel = get_element_channel
    channel_cache = {}
    cmd.iterate(
            sele,
            'channel = get_channel(channels, elem, channel_cache); '
            'color_counts[channel][color] += 1',
            space=locals(),
    )

    colors = []
    for channel in range(len(channels)):
        most_common = color_counts[channel].most_common(1)
        if most_common:
            color_i = most_common[0][0]
            rgb = cmd.get_color_tuple(color_i)
        else:
            rgb = (1, 1, 1)
        colors.append(rgb)

    return colors

def load_tag(tag):
    # Tags don't necessarily have associated PDB/mmCIF files, they just need to 
    # specify enough information to create an `atoms` data frame.  For this 
    # reason, the core atompaint library doesn't provide a function for 
    # extracting a path from a tag.  We need that ability here, though, so we 
    # basically have to reverse-engineer the tag parser.
    #
    # Probably the most proper solution would be to teach atompaint how to 
    # generate PDB/mmCIF from `atoms` data frames, but for now this seems like 
    # too much work.

    m = re.fullmatch('pisces/(\w{4})\w+', tag)
    pdb_id = m.group(1)

    pdb_path = get_pdb_redo_path(pdb_id)
    cmd.load(pdb_path)

    return pdb_path.stem

def cgo_voxels(img, grid, channel_colors=None):
    c, w, h, d = img.shape
    voxels = []

    alpha = get_alpha(img)
    face_masks = pick_faces(alpha)

    if channel_colors is None:
        from matplotlib.cm import tab10
        channel_colors = tab10.colors[:c]
    if len(channel_colors) != c:
        raise ValueError(f"Image has {c} channels, but only {len(channel_colors)} were specified")

    for i, j, k in product(range(w), range(h), range(d)):
        if alpha[i, j, k] == 0:
            continue

        voxels += cgo_cube(
                get_voxel_center_coords(grid, np.array([i, j, k])),
                grid.resolution_A,
                color=mix_colors(channel_colors, img[:, i, j, k]),
                alpha=alpha[i, j, k],
                face_mask=face_masks[:, i, j, k],
        )

    return voxels

def cgo_cube(center, length, color=(1, 1, 1), alpha=1.0, face_mask=6 * (1,)):
    # The starting point for this function came from the PyMOL wiki:
    #
    # https://pymolwiki.org/index.php/Cubes
    #
    # However, this starting point (i) didn't support color or transparency and 
    # (ii) had some bugs relating to surface normals.

    verts = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
    ])
    verts = length * (verts - 0.5) + np.array(center)

    # The order in which the vertices are specified is important: it determines 
    # which direction the triangle faces.  Specifically, a triangle is facing 
    # the camera when its vertices appear in counter-clockwise order.
    #
    # https://stackoverflow.com/questions/8142388/in-what-order-should-i-send-my-vertices-to-opengl-for-culling#8142461
    #
    # Cube:
    #   2───6    y
    #  ╱│  ╱│    │
    # 3─┼─7 │    │
    # │ 0─┼─4    o───x
    # │╱  │╱    ╱
    # 1───5    z 
    #
    # Faces:
    #   x     -x      y     -y      z     -z
    # 7───6  2───3  2───6  1───5  3───7  6───2
    # │   │  │   │  │   │  │   │  │   │  │   │
    # │   │  │   │  │   │  │   │  │   │  │   │
    # 5───4  0───1  3───7  0───4  1───5  4───0
    #
    # In all of the triangle fans below, I'll start with the lower-left vertex 
    # (e.g. 0 for the -x face) and continue counter-clockwise.

    def face(normal, indices):
        return [
                BEGIN, TRIANGLE_FAN,
                ALPHA, alpha,
                COLOR, *color,
                NORMAL, *normal,
                VERTEX, *verts[indices[0]],
                VERTEX, *verts[indices[1]],
                VERTEX, *verts[indices[2]],
                VERTEX, *verts[indices[3]],
                END,
        ]

    faces = []
    x, y, z = np.eye(3)

    if face_mask[0]: faces += face(+x, [5, 4, 6, 7])
    if face_mask[1]: faces += face(-x, [0, 1, 3, 2])
    if face_mask[2]: faces += face(+y, [3, 7, 6, 2])
    if face_mask[3]: faces += face(-y, [0, 4, 5, 1])
    if face_mask[4]: faces += face(+z, [1, 5, 7, 3])
    if face_mask[5]: faces += face(-z, [4, 0, 2, 6])

    return faces

def cgo_cube_edges(center, length, color=(1, 1, 1)):
    verts = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
    ])
    verts = length * (verts - 0.5) + np.array(center)

    #   2───6
    #  ╱│  ╱│
    # 3─┼─7 │
    # │ 0─┼─4
    # │╱  │╱
    # 1───5

    edges = [
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7),
            (6, 7),
    ]

    cube = [
            BEGIN, LINES,
            COLOR, *color,
    ]

    for i, j in edges:
        cube += [
                VERTEX, *verts[i],
                VERTEX, *verts[j],
        ]

    cube += [
            END,
    ]

    return cube

def cgo_axes():
    w = 0.06        # cylinder width 
    l1 = 0.75       # cylinder length
    l2 = l1 + 0.25  # cylinder + cone length
    d = w * 1.618   # cone base diameter

    origin = np.zeros(3)
    x, y, z = np.eye(3)
    r, g, b = np.eye(3)

    return [
            CYLINDER, *origin, *(l1 * x), w, *r, *r,
            CYLINDER, *origin, *(l1 * y), w, *g, *g,
            CYLINDER, *origin, *(l1 * z), w, *b, *b,
            CONE, *(l1 * x), *(l2 * x), d, 0, *r, *r, 1, 1,
            CONE, *(l1 * y), *(l2 * y), d, 0, *g, *g, 1, 1,
            CONE, *(l1 * z), *(l2 * z), d, 0, *b, *b, 1, 1,
    ]

def get_coord(coord_or_sele):
    coord_pat = r'\s+'.join(3 * [r'([+-]?\d*\.?\d*)'])
    if m := re.match(coord_pat, coord_or_sele):
        return np.fromiter(m.groups(), dtype=float)
    else:
        return np.array(cmd.centerofmass(coord_or_sele))

def get_alpha(img):
    img = np.sum(img, axis=0)
    return np.clip(img, 0, 1)

def pick_faces(img):
    face_masks = np.ones((6, *img.shape), dtype=bool)

    face_masks[0, :-1] = img[:-1] > img[1:]
    face_masks[1, 1:] = img[1:] > img[:-1]
    face_masks[2, :, :-1] = img[:, :-1] > img[:, 1:]
    face_masks[3, :, 1:] = img[:, 1:] > img[:, :-1]
    face_masks[4, :, :, :-1] = img[:, :, :-1] > img[:, :, 1:]
    face_masks[5, :, :, 1:] = img[:, :, 1:] > img[:, :, :-1]

    return face_masks

def mix_colors(colors, weights=None):
    if weights is None:
        weights = np.ones(len(colors))

    weights = np.array(weights).reshape(-1, 1)
    ratios = weights / np.sum(weights)

    latent_in = np.array([
            mixbox.float_rgb_to_latent(x)
            for x in colors
    ])
    latent_out = np.sum(latent_in * ratios, axis=0)

    return mixbox.latent_to_float_rgb(latent_out)

