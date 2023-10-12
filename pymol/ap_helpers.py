import pymol
import numpy as np
import mixbox
import random
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
from atompaint.transform_pred.datasets.recording import (
        init_manual_predictions, record_manual_prediction,
        load_recording, load_frames_ab, load_img_params,
        load_all_training_example_ids, load_training_example,
)
from atompaint.transform_pred.datasets.origins import (
        OriginParams, choose_origins_for_atoms,
)

def ap_voxelize(
        center_sele=None,
        all_sele='all',
        length_voxels=21,
        resolution_A=0.75,
        channels='C,N,O',
        element_radii_A=None,
        outline=False,
        state=-1,
        sele_name='within_img',
        obj_name='voxels',
        outline_name='outline',
):
    atoms = atoms_from_pymol(all_sele, state)
    length_voxels = int(length_voxels)
    resolution_A = float(resolution_A)
    center_A = np.array(cmd.centerofmass(center_sele or all_sele, state))
    channels = parse_channels(channels)
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
    select_view(
            sele_name,
            all_sele,
            img_params.grid,
    )
    render_view(
            obj_names=dict(
                voxels=obj_name,
                outline=outline_name,
            ),
            atoms_x=atoms,
            img_params=img_params,
            channel_colors=pick_channel_colors(sele_name, channels),
            outline=outline,
    )

pymol.cmd.extend('ap_voxelize', ap_voxelize)
cmd.auto_arg[0]['ap_voxelize'] = cmd.auto_arg[0]['zoom']

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

    def __init__(self, recording_path, initial_example_id=None):
        super().__init__()

        self.db = db = load_recording(recording_path)
        self.example_ids = load_all_training_example_ids(db)
        self.img_params = load_img_params(db)

        if initial_example_id is None:
            self._i = 0
        else:
            self._i = self.example_ids.find(initial_example_id)

        self.curr_training_example = None
        self.curr_pdb_obj = None

        self.update_training_example()

    def get_panel(self):
        panel = [
                [1, "AtomPaint Training Examples", ''],
                [2, "Next", 'cmd.get_wizard().show_next_training_example()'],
                [2, "Previous", 'cmd.get_wizard().show_prev_training_example()'],
                [2, "Done", 'cmd.set_wizard()'],
        ]
        return panel

    def get_prompt(self):
        return [f"Example: {self.curr_training_example_id}"]

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
        self._i = (self._i + 1) % len(self.example_ids)
        self.update_training_example()

    def show_prev_training_example(self):
        self._i = (self._i - 1) % len(self.example_ids)
        self.update_training_example()

    def update_training_example(self):
        id = self.curr_training_example_id
        self.curr_training_example = ex = load_training_example(self.db, id)

        if self.curr_pdb_obj:
            cmd.delete(self.curr_pdb_obj)

        self.curr_pdb_obj = load_tag(ex['tag'], all_chains=True)
        cmd.util.cbc('elem C')

        frame_ia = ex['frame_ia']
        frame_ai = invert_coord_frame(frame_ia)
        frame_ib = ex['frame_ab'] @ frame_ia
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
                img=ex['input'][0],
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
                ),
        )
        render_image(
                img=ex['input'][1],
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
                ),
        )

        cmd.show('sticks', 'byres (sele_a or sele_b)')
        cmd.zoom('sele_a or sele_b', buffer=10)
        cmd.center('sele_a')

    @property
    def curr_training_example_id(self):
        return self.example_ids[self._i]

def ap_training_examples(recording_path, validation_path=None):
    wizard = TrainingExamples(recording_path, validation_path)
    cmd.set_wizard(wizard)

pymol.cmd.extend('ap_training_examples', ap_training_examples)

class ManualClassifier(Wizard):

    def __init__(self, recording_path):
        super().__init__()

        self.db = db = load_recording(recording_path)
        self.frames_ab = load_frames_ab(db)
        self.frame_names = get_frame_names(self.frames_ab)
        self.img_params = load_img_params(db)

        self.example_ids = load_all_training_example_ids(db)
        self.curr_example_id = None
        self.curr_training_example = None

        init_manual_predictions(db)
        self.init_settings()
        self.init_view_boxes()
        self.init_random_example()

    def get_panel(self):
        return [
                [1, "AtomPaint Manual Classifier", ''],
                [2, "Submit", 'cmd.get_wizard().submit_guess()'],
                [2, "Skip", 'cmd.get_wizard().skip_guess()'],
                [2, "Done", 'cmd.set_wizard()'],
        ]

    def get_prompt(self):
        return [self.frame_names[self.curr_b]]

    def do_key(self, key, x, y, mod):
        tab = (9, 0)
        ctrl_tab = (9, 2)

        def update_guess(step):
            next_b = (self.curr_b + step) % len(self.frames_ab)
            self.update_guess(next_b)

        if (key, mod) == tab:
            update_guess(1)
            return 1
        if (key, mod) == ctrl_tab:
            update_guess(-1)
            return 1

        return 0

    def get_event_mask(self):
        return Wizard.event_mask_key

    def init_settings(self):
        cmd.set('cartoon_gap_cutoff', 0)

    def init_view_boxes(self):
        grid = self.img_params.grid
        bright_yellow = 1, 1, 0
        dim_yellow = 0.4, 0.4, 0
        dim_red = 0.4, 0, 0
        dim_green = 0, 0.4, 0
        dim_blue = 0, 0, 0.4

        frame_colors = {
                '+X': dim_red,
                '+Y': dim_green,
                '+Z': dim_blue,
        }

        boxes = []
        boxes += cgo_cube_edges(grid.center_A, grid.length_A, dim_yellow)

        for i, frame_ab in enumerate(self.frames_ab):
            origin = get_origin(frame_ab)
            color = frame_colors.get(self.frame_names[i], dim_yellow)
            boxes += cgo_cube_edges(origin, grid.length_A, color)

        cmd.load_cgo(boxes, 'positions')
        
    def init_random_example(self):
        self.curr_example_id = random.choice(self.example_ids)
        self.curr_training_example = load_training_example(
                self.db,
                self.curr_example_id,
        )

        frame_ia = self.curr_frame_ia
        frame_ib = self.curr_frame_ab @ frame_ia

        pdb_obj = load_tag(self.curr_tag)
        cmd.hide('everything', pdb_obj)

        select_view(
                name='sele_a',
                sele=pdb_obj,
                grid=self.img_params.grid,
                frame_ix=frame_ia,
        )
        select_view(
                name='sele_b',
                sele=pdb_obj,
                grid=self.img_params.grid,
                frame_ix=frame_ib,
        )

        cmd.delete('view_a')
        cmd.delete('view_b')

        cmd.extract('view_a', 'sele_a')
        cmd.extract('view_b', 'sele_b')

        cmd.delete('sele_a')
        cmd.delete('sele_b')
        cmd.delete(pdb_obj)

        cmd.util.cbag()

        frame_1d = list(frame_ia.flat)
        cmd.set_object_ttt('view_a', frame_1d)

        self.update_guess(0)

        cmd.show('cartoon', 'view_a or view_b')
        cmd.show('sticks', 'view_a or view_b')
        cmd.zoom('positions', buffer=5)

    def update_guess(self, b):
        frame_aB = self.curr_frame_ab
        frame_ba = invert_coord_frame(self.frames_ab[b])
        frame_ia = frame_ba @ frame_aB @ self.curr_frame_ia
        frame_1d = list(frame_ia.flat)
        cmd.set_object_ttt('view_b', frame_1d)
        
        self.curr_b = b
        cmd.refresh_wizard()

    def submit_guess(self):
        record_manual_prediction(self.db, self.curr_example_id, self.curr_b)

        guess = self.frame_names[self.curr_b]
        answer = self.frame_names[self.curr_true_b]
        correct = (self.curr_b == self.curr_true_b)

        print(f"Guess: {guess};  Answer: {answer};  {'Correct' if correct else 'Incorrect'}!")

        self.init_random_example()

    def skip_guess(self):
        record_manual_prediction(self.db, self.curr_example_id, None)

        answer = self.frame_names[self.curr_true_b]
        print(f"Guess: --;  Answer: {answer};  Skipped!")

        self.init_random_example()

    @property
    def curr_tag(self):
        return self.curr_training_example['tag']

    @property
    def curr_frame_ia(self):
        return self.curr_training_example['frame_ia']

    @property
    def curr_frame_ab(self):
        return self.curr_training_example['frame_ab']

    @property
    def curr_true_b(self):
        return self.curr_training_example['b']

def ap_manual_classifier(recording_path):
    wizard = ManualClassifier(recording_path)
    cmd.set_wizard(wizard)

pymol.cmd.extend('ap_manual_classifier', ap_manual_classifier)

def render_view(
        *,
        obj_names,
        atoms_x,
        img_params,
        channel_colors,
        axes=False,
        outline=False,
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
            outline=outline,
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

def select_view(name, sele, grid, frame_ix=None):
    indices = []
    cmd.iterate(
            selection=sele,
            expression='indices.append(index)',
            space=locals(),
    )

    coords_i = np.zeros((len(indices), 4))
    i_from_index = {x: i for i, x in enumerate(indices)}
    cmd.iterate_state(
            selection=sele,
            expression='coords_i[i_from_index[index]] = (x, y, z, 1)',
            space=locals(),
            state=1,
    )

    if frame_ix is not None:
        coords_x = transform_coords(coords_i, frame_ix)
    else:
        coords_x = coords_i

    coords_x = coords_x[:,:3] - grid.center_A
    half_len = grid.length_A / 2
    within_grid = np.logical_and(
            coords_x >= -half_len,
            coords_x <= half_len,
    ).all(axis=1)

    cmd.alter(sele, 'b = within_grid[i_from_index[index]]', space=locals())
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

def load_tag(tag, all_chains=False):
    """
    Load the structure referenced by the given tag.

    When AtomPaint constructs training examples, it doesn't necessarily use all 
    of the atoms that are present in the underlying structure.  The purpose of 
    this function is to load even those atoms that weren't included, so we can 
    see if AtomPaint is missing anything important.

    Unfortunately, we have to do this in a somewhat hacky way.  The issue is 
    that tags are meant to be opaque names for sets of atoms.  They aren't 
    guaranteed to be associated with any PDB/mmCIF files, so AtomPaint doesn't 
    provide any way to get the "whole structure" associated with any tag.  So, 
    here we basically have to duplicate the tag-parsing logic.
    """
    m = re.fullmatch(r'pisces/(\w{4})(\w)', tag)
    pdb_id, chain = m.groups()

    pdb_path = get_pdb_redo_path(pdb_id)
    pdb_obj = pdb_path.stem

    cmd.load(pdb_path)

    if not all_chains:
        cmd.remove(f'{pdb_obj} and not chain {chain}')

    return pdb_obj

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
    if color and not isinstance(color, tuple):
        color = (1, 1, 0)

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

def get_frame_names(frames_ab):
    # Currently, only "cube face" frames are supported.
    names_from_origins = {
            ( 1,  0,  0): '+X',
            (-1,  0,  0): '-X',
            ( 0,  1,  0): '+Y',
            ( 0, -1,  0): '-Y',
            ( 0,  0,  1): '+Z',
            ( 0,  0, -1): '-Z',
    }
    names = []

    for frame_ab in frames_ab:
        origin = get_origin(frame_ab)
        direction = origin / np.linalg.norm(origin)
        key = tuple(np.rint(direction).astype(int))
        name = names_from_origins[key]
        names.append(name)

    return names

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

