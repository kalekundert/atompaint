import pymol
import numpy as np
import mixbox

from pymol import cmd
from pymol.cgo import *
from itertools import product
from collections import Counter, defaultdict
from pprint import pprint

from atompaint.datasets import voxelize, transform_pred
from atompaint.datasets.atoms import atoms_from_pymol
from atompaint.datasets.coords import invert_coord_frame

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

    img_params = voxelize.ImageParams(
            grid=voxelize.Grid(
                length_voxels=length_voxels,
                resolution_A=resolution_A,
                center_A=center_A,
            ),
            channels=channels,
            element_radii_A=element_radii_A,
    )
    render_view(
            dict(
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
        channels='C,N,O',
        element_radii_A=None,
        min_neighbors=10,
        neighbor_radius_A=5,
        min_dist_A=10,
        max_dist_A=15,
        random_seed=0,
        state=-1,
):
    atoms = atoms_from_pymol(sele, state)
    length_voxels = int(length_voxels)
    resolution_A = float(resolution_A)
    channels = parse_channels(channels)
    channel_colors = pick_channel_colors(sele, channels)
    element_radii_A = parse_element_radii_A(element_radii_A, resolution_A)
    min_neighbors = int(min_neighbors)
    neighbor_radius_A = float(neighbor_radius_A)
    min_dist_A = float(min_dist_A)
    max_dist_A = float(max_dist_A)
    rng = np.random.default_rng(int(random_seed))
    state = int(state)

    origin_params = transform_pred.OriginParams(
            min_neighbors=min_neighbors,
            radius_A=neighbor_radius_A,
    )
    view_pair_params = transform_pred.ViewPairParams(
            min_dist_A=min_dist_A,
            max_dist_A=max_dist_A,
    )
    img_params = voxelize.ImageParams(
            grid=voxelize.Grid(
                length_voxels=length_voxels,
                resolution_A=resolution_A,
            ),
            channels=channels,
            element_radii_A=element_radii_A,
    )

    origins = transform_pred.choose_origins_for_atoms(sele, atoms, origin_params)
    origin_a, _ = transform_pred.sample_origin(rng, origins)

    view_pair = transform_pred.sample_view_pair(
            rng, atoms, origin_a, origins, view_pair_params)

    np.set_printoptions(precision=3)
    print(
            view_pair.frame_ai,
    )
    render_view(
            dict(
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
            dict(
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
        min_neighbors=10,
        neighbor_radius_A=5,
        random_seed=0,
        palette='blue_yellow',
        state=-1,
):
    atoms = atoms_from_pymol(sele, state)
    min_neighbors = int(min_neighbors)
    neighbor_radius_A = float(neighbor_radius_A)
    state = int(state)

    origin_params = transform_pred.OriginParams(
            min_neighbors=min_neighbors,
            radius_A=neighbor_radius_A,
    )
    origins = transform_pred.choose_origins_for_atoms(sele, atoms, origin_params)

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

def render_view(
        obj_names,
        atoms_x,
        img_params,
        channel_colors,
        axes=False,
        frame_xi=None,
        state=-1,
):
    img = voxelize.image_from_atoms(atoms_x, img_params)
    voxels = cgo_voxels(img, img_params.grid, channel_colors)

    # Important to render the axes before the voxels.  I don't know why, but if 
    # the voxels are rendered first, PyMOL regards them as opaque (regardless 
    # of the `transparency_mode` setting.
    if axes:
        ax = cgo_axes()
        cmd.delete(obj_names['axes'])
        cmd.load_cgo(ax, obj_names['axes'])

    # If `transparency_mode` is disabled (which is the default), CGOs will be 
    # opaque no matter what.
    cmd.set('transparency_mode', 1)

    cmd.delete(obj_names['voxels'])
    cmd.load_cgo(voxels, obj_names['voxels'])

    if frame_xi is not None:
        for obj in obj_names.values():
            frame_1d = list(frame_xi.flat)
            cmd.set_object_ttt(obj, frame_1d, state)

def parse_channels(channels_str):
    return channels_str.split(',') + ['.*']

def parse_element_radii_A(element_radii_A, resolution_A):
    if element_radii_A is None:
        return resolution_A / 2
    else:
        return float(element_radii_A)

def pick_channel_colors(sele, channels):
    color_counts = defaultdict(Counter)
    get_element_channel = voxelize._get_element_channel
    channel_cache = {}
    cmd.iterate(
            sele,
            'channel = get_element_channel(channels, elem, channel_cache); '
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
                voxelize._get_voxel_center_coords(grid, np.array([i, j, k])),
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
    # However, this given code (i) didn't support color or transparency and 
    # (ii) had some bugs relating to surface normals.

    x, y, z = np.eye(3)
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

    if face_mask[0]: faces += face(+x, [5, 4, 6, 7])
    if face_mask[1]: faces += face(-x, [0, 1, 3, 2])
    if face_mask[2]: faces += face(+y, [3, 7, 6, 2])
    if face_mask[3]: faces += face(-y, [0, 4, 5, 1])
    if face_mask[4]: faces += face(+z, [1, 5, 7, 3])
    if face_mask[5]: faces += face(-z, [4, 0, 2, 6])

    return faces

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

