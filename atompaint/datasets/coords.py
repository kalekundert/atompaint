import numpy as np

from scipy.spatial.transform import Rotation
from numpy.typing import NDArray
from typing import TypeAlias, Annotated

"""\
Naming conventions
==================
When it's possibly unclear, a suffix should be used to indicate:

- Which coordinate frame a coordinate belongs to.

- Which coordinate frames a matrix transforms between.  Note that these 
  matrices are referred to as "frames" themselves, but are more accurately 
  thought of as transformations between two frames.

A coordinate frame is identified with a single letter.  Commonly used suffixes 
(in this code base) are:

- i: The input coordinate frame, or in other words, the coordinate frame 
  associated with the identity matrix.

- a, b: Two coordinate frames that have an implied order, i.e. 'a' comes before 
  'b' in some sense.

- x, y: Two generic coordinate frames.  These suffixes are most often used with 
  arguments to general-purpose functions

Matrices use two-letter suffixes.  The first letter is the starting coordinate 
frame, and the second letter is the resulting coordinate frame.  For example, 
consider the following line of code:

    >>> coords_y = transform_coords(coords_x, frame_xy)

We start with a coordinate in frame "X", apply a transformation matrix that 
goes from "X" to "Y", and end up with a coordinate in frame "Y".
"""

Coord: TypeAlias = Annotated[NDArray[float], (3)]
Coords3: TypeAlias = Annotated[NDArray[float], (-1, 3)]
Coords4: TypeAlias = Annotated[NDArray[float], (-1, 4)]
Frame: TypeAlias = Annotated[NDArray[float], (4, 4)]

def make_coord_frame(origin: Coord, rot_vec_rad: Coord) -> Frame:
    """\
    Provide a convenient way to construct coordinate frame matrices.

    For the purposes of describing how exactly this function works, let us 
    define that the matrix returned by this function will transform coordinates 
    from a coordinate frame labeled "X" to one labeled "Y".

    Arguments:
        origin:
            The origin of frame Y, expressed in frame X coordinates.  For 
            example, consider a coordinate frame located at (1,2,3).  If you 
            transform (0,0,0) into this frame, you'll get (-1,-2,-3) because 
            (0,0,0) is the origin of X in frame Y, which is the opposite of 
            what was specified.  If you transform (1,2,3), you'll get (0,0,0).

        rot_vec_rad:
            The orientation of frame Y, as seen from X.  The direction of the 
            vector gives the axis of rotation, while the magnitude gives the 
            angle (in radians).  The rotation obeys the right-hand rule, which 
            means that it's counter-clockwise from the perspective of an 
            observer facing the direction vector.

            For example, consider a frame Y that is rotated 90° around the 
            z-axis relative to frame X.  If you transform (1,0,0), you'll get 
            (0,-1,0).  Note that if we looked at these two coordinates in the 
            same frame, the second would appear to be a -90° rotation or the 
            first.  This happens because it's the *coordinate frame* that's 
            being rotated by 90°, not the coordinates themselves.
    """
    assert origin.size == 3
    assert rot_vec_rad.size == 3

    rot = Rotation.from_rotvec(rot_vec_rad).as_matrix()

    frame_yx = np.eye(4)
    frame_yx[0:3, 0:3] = rot
    frame_yx[0:3,   3] = origin.ravel()

    # We need to invert the matrix, because the arguments are both from the 
    # perspective of frame X, but the actual components of the matrix have to 
    # be from the perspective of frame Y.

    return invert_coord_frame(frame_yx)

def invert_coord_frame(frame: Frame) -> Frame:
    assert frame.shape == (4, 4)

    # https://math.stackexchange.com/questions/1234948/inverse-of-a-rigid-transformation

    r_inv = frame[0:3, 0:3].T

    inv = np.eye(4)
    inv[0:3, 0:3] = r_inv
    inv[0:3,   3] = -r_inv @ frame[0:3, 3]

    return inv

def get_origin(frame: Frame):
    frame = invert_coord_frame(frame)
    return frame[0:3, 3]

def get_rotation_matrix(frame: Frame):
    frame = invert_coord_frame(frame)
    return frame[0:3, 0:3]


def transform_coords(coords_x: Coords4, frame_xy: Frame) -> Coords4:
    assert coords_x.shape[-1] == 4
    return coords_x @ frame_xy.T

def homogenize_coords(coords: Coords3) -> Coords4:
    assert coords.shape[-1] == 3
    shape = *coords.shape[:-1], 1
    return np.hstack((coords, np.ones(shape)))
