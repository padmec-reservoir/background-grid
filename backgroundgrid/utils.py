import numpy as np


def find_aabb_of_volumes(volumes: np.ndarray) -> np.ndarray:
    """Computes the bounding box for each volume in `volumes`.

    Parameters
    ----------
    volumes : numpy.ndarray
        3D array which items are the coordinates of the vertices of
        each volume.

    Returns
    -------
    bbox : numpy.ndarray
        An array containing the bounding boxes for each volume in the input.
        Each line represents the bounding box of the corresponding volume at
        the same position. The bounding boxes are represented as 
        (x_min, y_min, z_min, x_max, y_max, z_max).
    """

    x_min, x_max = volumes[:, :, 0].min(axis=1), volumes[:, :, 0].max(axis=1)
    y_min, y_max = volumes[:, :, 1].min(axis=1), volumes[:, :, 1].max(axis=1)
    z_min, z_max = volumes[:, :, 2].min(axis=1), volumes[:, :, 2].max(axis=1)
    all_bboxes = np.vstack((x_min, y_min, z_min, x_max, y_max, z_max)).T
    return all_bboxes


def list_argmax(l: list) -> int:
    return max(zip(l, range(len(l))))[1]
