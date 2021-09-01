import numpy as np


def find_aabb_of_volumes(volumes: np.ndarray) -> tuple:
    """Computes the bounding box for each volume in `volumes`.

    Parameters
    ----------
    volumes : numpy.ndarray
        3D array which items are the coordinates of the vertices of
        each volume.

    Returns
    -------
    bbox : tuple
        A tuple representing the bounding box as (x_min, x_max, y_min, y_max, z_min, z_max).
    """

    x_min, x_max = volumes[:, :, 0].min(axis=1), volumes[:, :, 0].max(axis=1)
    y_min, y_max = volumes[:, :, 1].min(axis=1), volumes[:, :, 1].max(axis=1)
    z_min, z_max = volumes[:, :, 2].min(axis=1), volumes[:, :, 2].max(axis=1)
    bbox = (x_min, x_max, y_min, y_max, z_min, z_max)
    return bbox
