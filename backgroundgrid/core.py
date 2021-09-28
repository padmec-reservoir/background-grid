import numpy as np
from scipy.spatial import Delaunay
from preprocessor.meshHandle.finescaleMesh import FineScaleMesh


class BackgroundGrid(object):
    def __init__(self, finescale_mesh_file: str, bg_mesh_file: str) -> None:
        self.finescale_mesh = FineScaleMesh(finescale_mesh_file)
        self.bg_mesh = FineScaleMesh(bg_mesh_file)

    def run(self) -> None:
        pass

    def assign_finescale_volumes_to_bg_volumes(self) -> None:
        finescale_volumes_centroids = self.finescale_mesh.volumes.center[:]
        bg_volumes_ids = self.bg_mesh.volumes.all[:]
        bg_volumes_connectivities = self.bg_mesh.volumes.connectivities[:]
        bg_volumes_vertices = self.bg_mesh.nodes.coords[bg_volumes_connectivities.flatten()].reshape(
            (bg_volumes_connectivities.shape[0], 4, 3))
        bg_volumes_hulls = [Delaunay(vertices) for vertices in bg_volumes_vertices]
        centroids_in_bg_volumes_check = [hull.find_simplex(
            finescale_volumes_centroids) >= 0 for hull in bg_volumes_hulls]

        for centroids_in_bg_volume, bg_id in zip(centroids_in_bg_volumes_check, bg_volumes_ids):
            self.finescale_mesh.bg_volume[centroids_in_bg_volume] = int(bg_id)

    def assemble_primal_volumes(self) -> None:
        pass

    def find_disconnected_volumes(self) -> None:
        pass

    def fix_primal_volume_assignment(self, finescale_volumes) -> None:
        pass

    def set_primal_coarse_faces(self) -> None:
        pass

    def compute_primal_centers(self) -> None:
        pass

    def primal_face_to_center_path(self) -> None:
        pass
