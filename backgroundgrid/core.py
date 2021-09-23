import numpy as np
from preprocessor.meshHandle.finescaleMesh import FineScaleMesh
from rtree import index
from .utils import find_aabb_of_volumes


class BackgroundGrid(object):
    def __init__(self, finescale_mesh_file: str, bg_mesh_file: str) -> None:
        self.finescale_mesh = FineScaleMesh(finescale_mesh_file)
        self.bg_mesh = FineScaleMesh(bg_mesh_file)

        p = index.Property()
        p.dimension = 3
        self.bg_volumes_tree = index.Index(properties=p)

    def run(self) -> None:
        pass

    def assign_finescale_volumes_to_bg_volumes(self) -> None:
        # Construct bounding boxes for coarse volumes.
        bg_volumes_vertices_handles = self.bg_mesh.volumes.connectivities[:]
        bg_num_volumes, num_vertices_of_bg_volume = bg_volumes_vertices_handles.shape
        bg_volumes_vertices_coords = self.bg_mesh.nodes.coords[bg_volumes_vertices_handles.flatten()].reshape(
            (bg_num_volumes, num_vertices_of_bg_volume, 3))

        bg_volumes_bboxes = find_aabb_of_volumes(bg_volumes_vertices_coords)
        bg_volumes_ids = self.bg_mesh.volumes.all[:]

        # Insert bounding boxes into R-tree.
        [self.bg_volumes_tree.insert(volume_id, bbox) for volume_id, bbox in zip(bg_volumes_ids, bg_volumes_bboxes)]

        # Find the nearest bounding box for each finescale volume.
        finescale_volumes_centroids = self.finescale_mesh.volumes.center[:]
        nearest_bg_volumes_by_fine_volume = [
            list(self.bg_volumes_tree.intersection(
                (centroid[0],
                 centroid[1],
                 centroid[2],
                 centroid[0],
                 centroid[1],
                 centroid[2]))) for centroid in finescale_volumes_centroids]

        i = 0
        for centroid, nearest_bg_volumes in zip(finescale_volumes_centroids, nearest_bg_volumes_by_fine_volume):
            D = self.bg_mesh.volumes.center[nearest_bg_volumes] - centroid
            ds = np.linalg.norm(D, axis=1)
            nearest_bg_volume_id = nearest_bg_volumes[ds.argmin()]
            self.finescale_mesh.bg_volume[i] = nearest_bg_volume_id
            i += 1

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
