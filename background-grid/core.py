import numpy as np
from preprocessor.meshHandle.finescaleMesh import FineScaleMesh


class BackgroundGrid(object):
    def __init__(self, finescale_mesh_file: str, bg_mesh_file: str) -> None:
        self.finescale_mesh = FineScaleMesh(finescale_mesh_file)
        self.bg_mesh = FineScaleMesh(bg_mesh_file)
        self.bg_volumes_tree = None

    def run(self) -> None:
        pass

    def assign_finescale_volumes_to_bg_volumes(self) -> None:
        pass

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
