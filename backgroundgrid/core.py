import numpy as np
import networkx as nx
import itertools
from scipy.spatial import Delaunay
from preprocessor.meshHandle.finescaleMesh import FineScaleMesh
from .utils import list_argmax


class BackgroundGrid(object):
    def __init__(self, finescale_mesh_file: str, bg_mesh_file: str) -> None:
        self.finescale_mesh = FineScaleMesh(finescale_mesh_file)
        self.bg_mesh = FineScaleMesh(bg_mesh_file)

        all_fine_vols = self.finescale_mesh.volumes.all[:]
        self.all_fine_volumes_neighbors = self.finescale_mesh.volumes.bridge_adjacencies(all_fine_vols, 2, 3)

        self.finescale_mesh.primal_volume_center[:] = -1
        self.finescale_mesh.primal_face_center[:] = -1

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

    def fix_primal_volume_assignment(self) -> None:
        disconnected_clusters = self.get_disconnected_clusters()

        while len(disconnected_clusters) > 0:
            disconnected_volumes = list(itertools.chain.from_iterable(disconnected_clusters))

            while len(disconnected_volumes) > 0:
                vol = disconnected_volumes.pop(0)
                vol_neighbors = self.all_fine_volumes_neighbors[vol]
                not_disconnected_neighbors = ~np.isin(vol_neighbors, disconnected_volumes)

                if np.any(not_disconnected_neighbors):
                    a_neighbor = vol_neighbors[not_disconnected_neighbors][0]
                    new_bg_volume_value = self.finescale_mesh.bg_volume[a_neighbor][0, 0]
                    self.finescale_mesh.bg_volume[vol] = int(new_bg_volume_value)
                else:
                    disconnected_volumes.append(vol)

            disconnected_clusters = self.get_disconnected_clusters()

    def get_disconnected_clusters(self) -> list:
        all_finescale_volumes = self.finescale_mesh.volumes.all[:]
        finescale_bg_volumes_values = self.finescale_mesh.bg_volume[:].flatten()

        finescale_volumes_by_bg_value = zip(finescale_bg_volumes_values, all_finescale_volumes)
        def keyfunc(it): return it[0]
        sorted_finescale_volumes_by_bg_value = sorted(finescale_volumes_by_bg_value, key=keyfunc)

        finescale_volumes_grouped_by_bg_value = [list(bg_value_id_pairs) for _, bg_value_id_pairs in itertools.groupby(
            sorted_finescale_volumes_by_bg_value, keyfunc)]
        finescale_clusters = [
            [vol_bg_value_pair[1] for vol_bg_value_pair in bg_value_id_pairs]
            for bg_value_id_pairs in finescale_volumes_grouped_by_bg_value]
        finescale_clusters_graphs = [nx.Graph() for _ in range(len(finescale_clusters))]

        for cluster, G in zip(finescale_clusters, finescale_clusters_graphs):
            face_neighbors = self.all_fine_volumes_neighbors[cluster]
            all_face_connections = [[(neighbor, vol) for neighbor in neighbors if neighbor in cluster]
                                    for vol, neighbors in zip(cluster, face_neighbors)]
            G.add_edges_from(itertools.chain.from_iterable(all_face_connections))

        finescale_non_connected_clusters_components = [
            list(nx.connected_components(G)) for G in finescale_clusters_graphs
            if nx.number_connected_components(G) > 1]
        finescale_non_connected_clusters_components_sizes = [
            [len(cluster) for cluster in clusters]
            for clusters in finescale_non_connected_clusters_components]
        indices_of_large_clusters = [list_argmax(sizes) for sizes in finescale_non_connected_clusters_components_sizes]

        for cluster, i in zip(finescale_non_connected_clusters_components, indices_of_large_clusters):
            del cluster[i]

        finescale_non_connected_clusters_components_flat = list(
            itertools.chain.from_iterable(finescale_non_connected_clusters_components))

        return finescale_non_connected_clusters_components_flat

    def assemble_primal_volumes(self) -> None:
        pass

    def set_primal_coarse_faces(self) -> None:
        pass

    def compute_primal_centers(self) -> None:
        bg_volumes = self.bg_mesh.volumes.all[:]
        bg_volumes_centroids = self.bg_mesh.volumes.center[:]
        fine_volumes_bg_volume_values = self.finescale_mesh.bg_volume[:].flatten()

        for bg_vol_id, bg_vol_center in zip(bg_volumes, bg_volumes_centroids):
            # Retrieve volumes in the background grid volume.
            fine_volumes_centers_in_bg_vol = self.finescale_mesh.volumes.center[fine_volumes_bg_volume_values ==
                                                                                bg_vol_id]

            # Compute primal volume centers.
            vol_centers_distance = np.linalg.norm(fine_volumes_centers_in_bg_vol - bg_vol_center, axis=1)
            primal_volume_center = np.argmin(vol_centers_distance)
            self.finescale_mesh.primal_volume_center[primal_volume_center] = 1

            # Compute the centers of the primal volume's faces.
            bg_vol_faces = self.bg_mesh.volumes.adjacencies[bg_vol_id]
            bg_vol_faces_centers = self.bg_mesh.faces.center[bg_vol_faces]

            # Find the fine faces in the BG volume boundary.
            fine_volumes_in_bg_volume = self.finescale_mesh.volumes.all[fine_volumes_bg_volume_values == bg_vol_id]
            fine_faces_in_bg_volume = np.unique(
                self.finescale_mesh.volumes.adjacencies[fine_volumes_in_bg_volume].flatten())
            fine_faces_neighbors = self.finescale_mesh.faces.bridge_adjacencies(fine_faces_in_bg_volume, 2, 3)
            boundary_fine_faces_mask = [(~np.isin(neighbors, fine_volumes_in_bg_volume)).any()
                                        for neighbors in fine_faces_neighbors]
            boundary_fine_faces = fine_faces_in_bg_volume[boundary_fine_faces_mask]
            boundary_fine_faces_centers = self.finescale_mesh.faces.center[boundary_fine_faces]

            # Compute primal faces centers.
            primal_faces_centers_mask = [np.linalg.norm(boundary_fine_faces_centers - c, axis=1).argmin()
                                         for c in bg_vol_faces_centers]
            primal_faces_centers = boundary_fine_faces[primal_faces_centers_mask]
            self.finescale_mesh.primal_face_center[primal_faces_centers] = 1

    def primal_face_to_center_path(self) -> None:
        pass
