import numpy as np
import networkx as nx
import itertools
from scipy.spatial import Delaunay
from scipy.special import comb
from preprocessor.meshHandle.finescaleMesh import FineScaleMesh
from .utils import list_argmax, remove_tuples_duplicate


class BackgroundGrid(object):
    def __init__(self, finescale_mesh_file: str, bg_mesh_file: str) -> None:
        self.finescale_mesh = FineScaleMesh(finescale_mesh_file)
        self.bg_mesh = FineScaleMesh(bg_mesh_file)
        self.primal_volumes = None

        all_fine_vols = self.finescale_mesh.volumes.all[:]
        all_fine_faces = self.finescale_mesh.faces.all[:]
        self.all_fine_volumes_neighbors = self.finescale_mesh.volumes.bridge_adjacencies(
            all_fine_vols, 2, 3)
        self.all_fine_volumes_sharing_face = self.finescale_mesh.faces.bridge_adjacencies(
            all_fine_faces, 2, 3)
        self.all_fine_volumes_adjacencies = self.finescale_mesh.volumes.adjacencies[:]

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
        disconnected_clusters = self._get_disconnected_clusters()

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

            disconnected_clusters = self._get_disconnected_clusters()

        all_clusters = self._group_fine_volumes_by_bg_value()
        singletons = [cluster[0] for cluster in all_clusters if len(cluster) == 1]

        if len(singletons) > 0:
            singletons_neighbors = self.finescale_mesh.volumes.bridge_adjacencies(singletons, 2, 3)
            new_bg_values_of_singletons = [self.finescale_mesh.bg_volume[neighbors[0]][0, 0]
                                           for neighbors in singletons_neighbors]
            self.finescale_mesh.bg_volume[singletons] = new_bg_values_of_singletons

        self.primal_volumes = self._group_fine_volumes_by_bg_value()

    def set_primal_coarse_faces(self):
        clusters_fine_faces = [
            self.finescale_mesh.volumes.adjacencies[cluster]
            for cluster in self.primal_volumes]

        for cluster, cluster_faces in zip(self.primal_volumes, clusters_fine_faces):
            cluster_faces_flat = np.unique(np.concatenate(cluster_faces))

            # Find the fine volumes sharing the faces inside the cluster.
            adjacent_volumes = self.all_fine_volumes_sharing_face[cluster_faces_flat]

            # Find the fine faces at the boundary of the global domain. Those are by definition
            # primal faces.
            num_of_adjacent_volumes_per_face = np.vectorize(len)(adjacent_volumes)
            global_boundary_fine_faces = cluster_faces_flat[num_of_adjacent_volumes_per_face == 1]

            # Now, find which fine faces are on the boundary of the primal volume, i.e., the faces that
            # are shared with a fine volume outside the current primal volume.
            internal_fine_faces = cluster_faces_flat[num_of_adjacent_volumes_per_face == 2]
            internal_fine_faces_adjacent_volumes = np.vstack(adjacent_volumes[num_of_adjacent_volumes_per_face == 2])
            primal_internal_faces_mask = np.any(
                ~np.isin(internal_fine_faces_adjacent_volumes, cluster, assume_unique=True), axis=1)
            primal_internal_faces = internal_fine_faces[primal_internal_faces_mask]

            # Finally, set the primal faces.
            self.finescale_mesh.primal_face[global_boundary_fine_faces] = 1
            self.finescale_mesh.primal_face[primal_internal_faces] = 1

    def compute_primal_centers(self) -> None:
        # Compute primal volume centers.
        for cluster in self.primal_volumes:
            bg_vol_id = int(self.finescale_mesh.bg_volume[cluster[0]][0, 0])
            bg_vol_center = self.bg_mesh.volumes.center[bg_vol_id][0]
            fine_volumes_centers_in_bg_vol = self.finescale_mesh.volumes.center[cluster]

            vol_centers_distance = np.linalg.norm(fine_volumes_centers_in_bg_vol - bg_vol_center, axis=1)
            primal_volume_center = cluster[np.argmin(vol_centers_distance)]
            self.finescale_mesh.primal_volume_center[primal_volume_center] = 1

        # Compute primal faces centers.
        all_bg_faces = self.bg_mesh.faces.all[:]
        primal_face_prop_values = self.finescale_mesh.primal_face[:].flatten()
        fine_faces_in_primal_faces = self.finescale_mesh.faces.all[primal_face_prop_values == 1]
        fine_faces_in_primal_faces_centers = self.finescale_mesh.faces.center[fine_faces_in_primal_faces]

        for bg_face in all_bg_faces:
            bg_face_center = self.bg_mesh.faces.center[bg_face].flatten()
            primal_face_center_index = np.linalg.norm(
                fine_faces_in_primal_faces_centers - bg_face_center, axis=1).argmin()
            primal_face_center = fine_faces_in_primal_faces[primal_face_center_index]
            self.finescale_mesh.primal_face_center[primal_face_center] = 1

    def compute_dual_mesh_edges(self) -> None:
        # First, retrieve the primal volumes clusters and its faces.
        clusters_faces = [np.unique(self.finescale_mesh.volumes.adjacencies[cluster].flatten())
                          for cluster in self.primal_volumes]

        # Retrieve the primal faces centers for each cluster.
        clusters_primal_face_center_values = [
            self.finescale_mesh.primal_face_center[faces].flatten() for faces in clusters_faces]
        clusters_faces_centers = [
            faces[primal_face_center_values == 1] for primal_face_center_values,
            faces in zip(clusters_primal_face_center_values, clusters_faces)]
        clusters_faces_centers_centroids = [self.finescale_mesh.faces.center[primal_faces_centers]
                                            for primal_faces_centers in clusters_faces_centers]

        # Retrieve the primal volumes centers.
        clusters_primal_volume_center_values = [
            self.finescale_mesh.primal_volume_center[vols].flatten() for vols in self.primal_volumes]
        index_of_clusters_volumes_centers = [
            np.where(primal_volume_center_values == 1)[0][0]
            for primal_volume_center_values in clusters_primal_volume_center_values]
        clusters_volumes_centers = [cluster[i]
                                    for cluster, i in zip(self.primal_volumes, index_of_clusters_volumes_centers)]
        clusters_volumes_centers_centroids = [self.finescale_mesh.volumes.center[primal_center].flatten()
                                              for primal_center in clusters_volumes_centers]

        # Finally, compute the path between the primal volume center and its faces centers.
        for cluster, cluster_faces, primal_center, primal_faces_centers in zip(self.primal_volumes, clusters_faces,
                                                                               clusters_volumes_centers_centroids,
                                                                               clusters_faces_centers_centroids):
            fine_volumes_in_dual_edge = np.concatenate([self._check_intersections_along_axis(
                primal_center, primal_face_center, cluster_faces, cluster)
                for primal_face_center in primal_faces_centers])
            self.finescale_mesh.dual_mesh_edge[fine_volumes_in_dual_edge] = 1

    def set_dual_mesh_faces(self) -> None:
        coarse_volumes = self.bg_mesh.volumes.all[:]
        coarse_faces = self.bg_mesh.faces.all[:]

        coarse_volumes_adjacencies = self.bg_mesh.volumes.adjacencies[:]
        coarse_faces_neighbors = self.bg_mesh.faces.bridge_adjacencies(coarse_faces, 1, 2)

        num_of_coarse_vols = coarse_volumes.shape[0]
        num_faces_of_coarse_vol = coarse_volumes_adjacencies.shape[1]
        num_of_coarse_faces_pairs = int(comb(num_faces_of_coarse_vol, 2))
        faces_centers_pairs_per_volume = np.empty((num_of_coarse_vols, num_of_coarse_faces_pairs, 2), dtype=int)

        for coarse_vol_adjacencies, coarse_vol in zip(coarse_volumes_adjacencies, coarse_volumes):
            adjacencies_neighbors = coarse_faces_neighbors[coarse_vol_adjacencies]
            neighbors_in_coarse_vol = [
                neigh[np.isin(neigh, coarse_vol_adjacencies, assume_unique=True)] for neigh in adjacencies_neighbors]
            faces_pairs = [list(itertools.zip_longest([f], neigh, fillvalue=f))
                           for f, neigh in zip(coarse_vol_adjacencies, neighbors_in_coarse_vol)]
            faces_pairs_flat = list(itertools.chain.from_iterable(faces_pairs))
            faces_pairs_flat_unique = remove_tuples_duplicate(faces_pairs_flat)
            faces_centers_pairs_per_volume[coarse_vol, :, :] = np.array(faces_pairs_flat_unique)

        coarse_centers = self.bg_mesh.volumes.center[:]
        coarse_faces_centers_pairs = self.bg_mesh.faces.center[faces_centers_pairs_per_volume.flatten()].reshape(
            (num_of_coarse_vols, num_of_coarse_faces_pairs, 2, 3))

        # Computing the plane parameters for each dual face.
        C0 = coarse_faces_centers_pairs[:, :, 0, :]
        C1 = coarse_faces_centers_pairs[:, :, 1, :]

        # Array of normal vectors of each dual face plane.
        N = np.cross(C0 - coarse_centers[:, np.newaxis], C1 - coarse_centers[:, np.newaxis])

        # For each primal volume, find which fine volumes in the primal volume
        # are intersected by a dual face plane.
        fine_vols_bg_value = self.finescale_mesh.bg_volume[:].flatten()

        # Number of vertices in a fine volume.
        Nv_fine = self.finescale_mesh.volumes.connectivities[0].shape[0]

        for coarse_vol in coarse_volumes:
            fine_vols_in_coarse_vol = self.finescale_mesh.volumes.all[fine_vols_bg_value == coarse_vol]
            N_fine_vols = fine_vols_in_coarse_vol.shape[0]

            fine_vols_connectivities = self.finescale_mesh.volumes.connectivities[fine_vols_in_coarse_vol].flatten()
            fine_vols_connectivities_coords = self.finescale_mesh.nodes.coords[fine_vols_connectivities].reshape(
                (N_fine_vols, Nv_fine, 3))

            # For each fine volume, check whether it has vertices on opposing
            # sides of the dual faces planes.
            coarse_center = coarse_centers[coarse_vol]
            dist_vectors = fine_vols_connectivities_coords - coarse_center
            P = dist_vectors.dot(N[coarse_vol].T)
            plane_intersection_matrix = np.abs(np.sign(P).sum(axis=1)) < Nv_fine

            # Check which projections are inside a dual face.
            for i in range(num_of_coarse_faces_pairs):
                fine_vols_intersected_by_face_plane = fine_vols_in_coarse_vol[plane_intersection_matrix[:, i]]

                intersected_centroids = self.finescale_mesh.volumes.center[fine_vols_intersected_by_face_plane]
                n = N[coarse_vol, i] / np.linalg.norm(N[coarse_vol, i])
                D = intersected_centroids - coarse_center
                proj_centroids = intersected_centroids - np.dot(D, n)[:, np.newaxis] * n

                # Computing the vectors defining the inside region of the dual face.
                p0, p1, p2 = coarse_center[:], C0[coarse_vol, i], C1[coarse_vol, i]
                A = p1 - p0
                B = p2 - p0

                # Find the orientation of the projections regarding A and B.
                V = proj_centroids - p0

                AxB = N[coarse_vol, i]
                VxA = np.cross(V, A)
                VxB = np.cross(V, B)

                AxB_dot_AxV = np.dot(AxB, -VxA.T)
                VxB_dot_VxA = np.einsum("ij,ij->i", VxB, VxA)

                # Check if the orthogonal vectors have opposing senses.
                dual_face_vols = fine_vols_intersected_by_face_plane[(AxB_dot_AxV >= 0) & (VxB_dot_VxA <= 0)]

                self.finescale_mesh.dual_mesh_face[dual_face_vols] = 1

    def _get_disconnected_clusters(self) -> list:
        finescale_clusters = self._group_fine_volumes_by_bg_value()
        finescale_clusters_graphs = [nx.Graph() for _ in range(len(finescale_clusters))]

        for cluster, G in zip(finescale_clusters, finescale_clusters_graphs):
            G.add_nodes_from(cluster)
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

    def _group_fine_volumes_by_bg_value(self):
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

        return finescale_clusters

    def _check_intersections_along_axis(self, c1, c2, faces, fine_volumes_cluster) -> np.ndarray:
        # Check for intersection between the box's axis and the mesh faces.
        num_faces = len(faces)
        faces_nodes_handles = self.finescale_mesh.faces.connectivities[faces]
        num_vertices_in_face = faces_nodes_handles.shape[1]
        faces_vertices = self.finescale_mesh.nodes.coords[faces_nodes_handles.flatten()].reshape(
            (num_faces, num_vertices_in_face, 3))

        # Plane parameters of each face.
        R_0 = faces_vertices[:, 0, :]
        N = np.cross(faces_vertices[:, 1, :] - R_0,
                     faces_vertices[:, 2, :] - R_0)

        # Compute the parameters of the main axis line.
        num = np.einsum("ij,ij->i", N, R_0 - c1)
        denom = N.dot(c2 - c1)

        non_zero_denom = denom[np.abs(denom) > 1e-6]
        non_zero_num = num[np.abs(denom) > 1e-6]
        r = non_zero_num / non_zero_denom

        # Check faces intersected by the axis' line.
        filtered_faces = faces[np.abs(denom) > 1e-6]
        filtered_faces = filtered_faces[(r >= 0) & (r <= 1)]
        filtered_nodes = faces_vertices[np.abs(denom) > 1e-6]
        filtered_nodes = filtered_nodes[(r >= 0) & (r <= 1)]

        r = r[(r >= 0) & (r <= 1)]
        P = c1 + r[:, np.newaxis]*(c2 - c1)

        # Compute the intersection point between the face plane and the axis
        # line and check if such point is in the face.
        angle_sum = np.zeros(filtered_nodes.shape[0])
        for i in range(num_vertices_in_face):
            p0, p1 = filtered_nodes[:, i, :], filtered_nodes[:,
                                                             (i+1) % num_vertices_in_face, :]
            a = p0 - P
            b = p1 - P
            norm_prod = np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)
            # If the point of intersection is too close to a vertex, then
            # take it as the vertex itself.
            angle_sum[norm_prod <= 1e-6] = 2*np.pi
            cos_theta = np.einsum("ij,ij->i", a, b) / norm_prod
            theta = np.arccos(cos_theta)
            angle_sum += theta

        # If the sum of the angles around the intersection point is 2*pi, then
        # the point is inside the polygon.
        intersected_faces = filtered_faces[np.abs(2*np.pi - angle_sum) < 1e-6]

        try:
            volumes_sharing_face = self.finescale_mesh.faces.bridge_adjacencies(
                intersected_faces, "faces", "volumes")
        except:
            volumes_sharing_face = np.array([], dtype=int)

        try:
            volumes_sharing_face_flat = np.concatenate(volumes_sharing_face)
        except:
            volumes_sharing_face_flat = volumes_sharing_face[:]

        unique_volumes = np.unique(volumes_sharing_face_flat)
        unique_volumes_in_cluster = unique_volumes[np.isin(unique_volumes, fine_volumes_cluster)]

        return unique_volumes_in_cluster
