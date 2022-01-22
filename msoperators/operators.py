import numpy as np
from scipy.sparse import dia_matrix, lil_matrix, dok_matrix
from itertools import chain


class MsRSBOperator(object):
    def __init__(self, finescale_mesh, coarse_mesh,
                 support_regions, support_boundaries, A):
        self.finescale_mesh = finescale_mesh
        self.coarse_mesh = coarse_mesh
        self.support_regions = support_regions
        self.support_boundaries = support_boundaries
        self.A = A

    def _init_mesh_data(self):
        all_volumes = self.finescale_mesh.core.all_volumes[:]
        all_tags = self.finescale_mesh.core.mb.tag_get_tags_on_entity(all_volumes[0])

        bg_volume_tag = [tag for tag in all_tags if tag.get_name() == "bg_volume"].pop()
        dual_face_tag = [tag for tag in all_tags if tag.get_name() == "dual_mesh_face"].pop()
        dual_edge_tag = [tag for tag in all_tags if tag.get_name() == "dual_mesh_edge"].pop()

        bg_volume_data = self.finescale_mesh.core.mb.tag_get_data(bg_volume_tag,
                                                                  all_volumes, flat=True)
        dual_face_data = self.finescale_mesh.core.mb.tag_get_data(dual_face_tag,
                                                                  all_volumes, flat=True)
        dual_edge_data = self.finescale_mesh.core.mb.tag_get_data(dual_edge_tag,
                                                                  all_volumes, flat=True)

        self.finescale_mesh.bg_volume[:] = bg_volume_data
        self.finescale_mesh.dual_mesh_face_all[:] = dual_face_data + dual_edge_data

    def compute_operators(self):
        # Initialize mesh data.
        self._init_mesh_data()

        # Initialize prolongation operator with initial guess.
        m = len(self.finescale_mesh.volumes)
        n = len(self.coarse_mesh.volumes)

        bg_volume_values = self.finescale_mesh.bg_volume[:].flatten()
        fine_vols_idx = self.finescale_mesh.volumes.all[:]

        P = lil_matrix((m, n))
        P[fine_vols_idx, bg_volume_values] = 1.0
        P = P.tocsr()

        # Initialize the increment Q of the prolongation operator.
        omega = 2.0 / 3.0
        D_inv = dia_matrix((m, m))
        A_diag = self.A.diagonal()
        D_inv.setdiag(1.0 / A_diag)
        Q = -omega * D_inv * self.A

        # Initialize matrix H of coarse volumes indices for each fine
        # volume in the global support boundary.
        G = np.unique(list(chain.from_iterable(self.support_boundaries.values())))
        H = dok_matrix((m, n))
        I = {}
        for coarse_id in self.support_boundaries:
            fine_idx_in_support_region = np.array(self.support_regions[coarse_id])
            fine_idx_mask = np.isin(fine_idx_in_support_region, G, assume_unique=True)

            fine_idx_in_G = fine_idx_in_support_region[fine_idx_mask]
            fine_idx_not_in_G = fine_idx_in_support_region[~fine_idx_mask]

            H[fine_idx_in_G, coarse_id] = 1.0
            I[coarse_id] = fine_idx_not_in_G

        H = H.tocsr()

        # Construct prolongation operator iteratively.
        tol = 1e-8
        d = dok_matrix((m, n))
        e = np.full(n, np.inf)
        not_in_G_mask = ~np.isin(fine_vols_idx, G, assume_unique=True)
        while np.linalg.norm(e, ord=np.inf) > tol:
            # Compute estimate of increment.
            d_hat = Q * P
            d_hat_sum = d_hat.multiply(H).sum(axis=1)

            # Update increment.
            for j in range(n):
                # Set the increment for the fine volumes in the global
                # support boundary.
                G_idx = self.support_boundaries[j]
                d[G_idx, j] = (d_hat[G_idx, j] - P[G_idx, j].multiply(d_hat_sum[G_idx])) / (1 + d_hat_sum[G_idx])

                # Set the increment for the fine volumes in the support
                # region but not in the global support boundary.
                d[I[j], j] = d_hat[I[j], j].todense()

            # Update operator.
            P = P + d

            # Update error.
            e = np.asarray(d_hat[not_in_G_mask].max(axis=0).todense()).flatten()

        # Set the restriction operator as the transpose of the
        # prolongation one.
        R = P.transpose()

        return P, R
