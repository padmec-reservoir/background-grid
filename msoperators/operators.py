import numpy as np
from scipy.sparse import dia_matrix, lil_matrix, dok_matrix, diags
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

        bg_volume_data = self.finescale_mesh.core.mb.tag_get_data(bg_volume_tag,
                                                                  all_volumes, flat=True)

        self.finescale_mesh.bg_volume[:] = bg_volume_data

    def compute_operators(self, dirichlet_idx=[], tol=1e-3):
        # Initialize mesh data.
        self._init_mesh_data()

        # Initialize prolongation operator with the initial guess.
        m = len(self.finescale_mesh.volumes)
        n = len(self.coarse_mesh.volumes)

        bg_volume_values = self.finescale_mesh.bg_volume[:].flatten()
        fine_vols_idx = self.finescale_mesh.volumes.all[:]

        P = lil_matrix((m, n))
        P[fine_vols_idx, bg_volume_values] = 1.0
        P[dirichlet_idx, :] = 0.0
        P = P.tocsr()

        # Initialize the increment Q of the prolongation operator.
        omega = 2.0 / 3.0
        D_inv = dia_matrix((m, m))
        A_diag = self.A.diagonal()
        D_inv.setdiag(1.0 / A_diag)
        Q = -omega * (D_inv @ self.A)

        # Get all volumes in the global support boundary.
        G = np.unique(list(chain.from_iterable(self.support_boundaries.values())))

        M = lil_matrix((n, m))
        for j, I_j in self.support_regions.items():
            M[j, I_j] = 1.0
        M = M.tocsc().transpose()

        # Construct prolongation operator iteratively.
        eps = 1e-10
        e = np.ones(m)
        not_in_G_mask = ~np.isin(fine_vols_idx, G, assume_unique=True)
        while np.linalg.norm(e, ord=np.inf) > tol:
            # Compute increment.
            d = (Q @ P).multiply(M)

            # Update operator.
            P = P + d
            P_sum = diags(1 / (P.sum(axis=1).A.ravel() + eps))
            P = P_sum @ P

            # Update error.
            e = np.asarray(d[not_in_G_mask].max(axis=0).todense()).flatten()

            print("Current error: {}".format(np.linalg.norm(e, ord=np.inf)))

        # Reimpose dirichlet volumes in the prolongation operator.
        dirichlet_coarse_ids = self.finescale_mesh.bg_volume[dirichlet_idx].flatten()
        P = P.tolil()
        P[dirichlet_idx, dirichlet_coarse_ids] = 1.0
        P = P.tocsr()

        # Set the restriction operator as the transpose of the
        # prolongation one.
        R = P.transpose()

        return P, R
