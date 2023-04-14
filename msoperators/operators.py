import numpy as np
from scipy.sparse import dia_matrix, lil_matrix, diags
from itertools import chain


class MsRSBOperator(object):
    def __init__(self, finescale_mesh, coarse_mesh,
                 support_regions, support_boundaries, A, q):
        self.finescale_mesh = finescale_mesh
        self.coarse_mesh = coarse_mesh
        self.support_regions = support_regions
        self.support_boundaries = support_boundaries
        self.A = A
        self.q = q

    def _init_mesh_data(self):
        all_volumes = self.finescale_mesh.core.all_volumes[:]
        all_tags = self.finescale_mesh.core.mb.tag_get_tags_on_entity(all_volumes[0])

        bg_volume_tag = [tag for tag in all_tags if tag.get_name() == "bg_volume"].pop()

        bg_volume_data = self.finescale_mesh.core.mb.tag_get_data(bg_volume_tag,
                                                                  all_volumes, flat=True)

        self.finescale_mesh.bg_volume[:] = bg_volume_data

    def compute_operators(self, tol=1e-3, maxiter=50):
        # Initialize mesh data.
        self._init_mesh_data()

        # Initialize prolongation operator with the initial guess.
        m = len(self.finescale_mesh.volumes)
        n = len(self.coarse_mesh.volumes)

        bg_volume_values = self.finescale_mesh.bg_volume[:].flatten()
        fine_vols_idx = self.finescale_mesh.volumes.all[:]

        dirichlet_idx = self.finescale_mesh.faces.bridge_adjacencies(
            self.finescale_mesh.faces.boundary[:], 2, 3).flatten()
        dirichlet_primal_idx = self.finescale_mesh.bg_volume[dirichlet_idx].flatten()

        P = lil_matrix((m, n))
        P[fine_vols_idx, bg_volume_values] = 1.0
        P[dirichlet_idx, dirichlet_primal_idx] = 1.0
        P = P.tocsr()

        # Compute the preconditioned matrix.
        diag_A = self.A.diagonal()
        A_row_sum = self.A.sum(axis=1).A.ravel()
        A_precond = self.A - diags(A_row_sum - diag_A)

        # Initialize the increment Q of the prolongation operator.
        omega = 2 / 3
        D_inv = diags(1 / A_precond.diagonal())
        Q = -omega * (D_inv @ A_precond)

        # Get all volumes in the global support boundary.
        all_vols_in_sup_boundary = list(chain.from_iterable(self.support_boundaries.values()))

        # Get all volumes in the global support boundary (G).
        G = np.unique(all_vols_in_sup_boundary)
        notG = ~np.isin(fine_vols_idx, G, assume_unique=True)

        # Assemble some masks to filter volumes during the iterative
        # construction of the operators.
        # M -> all volumes in the support region
        # H -> all volumes in the support region and in the global support boundary
        # I -> all volumes in the support region but not in the global support boundary
        M = lil_matrix((n, m))
        H = lil_matrix((n, m))
        I = lil_matrix((n, m))
        for j in range(n):
            S_j, B_j = self.support_regions[j], self.support_boundaries[j]
            S_in_j = np.setdiff1d(S_j, B_j, assume_unique=True)
            I_j = np.setdiff1d(S_j, G, assume_unique=True)
            H_j = np.intersect1d(S_in_j, G, assume_unique=True)
            M[j, S_in_j] = 1.0
            H[j, H_j] = 1.0
            I[j, I_j] = 1.0

        # Remove the centres
        xP = np.nonzero(self.finescale_mesh.support_region_center[:])[0]
        xP_idx = self.finescale_mesh.bg_volume[xP].flatten()
        M[xP_idx, xP] = 0
        H[xP_idx, xP] = 0
        I[xP_idx, xP] = 0

        # Force the volumes belonging to a single support region
        # to hold the maximum value of the basis function.
        vols_in_a_single_sr = np.nonzero(M.sum(axis=0).A.ravel() == 1)[0]
        vols_in_a_single_sr_primal_ids = self.finescale_mesh.bg_volume[vols_in_a_single_sr].flatten()
        M[vols_in_a_single_sr_primal_ids, vols_in_a_single_sr] = 0
        H[vols_in_a_single_sr_primal_ids, vols_in_a_single_sr] = 0
        I[vols_in_a_single_sr_primal_ids, vols_in_a_single_sr] = 0

        # Convert the masks to CSC format for better performance.
        M = M.tocsc().transpose()
        H = H.tocsc().transpose()
        I = I.tocsc().transpose()

        # Construct the prolongation operator iteratively.
        e = np.ones(m)
        local_err = np.inf
        niter = 0
        while local_err > tol and niter < maxiter:
            niter += 1

            # Compute the initial increment.
            d_hat = (Q @ P).multiply(M)

            # Adapt the increment within the global support boundary.
            P_H = P.multiply(H)
            d_hat_in_H = d_hat.multiply(H)
            Sd_H = d_hat_in_H.sum(axis=1).A.ravel()
            d_H = (d_hat_in_H - P_H.multiply(Sd_H[:, None])).multiply((1 + Sd_H[:, None]) ** -1)

            # Filter the increment for the volumes outside of G.
            d_I = d_hat.multiply(I)

            # Set the final increment.
            d = d_H + d_I

            # Update the operator.
            P = P + d
            P_sum = diags(1 / P.sum(axis=1).A.ravel())
            P = P_sum @ P

            # Update error.
            e = np.asarray(d[notG].max(axis=0).todense()).flatten()
            local_err = np.linalg.norm(e, ord=np.inf)

            print("Current error: {}".format(local_err))

        # Set the restriction operator as the transpose of the
        # prolongation one.
        R = P.transpose()

        return P, R

    def assemble_neumann_problem(self, p_f):
        """Assemble the Neumann problem to compute a conservative
        finescale pressure field.

        Parameters
        ----------
        p_f: The multiscale solution for the pressure field, i.e., the prolongated solution.

        Returns
        -------
        A_neu: The LHS of the Neumann problem as a block diagonal matrix.
        b_neu: The RHS of the Neumann problem.
        idx_map: The mapping from the local indices to the global sorted indices.
        """
        A_blocks = []
        b_neu = np.zeros(len(self.finescale_mesh.volumes))

        local_idx_map = np.zeros(len(self.finescale_mesh.volumes))

        all_fine_vols = self.finescale_mesh.volumes.all[:]
        all_coarse_vols = self.coarse_mesh.volumes.all[:]
        dirichlet_vols = np.nonzero(
            self.finescale_mesh.dirichlet[:].flatten())[0]
        coarse_volume_values = self.finescale_mesh.bg_volume[:].flatten()

        primal_centers_fine_idx = np.nonzero(
            self.finescale_mesh.primal_volume_center[:])[0]
        primal_centers_coarse_idx = self.finescale_mesh.bg_volume[primal_centers_fine_idx].flatten(
        )

        it = 0
        for cvol in all_coarse_vols:
            # Assemble the local problems by slicing the part of the main
            # problem concerning only the fine cells in the coarse cell.
            fine_idx = all_fine_vols[coarse_volume_values == cvol]
            A_local = self.A[fine_idx[:, None], fine_idx]
            b_local = self.q[fine_idx]

            # If the coarse cell does not contain any dirichlet BC, then
            # force the primal center to hold the value of the prolongated
            # pressure.
            if len(np.intersect1d(fine_idx, dirichlet_vols, assume_unique=True)) == 0:
                cvol_center = primal_centers_fine_idx[primal_centers_coarse_idx == cvol][0]
                cvol_center_local_idx = np.argwhere(
                    fine_idx == cvol_center)[0, 0]
                A_local[cvol_center_local_idx, :] = 0
                A_local[cvol_center_local_idx, cvol_center_local_idx] = 1
                b_local[cvol_center_local_idx] = p_f[cvol_center]

            # Add the local problem to the global block matrix.
            A_blocks.append(A_local)
            b_neu[it:(it + len(fine_idx))] = b_local[:]
            local_idx_map[it:(it + len(fine_idx))] = fine_idx[:]

            it += len(fine_idx)

        A_neu = block_diag(A_blocks, format="csr")
        idx_map = np.argsort(local_idx_map)

        return A_neu, b_neu, idx_map
