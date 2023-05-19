import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, diags, block_diag
from itertools import chain


class MsCVOperator(object):
    def __init__(self, finescale_mesh, coarse_mesh, support_regions,
                 support_boundaries, A, q, mpfad_weights):
        self.finescale_mesh = finescale_mesh
        self.coarse_mesh = coarse_mesh
        self.support_regions = support_regions
        self.support_boundaries = support_boundaries
        self.A = A
        self.q = q

        # MPFA-D parameters.
        in_faces = self.finescale_mesh.faces.internal[:]
        in_faces_map = -np.ones(len(self.finescale_mesh.faces), dtype=int)
        in_faces_map[in_faces] = np.arange(in_faces.shape[0])
        self.in_faces_map = in_faces_map

        self.div = None
        self.mpfad_weights = mpfad_weights
        self.in_vols_pairs = None
        self.h_L = None
        self.h_R = None
        self.Ns = None
        self.Ns_norm = None
        self.Kn_L = None
        self.Kn_R = None
        self.D_JI = None
        self.D_JK = None

    def _init_mesh_data(self):
        all_volumes = self.finescale_mesh.core.all_volumes[:]
        all_tags = self.finescale_mesh.core.mb.tag_get_tags_on_entity(
            all_volumes[0])

        bg_volume_tag = [
            tag for tag in all_tags if tag.get_name() == "bg_volume"].pop()

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

        dirichlet_faces_flags = self.finescale_mesh.dirichlet_faces[:].flatten(
        )
        dirichlet_faces = self.finescale_mesh.faces.all[dirichlet_faces_flags == 1]
        dirichlet_idx = self.finescale_mesh.faces.bridge_adjacencies(
            dirichlet_faces, 2, 3).flatten()
        dirichlet_primal_idx = self.finescale_mesh.bg_volume[dirichlet_idx].flatten(
        )

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
        all_vols_in_sup_boundary = list(
            chain.from_iterable(self.support_boundaries.values()))

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
        vols_in_a_single_sr_primal_ids = self.finescale_mesh.bg_volume[vols_in_a_single_sr].flatten(
        )
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
            d_H = (
                d_hat_in_H - P_H.multiply(Sd_H[:, None])).multiply((1 + Sd_H[:, None]) ** -1)

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
        self._set_neumann_problem_params()

        A_blocks = []
        b_neu = np.zeros(len(self.finescale_mesh.volumes))

        local_idx_map = np.zeros(len(self.finescale_mesh.volumes))

        all_fine_vols = self.finescale_mesh.volumes.all[:]
        all_fine_faces = self.finescale_mesh.faces.all[:]
        all_coarse_vols = self.coarse_mesh.volumes.all[:]
        dirichlet_vols = np.nonzero(
            self.finescale_mesh.dirichlet[:].flatten())[0]
        coarse_volume_values = self.finescale_mesh.bg_volume[:].flatten()

        primal_centers_fine_idx = np.nonzero(
            self.finescale_mesh.primal_volume_center[:])[0]
        primal_centers_coarse_idx = self.finescale_mesh.bg_volume[primal_centers_fine_idx].flatten(
        )

        in_faces = self.finescale_mesh.faces.internal[:]
        bfaces = self.finescale_mesh.faces.boundary[:]
        fine_vols_faces = self.finescale_mesh.volumes.adjacencies[:]
        fine_faces_nodes = self.finescale_mesh.faces.bridge_adjacencies(
            all_fine_faces, 0, 0)

        primal_faces_flag = self.finescale_mesh.primal_face[:].flatten()
        primal_faces = all_fine_faces[primal_faces_flag == 1]
        in_primal_faces = np.intersect1d(primal_faces, in_faces)

        in_faces_flux = self._compute_ms_flux(p_f)
        bfaces_flux = self._compute_ms_boundary_flux(p_f)
        F = np.zeros(len(self.finescale_mesh.faces))
        F[in_faces] = in_faces_flux[:]
        F[bfaces] = bfaces_flux[:]

        it = 0
        for cvol in all_coarse_vols:
            # Assemble the local problems by slicing the part of the main
            # problem concerning only the fine cells in the coarse cell.
            fine_idx = all_fine_vols[coarse_volume_values == cvol]
            A_local = self.A[fine_idx[:, None], fine_idx]
            b_local = self.q[fine_idx]

            # Assign the neumann BC for the internal primal faces.
            cvol_faces = fine_vols_faces[fine_idx].flatten()
            cvol_bfaces = np.intersect1d(cvol_faces, primal_faces)
            cvol_internal_bfaces = np.intersect1d(cvol_faces, in_primal_faces)
            b_local += (self.div[:, cvol_bfaces] @ F[cvol_bfaces])[fine_idx]

            # Handle internal faces with nodes on the global boundary.
            cvol_bnodes = fine_faces_nodes[cvol_internal_bfaces]
            I, J, K = cvol_bnodes[:, 0], cvol_bnodes[:, 1], cvol_bnodes[:, 2]
            q_local = self._handle_boundary_nodes_neu_problem(
                p_f, cvol_internal_bfaces, I, J, K)
            b_local += q_local[fine_idx]

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

    def compute_conservative_flux(self, p_ms, p_f):
        """Computes a conservative flux field from the pressure obtained via
        solution of the Neumann problem and the prolongated pressure.

        Parameters
        ----------
        p_ms (numpy.ndarray): The pressure field obtained by solving the Neumann problem.
        p_f (numpy.ndarray): The pressure field obtained by prolongating the coarse scale solution.
        """
        in_faces = self.finescale_mesh.faces.internal[:]
        bfaces = self.finescale_mesh.faces.boundary[:]

        # Get the internal primal faces.
        primal_faces_flag = self.finescale_mesh.primal_face[:].flatten()
        primal_faces = self.finescale_mesh.faces.all[primal_faces_flag == 1]
        primal_in_faces = np.intersect1d(primal_faces, in_faces)

        # Compute the flux on the internal faces using both
        # the prolongated solution and the conservative one.
        F_b = self._compute_ms_boundary_flux(p_f)
        F_in_non_conserv = self._compute_ms_flux(p_f)
        F_in_conserv = self._compute_ms_flux(p_ms)

        F = np.zeros(len(self.finescale_mesh.faces))
        F[bfaces] = F_b[:]
        F[in_faces] = F_in_conserv[:]
        F[primal_in_faces] = F_in_non_conserv[self.in_faces_map[primal_in_faces]]

        return F

    def _set_neumann_problem_params(self):
        self._set_internal_vols_pairs()
        self._set_normal_vectors()
        self._set_normal_distances()
        self._set_normal_permeabilities()
        self._set_cdt_coefficients()
        self._set_div_operator()

    def _compute_ms_flux(self, p_ms):
        """Computes the MPFA-D flux through the internal faces.

        Parameters
        ----------
        p_ms: The pressure field computed using MsCV.

        Returns
        -------
        F_in: The flux through the internal faces.
        """
        L, R = self.in_vols_pairs[:, 0], self.in_vols_pairs[:, 1]
        uL, uR = p_ms[L], p_ms[R]

        in_faces = self.finescale_mesh.faces.internal[:]
        in_faces_nodes = self.finescale_mesh.faces.bridge_adjacencies(
            in_faces, 0, 0)
        I, J, K = (
            in_faces_nodes[:, 0],
            in_faces_nodes[:, 1],
            in_faces_nodes[:, 2])

        Kn_prod = self.Kn_L[:] * self.Kn_R[:]
        Keq = Kn_prod / ((self.Kn_L[:] * self.h_R[:]) +
                         (self.Kn_R[:] * self.h_L[:]))
        Keq_N = self.Ns_norm[:] * Keq

        djk, dji = self.D_JK[:], self.D_JI[:]

        wI, wJ, wK = (self.mpfad_weights[I, :],
                      self.mpfad_weights[J, :],
                      self.mpfad_weights[K, :])
        uI, uJ, uK = wI @ p_ms, wJ @ p_ms, wK @ p_ms

        F_in = -Keq_N * ((uR - uL) - 0.5 * djk *
                         (uJ - uI) + 0.5 * dji * (uJ - uK))

        return F_in

    def _compute_ms_boundary_flux(self, p_ms):
        """Computes the flux on the boundary faces of the finescale mesh.

        Parameters
        ----------
        p_ms: The multiscale pressure field.

        Returns
        -------
        F_D: The flux on the boundary faces of the finescale mesh.
        """
        bfaces = self.finescale_mesh.faces.boundary[:]
        bfaces_dirichlet_values = self.finescale_mesh.dirichlet_faces[bfaces].flatten(
        )
        dirichlet_faces = bfaces[bfaces_dirichlet_values == 1]

        dirichlet_nodes = self.finescale_mesh.faces.bridge_adjacencies(
            dirichlet_faces, 0, 0)
        dirichlet_volumes = self.finescale_mesh.faces.bridge_adjacencies(
            dirichlet_faces, 2, 3).flatten()

        L = self.finescale_mesh.volumes.center[dirichlet_volumes]
        I_idx, J_idx, K_idx = (
            dirichlet_nodes[:, 0],
            dirichlet_nodes[:, 1],
            dirichlet_nodes[:, 2])
        I, J, K = (
            self.finescale_mesh.nodes.coords[I_idx],
            self.finescale_mesh.nodes.coords[J_idx],
            self.finescale_mesh.nodes.coords[K_idx])

        N = 0.5 * np.cross(I - J, K - J)

        LJ = J - L
        N_test = np.sign(np.einsum("ij,ij->i", LJ, N))
        I[N_test < 0], K[N_test < 0] = K[N_test < 0], I[N_test < 0]
        N = 0.5 * np.cross(I - J, K - J)

        N_norm = np.linalg.norm(N, axis=1)

        tau_JK = np.cross(N, K - J)
        tau_JI = np.cross(N, I - J)

        h_L = np.abs(np.einsum("ij,ij->i", N, LJ) / N_norm)

        K_all = self.finescale_mesh.permeability[dirichlet_volumes].reshape(
            (len(dirichlet_volumes), 3, 3))

        Kn_L_partial = np.einsum("ij,ikj->ik", N, K_all)
        Kn_L = np.einsum("ij,ij->i", Kn_L_partial, N) / (N_norm ** 2)

        Kt_JK = np.einsum("ij,ij->i", Kn_L_partial, tau_JK) / (N_norm ** 2)

        Kt_JI = np.einsum("ij,ij->i", Kn_L_partial, tau_JI) / (N_norm ** 2)

        D_JI = -(np.einsum("ij,ij->i", tau_JK, LJ)
                 * Kn_L) / (2 * N_norm * h_L) + Kt_JK / 2
        D_JK = -(np.einsum("ij,ij->i", tau_JI, LJ)
                 * Kn_L) / (2 * N_norm * h_L) + Kt_JI / 2

        gD = self.finescale_mesh.dirichlet_nodes[dirichlet_nodes.flatten()].reshape(
            dirichlet_nodes.shape[0], 3)
        gD_I, gD_J, gD_K = gD[:, 0], gD[:, 1], gD[:, 2]
        gD_I[N_test < 0], gD_K[N_test < 0] = gD_K[N_test < 0], gD_I[N_test < 0]

        F_D = -((Kn_L * N_norm / h_L) *
                (p_ms[dirichlet_volumes] - gD_J) + D_JI * (gD_J - gD_I) + D_JK * (gD_K - gD_J))

        return F_D

    def _handle_boundary_nodes_neu_problem(self, p_ms, faces, I, J, K):
        in_faces_idx = self.in_faces_map[faces]

        I_left_vol, I_right_vol = (
            self.in_vols_pairs[in_faces_idx, 0],
            self.in_vols_pairs[in_faces_idx, 1])
        J_left_vol, J_right_vol = (
            self.in_vols_pairs[in_faces_idx, 0],
            self.in_vols_pairs[in_faces_idx, 1])
        K_left_vol, K_right_vol = (
            self.in_vols_pairs[in_faces_idx, 0],
            self.in_vols_pairs[in_faces_idx, 1])

        wI, wJ, wK = (self.mpfad_weights[I, :],
                      self.mpfad_weights[J, :],
                      self.mpfad_weights[K, :])
        uI, uJ, uK = wI @ p_ms, wJ @ p_ms, wK @ p_ms

        Kn_prod = self.Kn_L[in_faces_idx] * self.Kn_R[in_faces_idx]
        Keq = Kn_prod / ((self.Kn_L[in_faces_idx] * self.h_R[in_faces_idx]) +
                         (self.Kn_R[in_faces_idx] * self.h_L[in_faces_idx]))
        Keq_N = self.Ns_norm[in_faces_idx] * Keq

        I_term = 0.5 * Keq_N * self.D_JK[in_faces_idx] * uI
        J_term = 0.5 * Keq_N * \
            (self.D_JI[in_faces_idx] - self.D_JK[in_faces_idx]) * uJ
        K_term = -0.5 * Keq_N * self.D_JI[in_faces_idx] * uK

        q = np.zeros(len(self.finescale_mesh.volumes))

        np.add.at(q, I_left_vol, I_term)
        np.add.at(q, I_right_vol, -I_term)

        np.add.at(q, J_left_vol, J_term)
        np.add.at(q, J_right_vol, -J_term)

        np.add.at(q, K_left_vol, K_term)
        np.add.at(q, K_right_vol, -K_term)

        return q

    def _set_internal_vols_pairs(self):
        """Set the pairs of volumes sharing an internal face in the 
        attribute `in_vols_pairs`.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        internal_faces = self.finescale_mesh.faces.internal[:]
        self.in_vols_pairs = self.finescale_mesh.faces.bridge_adjacencies(
            internal_faces, 2, 3)

    def _set_normal_distances(self):
        """Compute the distances from the center of the internal faces to their
        adjacent volumes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        internal_faces = self.finescale_mesh.faces.internal[:]

        L = self.finescale_mesh.volumes.center[self.in_vols_pairs[:, 0]]
        R = self.finescale_mesh.volumes.center[self.in_vols_pairs[:, 1]]

        in_faces_nodes = self.finescale_mesh.faces.bridge_adjacencies(
            internal_faces, 0, 0)
        J_idx = in_faces_nodes[:, 1]
        J = self.finescale_mesh.nodes.coords[J_idx]

        LJ = J - L
        LR = J - R

        self.h_L = np.abs(np.einsum("ij,ij->i", self.Ns, LJ) / self.Ns_norm)
        self.h_R = np.abs(np.einsum("ij,ij->i", self.Ns, LR) / self.Ns_norm)

    def _set_normal_vectors(self):
        """Set the attribute `Ns` which stores the normal vectors 
        to the internal faces.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Retrieve the internal faces.
        internal_faces = self.finescale_mesh.faces.internal[:]

        # Retrieve the points that form the components of the normal vectors.
        internal_faces_nodes = self.finescale_mesh.faces.bridge_adjacencies(
            internal_faces,
            0, 0)
        I_idx = internal_faces_nodes[:, 0]
        J_idx = internal_faces_nodes[:, 1]
        K_idx = internal_faces_nodes[:, 2]

        I = self.finescale_mesh.nodes.coords[I_idx]
        J = self.finescale_mesh.nodes.coords[J_idx]
        K = self.finescale_mesh.nodes.coords[K_idx]

        n_vols_pairs = len(internal_faces)
        internal_volumes_centers_flat = self.finescale_mesh.volumes.center[self.in_vols_pairs.flatten(
        )]
        internal_volumes_centers = internal_volumes_centers_flat.reshape((
            n_vols_pairs,
            2, 3))

        LJ = J - internal_volumes_centers[:, 0]

        # Set the normal vectors.
        self.Ns = 0.5 * np.cross(I - J, K - J)
        self.Ns_norm = np.linalg.norm(self.Ns, axis=1)

        N_sign = np.sign(np.einsum("ij,ij->i", LJ, self.Ns))
        (self.in_vols_pairs[N_sign < 0, 0],
         self.in_vols_pairs[N_sign < 0, 1]) = (self.in_vols_pairs[N_sign < 0, 1],
                                               self.in_vols_pairs[N_sign < 0, 0])

    def _set_normal_permeabilities(self):
        """Compute the normal projections of the permeability tensors.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        n_vols_pairs = len(self.finescale_mesh.faces.internal)

        lvols = self.in_vols_pairs[:, 0]
        rvols = self.in_vols_pairs[:, 1]

        KL = self.finescale_mesh.permeability[lvols].reshape(
            (n_vols_pairs, 3, 3))
        KR = self.finescale_mesh.permeability[rvols].reshape(
            (n_vols_pairs, 3, 3))

        KnL_pre = np.einsum("ij,ikj->ik", self.Ns, KL)
        KnR_pre = np.einsum("ij,ikj->ik", self.Ns, KR)

        KnL = np.einsum("ij,ij->i", KnL_pre, self.Ns) / self.Ns_norm ** 2
        KnR = np.einsum("ij,ij->i", KnR_pre, self.Ns) / self.Ns_norm ** 2

        self.Kn_L = KnL[:]
        self.Kn_R = KnR[:]

    def _compute_tangent_permeabilities(self, tau_ij):
        """Computes the tangent projection of the permeability tensors
        given vectors `tau_ij`.

        Parameters
        ----------
        tau_ij: A N x 3 numpy array representing stacked vectors.

        Returns
        -------
        A tuple of arrays containing the projections to the left and
        to the right of the internal faces.
        """
        n_vols_pairs = len(self.finescale_mesh.faces.internal)

        n_vols_pairs = len(self.finescale_mesh.faces.internal)

        lvols = self.in_vols_pairs[:, 0]
        rvols = self.in_vols_pairs[:, 1]

        KL = self.finescale_mesh.permeability[lvols].reshape(
            (n_vols_pairs, 3, 3))
        KR = self.finescale_mesh.permeability[rvols].reshape(
            (n_vols_pairs, 3, 3))

        Kt_ij_L_pre = np.einsum("ij,ikj->ik", self.Ns, KL)
        Kt_ij_R_pre = np.einsum("ij,ikj->ik", self.Ns, KR)

        Kt_ij_L = np.einsum("ij,ij->i", Kt_ij_L_pre,
                            tau_ij) / self.Ns_norm ** 2
        Kt_ij_R = np.einsum("ij,ij->i", Kt_ij_R_pre,
                            tau_ij) / self.Ns_norm ** 2

        return Kt_ij_L, Kt_ij_R

    def _set_cdt_coefficients(self):
        """Compute the cross coefficients terms of the MPFA-D scheme.

        Parameters
        ----------
        None

        Returns
        -------
        A tuple of numpy arrays containing the terms D_JK and D_JI.
        """
        n_vols_pairs = len(self.finescale_mesh.faces.internal)

        in_vols_pairs_flat = self.in_vols_pairs.flatten()
        in_vols_centers_flat = self.finescale_mesh.volumes.center[in_vols_pairs_flat]
        in_vols_centers = in_vols_centers_flat.reshape((n_vols_pairs, 2, 3))

        LR = in_vols_centers[:, 1, :] - in_vols_centers[:, 0, :]

        internal_faces = self.finescale_mesh.faces.internal[:]

        internal_faces_nodes = self.finescale_mesh.faces.bridge_adjacencies(
            internal_faces,
            0, 0)
        I_idx = internal_faces_nodes[:, 0]
        J_idx = internal_faces_nodes[:, 1]
        K_idx = internal_faces_nodes[:, 2]

        I = self.finescale_mesh.nodes.coords[I_idx]
        J = self.finescale_mesh.nodes.coords[J_idx]
        K = self.finescale_mesh.nodes.coords[K_idx]

        tau_JK = np.cross(self.Ns, K - J)
        tau_JI = np.cross(self.Ns, I - J)

        Kt_JK_L, Kt_JK_R = self._compute_tangent_permeabilities(tau_JK)
        Kt_JI_L, Kt_JI_R = self._compute_tangent_permeabilities(tau_JI)

        A1_JK = np.einsum("ij,ij->i", tau_JK, LR) / (self.Ns_norm ** 2)
        A2_JK = (self.h_L * (Kt_JK_L / self.Kn_L) + self.h_R *
                 (Kt_JK_R / self.Kn_R)) / self.Ns_norm
        D_JK = A1_JK - A2_JK

        A1_JI = np.einsum("ij,ij->i", tau_JI, LR) / (self.Ns_norm ** 2)
        A2_JI = (self.h_L * (Kt_JI_L / self.Kn_L) + self.h_R *
                 (Kt_JI_R / self.Kn_R)) / self.Ns_norm
        D_JI = A1_JI - A2_JI

        self.D_JK = D_JK
        self.D_JI = D_JI

    def _set_div_operator(self):
        in_faces = self.finescale_mesh.faces.internal[:]
        bfaces = self.finescale_mesh.faces.boundary[:]
        bvols = self.finescale_mesh.faces.bridge_adjacencies(
            bfaces, 2, 3).flatten()

        d = -np.ones(2 * in_faces.shape[0] + bfaces.shape[0])
        d[in_faces.shape[0]:] *= -1

        in_vols_pairs_flat = self.in_vols_pairs.flatten(order="F")
        vols_idx = np.hstack((in_vols_pairs_flat, bvols))
        faces_idx = np.hstack((in_faces, in_faces, bfaces))

        self.div = csr_matrix((d, (vols_idx, faces_idx)), shape=(
            len(self.finescale_mesh.volumes), len(self.finescale_mesh.faces)))
