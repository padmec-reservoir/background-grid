import numpy as np
from scipy.sparse import lil_matrix


def assemble_tpfa_matrix(mesh):
    """
    Assemble the transmissibility matrix of the Two-Point Flux
    Approximation (TPFA) method.

    Parameters
    ----------
    mesh: FineScaleMesh
        An IMPRESS mesh object.

    Returns
    -------
    A_csr: csr_matrix
        The transmissibility matrix as a Scipy CSR matrix.
    q: array
        The source terms as a Numpy array.
    """

    # Cálculo das normais e faces internas.
    internal_faces = mesh.faces.internal[:]
    n_vols_pairs = len(internal_faces)

    internal_faces_nodes_1 = mesh.faces.connectivities[internal_faces][:, 0]
    internal_faces_nodes_2 = mesh.faces.connectivities[internal_faces][:, 1]
    V1 = mesh.nodes.coords[internal_faces_nodes_1]
    V2 = mesh.nodes.coords[internal_faces_nodes_2]
    internal_faces_centers = mesh.faces.center[internal_faces]
    N = np.cross(internal_faces_centers - V1, internal_faces_centers - V2)

    # Recuperando volumes que compartilham faces internas.
    internal_volumes = mesh.faces.bridge_adjacencies(internal_faces, 2, 3)

    # Cálculo da distância entre o centroide de um volume e o centroide de uma face.
    a_vol_center = mesh.volumes.center[internal_volumes[0, 0]]
    a_face_center = internal_faces_centers[0]
    h = np.linalg.norm(a_vol_center - a_face_center)

    # Cálculo das permeabilidades equivalentes.
    K_all = mesh.permeability[internal_volumes.flatten()].reshape((n_vols_pairs * 2, 3, 3))
    N_dup = np.hstack((N, N)).reshape((len(N) * 2, 3))
    K_eq_partial = np.einsum("ij,ikj->ik", N_dup, K_all)
    K_eq_all_part = np.einsum("ij,ij->i", K_eq_partial, N_dup) / np.linalg.norm(N_dup, axis=1)
    K_eq_all = K_eq_all_part.reshape((2, n_vols_pairs))

    # Cáculo das transmissibilidades por face.
    K_eq_sum = K_eq_all[0, :] + K_eq_all[1, :]
    K_eq_prod = K_eq_all[0, :] * K_eq_all[1, :]
    faces_trans = (K_eq_prod / K_eq_sum) / h

    # Montagem da matriz de transmissibilidades.
    N_vols = len(mesh.volumes)
    A = lil_matrix((N_vols, N_vols))

    dirichlet_values = mesh.dirichlet[:].flatten()
    non_null_dirichlet_mask = dirichlet_values != np.inf
    non_null_dirichlet_values = dirichlet_values[non_null_dirichlet_mask]
    non_null_dirichlet_volumes_idx = mesh.volumes.all[non_null_dirichlet_mask]

    dirichlet_mask = ~np.isin(internal_volumes[:, 0], non_null_dirichlet_volumes_idx)
    internal_volumes_filt = internal_volumes[dirichlet_mask]
    faces_trans_filt = faces_trans[dirichlet_mask]

    A[internal_volumes_filt[:, 0], internal_volumes_filt[:, 1]] = -faces_trans_filt

    dirichlet_mask = ~np.isin(internal_volumes[:, 1], non_null_dirichlet_volumes_idx)
    internal_volumes_filt = internal_volumes[dirichlet_mask]
    faces_trans_filt = faces_trans[dirichlet_mask]

    A[internal_volumes_filt[:, 1], internal_volumes_filt[:, 0]] = -faces_trans_filt

    A_diag = -A.sum(axis=1)
    A.setdiag(A_diag)

    # Atribuição das condições de contorno de Neumann.
    q = np.zeros(N_vols)

    neumann_values = mesh.neumann[:]
    non_null_neuman_values_mask = (neumann_values != np.inf).flatten()
    non_null_neuman_values = neumann_values[non_null_neuman_values_mask].flatten()

    if len(non_null_neuman_values) > 0:
        non_null_neumann_faces_idx = mesh.faces.all[non_null_neuman_values_mask]
        neumann_volumes = mesh.faces.bridge_adjacencies(non_null_neumann_faces_idx, 2, 3).flatten()
        q[neumann_volumes] += non_null_neuman_values

    # Atribuição das condições de contorno de Dirichlet.
    if len(non_null_dirichlet_values) > 0:
        # A[non_null_dirichlet_volumes_idx] = 0.0
        A[non_null_dirichlet_volumes_idx, non_null_dirichlet_volumes_idx] = 1.0
        q[non_null_dirichlet_volumes_idx] = non_null_dirichlet_values

    A_csr = A.tocsr()

    return A_csr, q
