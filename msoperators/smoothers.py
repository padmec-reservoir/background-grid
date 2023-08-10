import numpy as np
from scipy.sparse.linalg import spilu, inv, LinearOperator, bicgstab

class AMSUIterativeSmoother(object):
    def __init__(self, A, q, p_ms, P, R) -> None:
        self.A = A
        self.q = q
        self.p_ms = p_ms
        self.P = P
        self.R = R
    
    def apply(self, tol):
        p_curr = self.p_ms.copy()
        r_n = self.q - self.A @ p_curr
        err = np.inf

        A_ilu = spilu(self.A, drop_tol=1e-5)
        A_ilu_op = LinearOperator(self.A.shape, A_ilu.solve)

        A_c = self.R @ self.A @ self.P
        A_c_inv = inv(A_c)
        B = self.P @ A_c_inv @ self.R

        while err > tol:
            dp_n_1 = B @ r_n
            r_n_1 = r_n - self.A @ dp_n_1
            dp_n_2, _ = bicgstab(self.A, r_n_1, M=A_ilu_op)
            p_next = dp_n_1 + dp_n_2
            r_n = self.q - self.A @ p_next
            p_curr[:] = p_next[:]
            err = np.linalg.norm(p_curr)

        return p_curr
