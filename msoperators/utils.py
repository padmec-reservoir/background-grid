import numpy as np


def compute_ms_l2_error(u_ref, u_ms):
    return np.sum(np.abs(u_ref - u_ms) ** 2) / np.sum((np.abs(u_ref)) ** 2)


def compute_ms_inf_error(u_ref, u_ms):
    return np.max(np.abs(u_ref - u_ms))
