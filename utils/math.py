"""
math script
"""

import numpy as np


def expand_dims(array: np.array, expand_len: int, dim: int) -> np.array:
    return np.expand_dims(array, dim).repeat(expand_len, axis=dim)

def warp_to_pi(yaw: float) -> float:
    """warp yaw to [-pi, pi)

    Args:
        yaw (float): raw angle

    Returns:
        float: raw angle after warping
    """
    while yaw >= np.pi:
        yaw -= 2 * np.pi
    while yaw < -np.pi:
        yaw += 2 * np.pi
    return yaw

def ray_Dis(tp_thre_sq: float, cov_mat: np.mat, method: str = 'max') -> float:
    """Rayleigh cumulative distribution

    Args:
        tp_thre_sq (float): threshold square for determining tp
        cova_mat (np.mat): covariance matrix in kalman filter
        method (str, optional): fused x-y vars method. Defaults to 'max'.
        
    Returns:
        float: cumulative distribution
    """
    
    x_var, y_var = cov_mat[0, 0], cov_mat[1, 1]
    
    if method == 'max':
        var = max(x_var, y_var)
    elif method == 'min':
        var = min(x_var, y_var)
    elif method == 'avg':
        var = (x_var + y_var) / 2
    else:
        raise Exception('unsupport method')

    return 1 - np.exp(-tp_thre_sq / (2 * var))
    
    
        
