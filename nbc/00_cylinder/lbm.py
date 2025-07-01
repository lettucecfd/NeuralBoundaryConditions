import numpy as np

__all__ = [
    "D2Q9", "density", "momentum", "velocity"
]

class D2Q9:
    def __init__(self):
        self.e = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
        self.w = [4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4
        self.opposite = [0, 3, 4, 1, 2, 7, 8, 5, 6]

def density(distribution):
    """density"""
    return np.sum(distribution, axis=0)[None, ...]

def momentum(distribution, stencil):
    """momentum"""
    return np.einsum("qd,q...->d...",np.array(stencil.e), distribution)

def velocity(distribution, stencil):
    """velocity; the `acceleration` is used to compute the correct velocity
    in the presence of a forcing scheme."""
    rho = density(distribution)
    return momentum(distribution, stencil) / rho