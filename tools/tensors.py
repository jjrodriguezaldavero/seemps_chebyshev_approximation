import numpy as np
from typing import Callable

from seemps.state import MPS
from seemps.analysis.mesh import Mesh, mps_to_mesh_matrix
from seemps.analysis.sampling import evaluate_mps, random_mps_indices


def tensor_distance(A: np.ndarray, B: np.ndarray, norm_error: float = np.inf):
    if A.shape != B.shape:
        raise ValueError("The tensors are of different shape")
    error = np.linalg.norm(A - B, ord=norm_error)
    prefactor = np.prod(A.shape) ** (1 / norm_error)
    return error / prefactor


def sampled_error(
    func_tensor: Callable,
    mps: MPS,
    mesh: Mesh,
    mps_order="A",
    num_iters=10,
    num_indices=1000,
):
    errors = []
    for i in range(num_iters):
        rng = np.random.default_rng(i)
        mps_indices = random_mps_indices(
            mps.physical_dimensions(), rng=rng, num_indices=num_indices
        )
        y_mps = evaluate_mps(mps, mps_indices)
        sites_per_dimension = [
            int(np.log2(interval.size)) for interval in mesh.intervals
        ]
        M = mps_to_mesh_matrix(sites_per_dimension, mps_order=mps_order)
        mesh_coordinates = mesh[mps_indices @ M]
        y_vec = func_tensor(mesh_coordinates.T)
        errors.append(tensor_distance(y_vec, y_mps))
    return np.mean(errors), np.std(errors)
