import numpy as np
import os
from time import perf_counter
from typing import Callable, Optional

from tools import tensor_distance, save_pkl

from seemps.state import Strategy, Simplification, Truncation
from seemps.truncate import simplify
from seemps.analysis.mesh import Mesh, RegularInterval, mps_to_mesh_matrix

from seemps.analysis.cross import CrossStrategyMaxvol, cross_maxvol, BlackBoxLoadMPS
from seemps.analysis.sampling import random_mps_indices, evaluate_mps

import seemps

seemps.tools.DEBUG = 2

DATA_PATH = "data/"


def data_cross_md(
    func: Callable,
    m: int,
    n: int,
    r: int,
    t: float,
    mps_order: str,
    name: Optional[str] = None,
):
    """
    Script to collect the data for the evaluation of the performance of
    the multivariate tensor cross-interpolation algorithm, based on one-site
    optimizations using the rectangular skeleton decomposition.

    Parameters
    ----------
    func : Callable
        The function to approximate.
    m : int
        The dimension of the function.
    n : int
        The number of qubits of the MPS approximation.
    r : int
        The maximum bond dimension allowed for the interpolation.
    t : float
        The tolerance parameter of the variational simplification routine.

     Notes
    -----
    - By default, the rank kick is set to (0, 1). However, for the step function it is
    set to (1, 1) as else it gets frequently stuck.
    """
    if name is not None:
        suffix = f"_m{m}_n{n}_r{r}_t{t}_order{mps_order}"
        filename = name + suffix + ".pkl"
        if os.path.exists(DATA_PATH + filename):
            print(f"{filename} already exists. Skipping computation.")
            return
        print(f"Computing {filename}")

    # Set the default parameters
    a, b = -1, 1
    strategy = Strategy(
        method=Truncation.RELATIVE_NORM_SQUARED_ERROR,
        simplify=Simplification.VARIATIONAL,
        tolerance=t**2,
        simplification_tolerance=t,
        max_sweeps=4,
    )
    rank_kick = (1, 1) if (name and "step" in name) else (0, 1)
    interval = RegularInterval(a, b, 2**n)
    mesh = Mesh([interval] * m)

    # Do 10 repetitions to account for the random initial state
    # For simplicity, we only do statistics for the unsimplified metrics
    repetitions = 10
    list_time = []
    list_maxbond = []
    list_error = []
    list_evals = []
    for i in range(repetitions):
        rng = np.random.default_rng(i)

        # Perform the MPS tensor cross-interpolation
        cross_strategy = CrossStrategyMaxvol(
            maxbond=r,
            tol_sampling=1e-10,
            rank_kick=rank_kick,
            rng=rng,
        )
        time_start = perf_counter()
        black_box = BlackBoxLoadMPS(func, mesh, mps_order=mps_order)
        cross_results = cross_maxvol(black_box, cross_strategy)
        time_stop = perf_counter()
        mps_cross = cross_results.mps

        # Evaluate the maxbond and sampled error of the MPS
        maxbond = max(mps_cross.bond_dimensions())
        mps_indices = random_mps_indices(mps_cross.physical_dimensions(), rng=rng)
        y_mps = evaluate_mps(mps_cross, mps_indices)
        T = mps_to_mesh_matrix([n] * m, mps_order=mps_order)
        mesh = Mesh([interval] * m)
        mesh_coordinates = mesh[mps_indices @ T]
        y_vec = func(mesh_coordinates.T)
        sampled_error = tensor_distance(y_vec, y_mps)

        list_time.append(time_stop - time_start)
        list_maxbond.append(maxbond)
        list_error.append(sampled_error)
        list_evals.append(cross_results.evals)

    # Evaluate the simplified MPS
    time_start_simp = perf_counter()
    mps_simp = simplify(mps_cross, strategy=strategy)
    time_stop_simp = perf_counter()
    time_simp = time_stop_simp - time_start_simp

    maxbond_simp = max(mps_simp.bond_dimensions())
    y_mps_simp = evaluate_mps(mps_simp, mps_indices)
    error_simp = tensor_distance(y_vec, y_mps_simp)

    # Save the data as a dictionary in a Pickle file
    data = {
        # Results before simplification
        "mean_error": np.mean(list_error),
        "std_error": np.std(list_error),
        "mean_maxbond": np.mean(list_maxbond),
        "std_maxbond": np.std(list_maxbond),
        "mean_time": np.mean(list_time),
        "std_time": np.std(list_time),
        "mean_evals": np.mean(list_evals),
        "std_evals": np.std(list_evals),
        # Results after simplification
        "error_simp": error_simp,
        "maxbond_simp": maxbond_simp,
        "time_simp": time_simp,
    }
    print(data)
    if name is not None:
        save_pkl(data, filename, path=DATA_PATH)


if __name__ == "__main__":
    # fmt: off
    # Set parameter ranges
    range_m = range(1, 10 + 1)

    # Set fixed parameters
    fixed_n = 20
    max_r = 100
    fixed_t = 1e-14

    # Function definitions
    func_product = lambda tensor: np.exp(-np.sum(tensor**2, axis=0))
    func_squeezed = lambda tensor: np.exp(-np.sum(tensor, axis=0) ** 2)

    x_c = (1 / 2) + (1 / 2**5)
    func_abs = lambda tensor: np.abs(np.sum(tensor, axis=0) - x_c)
    func_step = lambda tensor: np.heaviside(np.sum(tensor, axis=0) - x_c, 1/2)

    # Collect data
    # NOTE: Some parameter settings require a fine-tuning of the maximum allowed bond dimension
    # max-r to achieve a good convergence. These values are not included here.
    for m in range_m:
        data_cross_md(func_product, m, fixed_n, max_r, fixed_t, mps_order="A", name="cross_md_prod")
        data_cross_md(func_product, m, fixed_n, max_r, fixed_t, mps_order="B", name="cross_md_prod")
        data_cross_md(func_squeezed, m, fixed_n, max_r, fixed_t, mps_order="A", name="cross_md_sqz")
        data_cross_md(func_squeezed, m, fixed_n, max_r, fixed_t, mps_order="B", name="cross_md_sqz")
        data_cross_md(func_abs, m, fixed_n, max_r, fixed_t, mps_order="A", name="cross_md_abs")
        data_cross_md(func_abs, m, fixed_n, max_r, fixed_t, mps_order="B", name="cross_md_abs")
        data_cross_md(func_step, m, fixed_n, max_r, fixed_t, mps_order="A", name="cross_md_step")
        data_cross_md(func_step, m, fixed_n, max_r, fixed_t, mps_order="B", name="cross_md_step")
