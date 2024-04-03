import numpy as np
import os
from time import perf_counter
from typing import Callable, Optional

from tools import tensor_distance, save_pkl

from seemps.state import MPS, Strategy, Simplification, Truncation
from seemps.truncate import simplify
from seemps.analysis.mesh import Mesh, RegularInterval
from seemps.analysis.cross import (
    BlackBoxLoadMPS,
    CrossStrategyMaxvol,
    cross_maxvol,
)
import seemps.tools

seemps.tools.DEBUG = 0

DATA_PATH = "data/"


def data_cross_1d(func: Callable, n: int, r: int, t: float, name: Optional[str] = None):
    """
    Script to collect the data for the evaluation of the performance of
    the univariate tensor cross-interpolation algorithm, based on one-site
    optimizations using the rectangular skeleton decomposition.

    Parameters
    ----------
    func : Callable
        The function to approximate.
    n : int
        The number of qubits of the MPS approximation.
    d : int
        The maximum bond dimension allowed for the interpolation.
    t : float
        The tolerance parameter of the variational simplification routine.

    Notes
    -----
    - By default, the rank kick is set to (0, 1). However, for the step function it is
    set to (1, 1) as else it gets frequently stuck.
    """
    if name is not None:
        suffix = f"_n{n}_r{r}_t{t}"
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
    mesh = Mesh([interval])
    x_vec = interval.to_vector()
    y_vec = func(x_vec)

    # Do 10 repetitions to account for the random initial state
    repetitions = 10
    list_time = []
    list_maxbond = []
    list_error = []
    list_evals = []
    for i in range(repetitions):
        rng = np.random.default_rng(seed=i)
        cross_strategy = CrossStrategyMaxvol(
            maxbond=r,
            tol_sampling=t,
            rank_kick=rank_kick,
            rng=rng,
        )
        # Perform maxvol tensor cross-interpolation
        time_start = perf_counter()
        black_box = BlackBoxLoadMPS(func, mesh)
        cross_results = cross_maxvol(black_box, cross_strategy=cross_strategy)
        time_stop = perf_counter()
        mps_cross = cross_results.mps

        # Evaluate the unsimplified MPS
        maxbond = max(mps_cross.bond_dimensions())
        error = tensor_distance(y_vec, mps_cross.to_vector())
        list_time.append(time_stop - time_start)
        list_maxbond.append(maxbond)
        list_error.append(error)
        list_evals.append(cross_results.evals)

    # Evaluate the simplified MPS
    time_start = perf_counter()
    mps_simplified = simplify(mps_cross, strategy=strategy)
    time_stop = perf_counter()
    time_simp = time_stop - time_start
    maxbond_simp = max(mps_simplified.bond_dimensions())
    error_simp = tensor_distance(y_vec, mps_simplified.to_vector())

    # Perform the same approximation with a Schmidt decomposition and evaluate it
    tick = perf_counter()
    x_vec = interval.to_vector()
    y_vec = func(x_vec)
    mps_svd = MPS.from_vector(y_vec, [2] * n, strategy=strategy, normalize=False)
    time_svd = perf_counter() - tick
    maxbond_svd = max(mps_svd.bond_dimensions())
    error_svd = tensor_distance(y_vec, mps_svd.to_vector())

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
        # SVD results
        "error_svd": error_svd,
        "maxbond_svd": maxbond_svd,
        "time_svd": time_svd,
    }
    print(data)
    if name is not None:
        save_pkl(data, filename, path=DATA_PATH)


if __name__ == "__main__":
    # fmt: off
    # Set parameter ranges
    range_n = range(2, 25 + 1)
    range_r = range(2, 30 + 1, 2)
    range_t = [10 ** -(exp) for exp in range(1, 14 + 1)]

    # Set fixed parameters
    fixed_n = 25
    fixed_t = 1e-14
    # Hardcode the threshold bond dimension to ensure optimal convergence
    fixed_r_g = 12
    fixed_r_o = 24
    fixed_r_a = 6
    fixed_r_s = 6

    # Define functions
    # 1. Gaussian
    σ, μ = 1 / 3, 0
    func_g = lambda x: (1 / (σ * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - μ) / σ) ** 2)
    # 2. Oscillating function
    ε = 1 / 100
    func_o = lambda x: np.cos(1 / (x**2 + ε))
    # 3, 4. Absolute value and step functions
    x_c = (1 / 2) + (1 / 2**5)
    func_a = lambda x: np.abs(x - x_c)
    func_s = lambda x: np.heaviside(x - x_c, 1 / 2)

    # Collect data
    for n in range_n:
        data_cross_1d(func_g, n, fixed_r_g, fixed_t, "cross_1d_gaussian")
        data_cross_1d(func_o, n, fixed_r_o, fixed_t, "cross_1d_osc")
        data_cross_1d(func_a, n, fixed_r_a, fixed_t, "cross_1d_abs")
        data_cross_1d(func_s, n, fixed_r_s, fixed_t, "cross_1d_step")

    for r in range_r:
        data_cross_1d(func_g, fixed_n, r, fixed_t, "cross_1d_gaussian")
        data_cross_1d(func_o, fixed_n, r, fixed_t, "cross_1d_osc")
        data_cross_1d(func_a, fixed_n, r, fixed_t, "cross_1d_abs")
        data_cross_1d(func_s, fixed_n, r, fixed_t, "cross_1d_step")

    for t in range_t:
        data_cross_1d(func_g, fixed_n, fixed_r_g, t, "cross_1d_gaussian")
        data_cross_1d(func_o, fixed_n, fixed_r_o, t, "cross_1d_osc")
        data_cross_1d(func_a, fixed_n, fixed_r_a, t, "cross_1d_abs")
        data_cross_1d(func_s, fixed_n, fixed_r_s, t, "cross_1d_step")
