import numpy as np
import os
from typing import Callable, Optional
from time import perf_counter

from tools import tensor_distance, save_pkl

from seemps.state import MPS, Strategy, Simplification, Truncation
from seemps.analysis.mesh import RegularInterval
from seemps.analysis.chebyshev import (
    interpolation_coefficients,
    cheb2mps,
)

import seemps.tools

seemps.tools.DEBUG = 0

DATA_PATH = "data/"


def data_chebyshev_1d(
    func: Callable, n: int, d: int, t: float, name: Optional[str] = None
):
    """
    Computes a dictionary containing data that evaluates the performance of
    the univariate MPS Chebyshev approximation algorithm, based on the
    interpolation on the Chebyshev-Gauss nodes.
    Saves it as a Pickle file with the given name and a suffix given by the parameters
    on the `/data` directory.

    Parameters
    ----------
    func : Callable
        The function to approximate.
    n : int
        The number of qubits of the MPS approximation.
    d : int
        The polynomial order of the Chebyshev approximation.
    t : float
        The tolerance parameter of the variational simplification routine.
    name : str
        The name of the simulation.
    """
    # Get the Pickle file name and check if it exists
    if name is not None:
        suffix = f"_n{n}_d{d}_t{t}"
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
    # Perform the MPS Chebyshev approximation
    # Based on the interpolation of the Chebyshev-Gauss (zero) nodes
    tick = perf_counter()
    domain = RegularInterval(a, b, 2**n)
    coefficients = interpolation_coefficients(func, d, a, b)
    mps_cheb = cheb2mps(coefficients, domain=domain, strategy=strategy)
    time = perf_counter() - tick

    # Perform the same approximation with a Schmidt decomposition
    tick = perf_counter()
    x_vec = domain.to_vector()
    y_vec = func(x_vec)
    mps_svd = MPS.from_vector(y_vec, [2] * n, strategy=strategy, normalize=False)
    time_svd = perf_counter() - tick

    # Evaluate the maximum bond dimension and error of the approximated MPS
    maxbond = max(mps_cheb.bond_dimensions())
    maxbond_svd = max(mps_svd.bond_dimensions())
    y_cheb = mps_cheb.to_vector()
    y_svd = mps_svd.to_vector()
    error = tensor_distance(y_vec, y_cheb)
    error_svd = tensor_distance(y_vec, y_svd)

    # Save the data as a dictionary in a Pickle file
    data = {
        "error": error,
        "maxbond": maxbond,
        "time": time,
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
    range_n = list(range(2, 25 + 1))
    range_d = list(range(2, 10)) + list(range(10, 100, 10)) + list(range(100, 1300 + 1, 100))
    range_t = [10 ** -(exp) for exp in range(1, 14 + 1)]

    # Set fixed parameters
    fixed_n = 25
    fixed_d = 1300
    fixed_t = 1e-14
    max_d_g = 50

    # Set functions
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
        data_chebyshev_1d(func_g, n, max_d_g, fixed_t, "chebyshev_1d_gaussian")
        data_chebyshev_1d(func_o, n, fixed_d, fixed_t, "chebyshev_1d_osc")
        data_chebyshev_1d(func_a, n, fixed_d, fixed_t, "chebyshev_1d_abs")
        data_chebyshev_1d(func_s, n, fixed_d, fixed_t, "chebyshev_1d_step")

    for d in range_d:
        if d <= max_d_g:
            data_chebyshev_1d(func_g, fixed_n, d, fixed_t, "chebyshev_1d_gaussian")
        data_chebyshev_1d(func_o, fixed_n, d, fixed_t, "chebyshev_1d_osc")
        data_chebyshev_1d(func_a, fixed_n, d, fixed_t, "chebyshev_1d_abs")
        data_chebyshev_1d(func_s, fixed_n, d, fixed_t, "chebyshev_1d_step")

    for t in range_t:
        data_chebyshev_1d(func_g, fixed_n, max_d_g, t, "chebyshev_1d_gaussian")
        data_chebyshev_1d(func_o, fixed_n, fixed_d, t, "chebyshev_1d_osc")
        data_chebyshev_1d(func_a, fixed_n, fixed_d, t, "chebyshev_1d_abs")
        data_chebyshev_1d(func_s, fixed_n, fixed_d, t, "chebyshev_1d_step")
