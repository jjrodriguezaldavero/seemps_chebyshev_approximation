import numpy as np
import os
from time import perf_counter
from typing import Callable, Optional

from tools import tensor_distance, save_pkl

from seemps.state import Strategy, Simplification, Truncation
from seemps.analysis.mesh import RegularInterval
from seemps.analysis.lagrange import (
    lagrange_basic,
    lagrange_rank_revealing,
    lagrange_local_rank_revealing,
)

DATA_PATH = "data/"


def data_lagrange(func: Callable, n: int, d: int, t: int, name: Optional[str] = None):
    """
    Script to collect the data for the evaluation of the performance of
    the univariate MPS Lagrange interpolation algorithm, that interpolates
    the Chebyshev-Lobatto nodes.

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
    d_lagrange = d // 2 - 1  # d = 2 * (d_lagrange + 1)
    d_lagrange = max(d_lagrange, 1)
    domain = RegularInterval(a, b, 2**n)
    x_vec = domain.to_vector()
    y_vec = func(x_vec)

    # Perform the basic Lagrange interpolation
    time_start = perf_counter()
    mps_basic = lagrange_basic(func, d_lagrange, n, start=a, stop=b, strategy=strategy)
    time_stop = perf_counter()
    time_basic = time_stop - time_start
    maxbond_basic = max(mps_basic.bond_dimensions())
    error_basic = tensor_distance(y_vec, mps_basic.to_vector())
    print(f"Lagrange basic: time {time_basic}")

    # Perform the rank-revealing Lagrange interpolation
    time_start = perf_counter()
    mps_rr = lagrange_rank_revealing(
        func, d_lagrange, n, start=a, stop=b, strategy=strategy
    )
    time_stop = perf_counter()
    time_rr = time_stop - time_start
    maxbond_rr = max(mps_rr.bond_dimensions())
    error_rr = tensor_distance(y_vec, mps_rr.to_vector())
    print(f"Lagrange rank revealing: time {time_rr}")

    # Perform the local Lagrange interpolation
    # Local order = 1
    time_start = perf_counter()
    mps_local_1 = lagrange_local_rank_revealing(
        func, d_lagrange, 1, n, start=a, stop=b, strategy=strategy
    )
    time_stop = perf_counter()
    time_local_1 = time_stop - time_start
    maxbond_local_1 = max(mps_local_1.bond_dimensions())
    error_local_1 = tensor_distance(y_vec, mps_local_1.to_vector())
    print(f"Lagrange local (m=1): time {time_local_1}")

    # Local order = 10
    time_start = perf_counter()
    mps_local_10 = lagrange_local_rank_revealing(
        func, d_lagrange, min(d_lagrange, 10), n, start=a, stop=b, strategy=strategy
    )
    time_stop = perf_counter()
    time_local_10 = time_stop - time_start
    maxbond_local_10 = max(mps_local_10.bond_dimensions())
    error_local_10 = tensor_distance(y_vec, mps_local_10.to_vector())
    print(f"Lagrange local (m=10): time {time_local_10}")

    # Local order = 30
    time_start = perf_counter()
    mps_local_30 = lagrange_local_rank_revealing(
        func, d_lagrange, min(d_lagrange, 30), n, start=a, stop=b, strategy=strategy
    )
    time_stop = perf_counter()
    time_local_30 = time_stop - time_start
    maxbond_local_30 = max(mps_local_30.bond_dimensions())
    error_local_30 = tensor_distance(y_vec, mps_local_30.to_vector())
    print(f"Lagrange local (m=30): time {time_local_30}")

    data = {
        # Basic Lagrange interpolation
        "error_basic": error_basic,
        "maxbond_basic": maxbond_basic,
        "time_basic": time_basic,
        # Rank-revealing Lagrange interpolation
        "error_rr": error_rr,
        "maxbond_rr": maxbond_rr,
        "time_rr": time_rr,
        # Local Lagrange interpolation (local order = 1)
        "error_local_1": error_local_1,
        "maxbond_local_1": maxbond_local_1,
        "time_local_1": time_local_1,
        # Local Lagrange interpolation (local order = 10)
        "error_local_10": error_local_10,
        "maxbond_local_10": maxbond_local_10,
        "time_local_10": time_local_10,
        # Local Lagrange interpolation (local order = 30)
        "error_local_30": error_local_30,
        "maxbond_local_30": maxbond_local_30,
        "time_local_30": time_local_30,
    }
    print(data)
    if name is not None:
        save_pkl(data, filename, path=DATA_PATH)


if __name__ == "__main__":
    # Set parameter ranges
    range_n = list(range(2, 25 + 1))
    range_d = (
        list(range(2, 10)) + list(range(10, 100, 10)) + list(range(100, 1300 + 1, 100))
    )
    range_t = [10 ** -(exp) for exp in range(1, 14 + 1)]

    # Set fixed parameters
    fixed_n = 25
    fixed_d = 1300
    fixed_t = 1e-14

    # Oscillating function
    ε = 1 / 100
    func_o = lambda x: np.cos(1 / (x**2 + ε))

    # Collect data
    for n in range_n:
        data_lagrange(func_o, n, fixed_d, fixed_t, "lagrange_osc")
    for d in range_d:
        data_lagrange(func_o, fixed_n, d, fixed_t, "lagrange_osc")
    for t in range_t:
        data_lagrange(func_o, fixed_n, fixed_d, t, "lagrange_osc")
