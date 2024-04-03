import numpy as np
import os
from time import perf_counter

from tools import save_pkl, sampled_error

from seemps.state import Strategy, Simplification, Truncation, DEFAULT_TOLERANCE
from seemps.truncate import simplify
from seemps.analysis.mesh import Mesh, RegularInterval
from seemps.analysis.factories import mps_interval, mps_tensor_sum
from seemps.analysis.chebyshev import (
    interpolation_coefficients,
    cheb2mps,
    estimate_order,
)

DATA_PATH = "data/"

import seemps.tools

seemps.tools.DEBUG = 0


def data_chebyshev_md(
    m: int,
    n: int,
    d: int,
    t: float,
    gaussian_type: str,
    mps_order: str,
    save: bool = True,
):
    """
    Computes a dictionary containing data that evaluates the performance of
    the multivariate MPS Chebyshev approximation algorithm, based on the
    interpolation on the Chebyshev-Gauss nodes.
    Saves it as a Pickle file with the given name and a suffix given by the parameters
    on the `/data` directory.

    Parameters
    ----------
    m : int
        The dimension of the function.
    n : int
        The number of qubits per dimension of the MPS approximation.
    d : int
        The polynomial order of the Chebyshev approximation.
    t : float
        The tolerance parameter of the variational simplification routine.
    gaussian_type : str
        The type of multivariate gaussian to approximate. Either 'product'
        or 'squeezed'.
    mps_order : str
        The order of the MPS qubits. Either 'A' or 'B'.

    Notes
    -----
    - The serial order is very sensible to simplifications of the initial MPS.
    Hence, we simplify the initial domain in serial order with a fixed tolerance
    while in the interleaved order we use the tolerance `t`.
    """
    # Get the Pickle file name and check if it exists
    if save:
        suffix = f"_m{m}_n{n}_d{d}_t{t}_order{mps_order}"
        filename = "chebyshev_md_" + gaussian_type + suffix + ".pkl"
        if os.path.exists(DATA_PATH + filename):
            print(f"{filename} already exists. Skipping computation.")
            return
        print(f"Computing {filename}")

    # Set the default parameters and Strategies
    a, b = -1, 1
    func = lambda x: np.exp(-x)
    strategy_cheb = Strategy(
        method=Truncation.RELATIVE_NORM_SQUARED_ERROR,
        simplify=Simplification.VARIATIONAL_EXACT_GUESS,
        tolerance=t**2,
        simplification_tolerance=t**2,
        max_sweeps=20,
    )
    strategy_domain_B = Strategy(
        method=Truncation.RELATIVE_NORM_SQUARED_ERROR,
        tolerance=t**2,
        simplify=Simplification.DO_NOT_SIMPLIFY,
    )

    # Perform the MPS Chebyshev interpolation on the Chebyshev zeros
    tick = perf_counter()
    interval = RegularInterval(a, b, 2**n)
    mps_x = mps_interval(interval)

    if gaussian_type == "product":
        mps_x = mps_interval(interval)
        mps_x_squared = mps_x * mps_x
        # mps_x_squared = mps_x * mps_x
        mps_domain = mps_tensor_sum([mps_x_squared] * m, mps_order=mps_order)
        if mps_order == "A":
            pass
        elif mps_order == "B":
            mps_domain = simplify(mps_domain, strategy=strategy_domain_B)
        # x**2 + y**2 + ...
        a_domain, b_domain = (0, m * b**2)
        d_truncated = min(estimate_order(func, a_domain, b_domain, tolerance=t), d)
        func_tensor = lambda tensor: func(np.sum(tensor**2, axis=0))

    elif gaussian_type == "squeezed":
        mps_x_sum = mps_tensor_sum([mps_x] * m, mps_order=mps_order)
        mps_domain = mps_x_sum * mps_x_sum
        if mps_order == "A":
            # mps_domain = simplify(mps_domain, strategy=strategy_domain_A)
            pass
        elif mps_order == "B":
            mps_domain = simplify(mps_domain, strategy=strategy_domain_B)
        # (x + y + ...)**2
        a_domain, b_domain = (0, (m * b) ** 2)
        d_truncated = min(estimate_order(func, a_domain, b_domain, tolerance=t), d)
        func_tensor = lambda tensor: func(np.sum(tensor, axis=0) ** 2)

    coefficients = interpolation_coefficients(func, d_truncated, a_domain, b_domain)
    mps = cheb2mps(coefficients, initial_mps=mps_domain, strategy=strategy_cheb)
    time = perf_counter() - tick

    # Evaluate the MPS maxbond and sampled error
    maxbond = max(mps.bond_dimensions())
    mesh = Mesh([interval] * m)
    error_mean, error_std = sampled_error(func_tensor, mps, mesh, mps_order=mps_order)

    # Save the data as a dictionary in a Pickle file
    data = {
        "time": time,
        "maxbond": maxbond,
        "mean_error": error_mean,
        "std_error": error_std,
    }
    print(data)
    if save:
        save_pkl(data, filename, path=DATA_PATH)


if __name__ == "__main__":
    # Set parameter ranges
    range_m = list(range(1, 10 + 1))
    range_t = [10 ** -(exp) for exp in range(1, 14 + 1)]

    # Set fixed parameters
    fixed_m = 10
    fixed_n = 20
    max_d = 100
    fixed_t = 1e-14

    # We impose a hard limit on the maximum dimension to avoid an exponentially large tensor
    max_m_product_b = 4

    # Collect data
    for m in range_m:
        data_chebyshev_md(m, fixed_n, max_d, fixed_t, "product", "A")
        if m <= max_m_product_b:
            data_chebyshev_md(m, fixed_n, max_d, fixed_t, "product", "B")
        data_chebyshev_md(m, fixed_n, max_d, fixed_t, "squeezed", "A")
        data_chebyshev_md(m, fixed_n, max_d, fixed_t, "squeezed", "B")

    for t in range_t:
        data_chebyshev_md(fixed_m, fixed_n, max_d, t, "product", "A")
        data_chebyshev_md(fixed_m, fixed_n, max_d, t, "squeezed", "A")
        data_chebyshev_md(fixed_m, fixed_n, max_d, t, "squeezed", "B")
