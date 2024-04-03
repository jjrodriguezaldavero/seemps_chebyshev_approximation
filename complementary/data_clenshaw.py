import numpy as np
import sys, os, pathlib
from typing import Callable
from time import perf_counter

# Add parent path to sys.path
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from tools import tensor_distance, save_pkl

from seemps.state import DEFAULT_STRATEGY
from seemps.analysis.mesh import RegularInterval
from seemps.analysis.chebyshev import (
    interpolation_coefficients,
    cheb2mps,
)


def data_clenshaw(func: Callable, n: int, d: int, t: float, name: str):
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
    suffix = f"_n{n}_d{d}_t{t}"
    filename = name + suffix + ".pkl"
    if os.path.exists("complementary/data/" + filename):
        print(f"{filename} already exists. Skipping computation.")
        return
    print(f"Computing {filename}")

    # Set the default parameters
    a, b = -1, 1
    strategy = DEFAULT_STRATEGY.replace(
        tolerance=t**2,
        simplification_tolerance=t**2,
    )
    domain = RegularInterval(a, b, 2**n)
    coefficients = interpolation_coefficients(func, d, a, b)

    # Perform the MPS Chebyshev approximation using the Clenshaw scheme
    time_start_clenshaw = perf_counter()
    mps_clenshaw = cheb2mps(
        coefficients, domain=domain, strategy=strategy, clenshaw=True
    )
    time_stop_clenshaw = perf_counter()
    time_clenshaw = time_stop_clenshaw - time_start_clenshaw

    # Perform the MPS Chebyshev approximation using the direct scheme
    time_start_direct = perf_counter()
    mps_direct = cheb2mps(
        coefficients, domain=domain, strategy=strategy, clenshaw=False
    )
    time_stop_direct = perf_counter()
    time_direct = time_stop_direct - time_start_direct

    # Evaluate the maximum bond dimension and error of the approximated MPS
    y_vec = func(domain.to_vector())
    maxbond_clenshaw = max(mps_clenshaw.bond_dimensions())
    maxbond_direct = max(mps_direct.bond_dimensions())
    y_clenshaw = mps_clenshaw.to_vector()
    y_direct = mps_direct.to_vector()
    error_clenshaw = tensor_distance(y_vec, y_clenshaw)
    error_direct = tensor_distance(y_vec, y_direct)

    # Save the data as a dictionary in a Pickle file
    data = {
        "time_clenshaw": time_clenshaw,
        "maxbond_clenshaw": maxbond_clenshaw,
        "error_clenshaw": error_clenshaw,
        "time_direct": time_direct,
        "maxbond_direct": maxbond_direct,
        "error_direct": error_direct,
    }
    print(data)
    save_pkl(data, filename, path="complementary/data/")


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

    # Set functions
    # 1. Oscillating function
    ε = 1 / 100
    func_o = lambda x: np.cos(1 / (x**2 + ε))

    # Collect data
    for n in range_n:
        data_clenshaw(func_o, n, fixed_d, fixed_t, "clenshaw_osc")

    for d in range_d:
        data_clenshaw(func_o, fixed_n, d, fixed_t, "clenshaw_osc")

    for t in range_t:
        data_clenshaw(func_o, fixed_n, fixed_d, t, "clenshaw_osc")
