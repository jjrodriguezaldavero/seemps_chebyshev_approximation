import numpy as np
from typing import Optional
from math import sqrt

from seemps.state import MPS, CanonicalMPS, MPSSum, Strategy, DEFAULT_STRATEGY
from seemps.truncate import simplify
from seemps.analysis.mesh import Interval
from seemps.analysis.factories import mps_interval, mps_affine
from seemps.tools import make_logger


def cheb2mps(
    coefficients: np.polynomial.Chebyshev,
    initial_mps: Optional[MPS] = None,
    domain: Optional[Interval] = None,
    strategy: Strategy = DEFAULT_STRATEGY,
    clenshaw: bool = True,
    rescale: bool = True,
) -> MPS:
    """
    Modification of `cheb2mps` that returns the intermediate bond dimensions in an array.
    """
    if isinstance(initial_mps, MPS):
        pass
    elif isinstance(domain, Interval):
        initial_mps = mps_interval(domain)
    else:
        raise ValueError("Either a domain or an initial MPS must be provided.")
    if rescale:
        orig = tuple(coefficients.linspace(2)[0])
        initial_mps = mps_affine(initial_mps, orig, (-1, 1))

    c = coefficients.coef
    I_norm = 2 ** (initial_mps.size / 2)
    normalized_I = CanonicalMPS(
        [np.ones((1, 2, 1)) / sqrt(2.0)] * initial_mps.size,
        center=0,
        is_canonical=True,
    )
    x_norm = initial_mps.norm()
    normalized_x = CanonicalMPS(
        initial_mps, center=0, normalize=True, strategy=strategy
    )

    bonds = initial_mps.bond_dimensions()

    logger = make_logger()
    if clenshaw:
        steps = len(c)
        logger("MPS Clenshaw evaluation started")
        y_i = y_i_plus_1 = normalized_I.zero_state()
        for i, c_i in enumerate(reversed(c)):
            y_i_plus_1, y_i_plus_2 = y_i, y_i_plus_1
            y_i = simplify(
                # coef[i] * I - y[i + 2] + (2 * x_mps) * y[i + 1],
                MPSSum(
                    weights=[c_i * I_norm, -1, 2 * x_norm],
                    states=[normalized_I, y_i_plus_2, normalized_x * y_i_plus_1],
                    check_args=False,
                ),
                strategy=strategy,
            )
            bonds = np.vstack((bonds, y_i.bond_dimensions()))
            logger(
                f"MPS Clenshaw step {i+1}/{steps}, maxbond={y_i.max_bond_dimension()}, error={y_i.error():6e}"
            )
        f_mps = simplify(
            MPSSum(
                weights=[1, -x_norm],
                states=[y_i, normalized_x * y_i_plus_1],
                check_args=False,
            ),
            strategy=strategy,
        )
        bonds = np.vstack((bonds, f_mps.bond_dimensions()))
    else:
        steps = len(c)
        logger("MPS Chebyshev expansion started")
        f_mps = simplify(
            MPSSum(
                weights=[c[0] * I_norm, c[1] * x_norm],
                states=[normalized_I, normalized_x],
                check_args=False,
            ),
            strategy=strategy,
        )
        T_i, T_i_plus_1 = I_norm * normalized_I, x_norm * normalized_x
        for i, c_i in enumerate(c[2:], start=2):
            T_i_plus_2 = simplify(
                MPSSum(
                    weights=[2 * x_norm, -1],
                    states=[normalized_x * T_i_plus_1, T_i],
                    check_args=False,
                ),
                strategy=strategy,
            )
            f_mps = simplify(
                MPSSum(weights=[1, c_i], states=[f_mps, T_i_plus_2], check_args=False),
                strategy=strategy,
            )
            bonds = np.vstack((bonds, f_mps.bond_dimensions()))
            logger(
                f"MPS expansion step {i+1}/{steps}, maxbond={f_mps.max_bond_dimension()}, error={f_mps.error():6e}"
            )
            T_i, T_i_plus_1 = T_i_plus_1, T_i_plus_2
    logger.close()
    return f_mps, bonds
