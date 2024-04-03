This repository contains the simulations required to replicate the figures in the paper "Chebyshev Approximation and Composition of Functions for Quantum-Inspired Numerical Analysis". All simulations are supported by SeeMPS (https://github.com/juanjosegarciaripoll/seemps2), an open-source Python library designed to facilitate prototyping and experimentation with MPS/QTT quantum-inspired algorithms.

## Structure

This repository is structured as follows:

- `data` and `figures`: These directories contain the data and figures for the main simulations presented in the paper. Each Pickle file in the `data` directory corresponds to a simulation with a given parameter setting and contains a dictionary where each field corresponds to a given figure of merit.

- `data_chebyshev_1d.py` and `figures_chebyshev_1d.ipynb`:  These files respectively contain the script for obtaining data and the notebook for generating figures related to one-dimensional Chebyshev approximations. The figures are based on the number of qubits `n`, the interpolation order `d`, and the tolerance parameter `t` (corresponding to `\epsilon` in the paper).

- `data_lagrange_1d.py` and `figures_lagrange_1d.ipynb`: These files respectively contain the script for obtaining data and the notebook for generating figures related to one-dimensional multiscale interpolative constructions of the oscillating function. The structure is similar to that used for one-dimensional Chebyshev approximations.

- `data_cross_1d.py` and `figures_cross_1d.ipynb`: These files respectively contain the script for obtaining data and the notebook for generating figures related to one-dimensional tensor cross-interpolation (TCI). This follows the same structure as for one-dimensional Chebyshev approximations, but replaces the interpolation order `d` with the maximum threshold bond dimension `r` (corresponding to `\chi_thr` in the paper).

- `figures_benchmark_1d.ipynb`: This notebook contains the figures corresponding to the benchmarks between the three algorithms for the univariate functions.

- `data_chebyshev_md.py` and `figures_chebyshev_md.ipynb`: These files respectively contain the script for obtaining data and the notebook for generating figures related to multivariate Chebyshev approximations, with respect to the dimension `m` and the tolerance parameter `t`. The actual figures are included in the notebook `figures_cross_md.ipynb` to integrate them with the TCI results.

- `data_cross_md.py` and `figures_figures_md.ipynb`: These files respectively contain the script for obtaining data and the notebook for generating figures related to multivariate tensor cross-interpolations, with respect to the dimension `m`. This notebook includes the figures presented in the paper, combining the results of Chebyshev approximations with TCI and the multidimensional benchmark figures.

- `complementary`: This directory contains simulation scripts and notebooks for generating complementary figures, such as those included in the Appendices of the paper.

- `tools`: This directory contains the required tools and scripts for data collection data and figure plotting.


## Authors
- Juan José Rodríguez-Aldavero
- Paula García-Molina
- Luca Tagliacozzo
- Juan José García-Ripoll

## Acknowledgments
This work has been supported by Spanish Projects No. PID2021-127968NB-I00 and No. PDC2022-133486-I00, funded by MCIN/AEI/10.13039/501100011033 and by the European Union “NextGenerationEU”/PRTR”1. The authors gratefully acknowledge the Scientific Computing Area (AIC), SGAI-CSIC, for their assistance while using the DRAGO Supercomputer for performing the simulations, and Centro de Supercomputación de Galicia (CESGA) who provided access to the supercomputer FinisTerrae. The authors thank the Institut Henri Poincaré (UAR 839 CNRS-Sorbonne Université) and the LabEx CARMIN (ANR-10-LABX-59-01) for their support. PGM acknowledges the funds given by ``FSE invierte en tu futuro" through an FPU Grant FPU19/03590 and by MCIN/AEI/10.13039/501100011033.
