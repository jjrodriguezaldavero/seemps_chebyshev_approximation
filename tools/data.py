import numpy as np
import os
import pickle
from itertools import product


def save_pkl(data, name: str, path: str):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, name)
    with open(file_path, "wb") as file_obj:
        pickle.dump(data, file_obj)


def merge_pkl(parameters: dict, name: str, path: str) -> np.ndarray:
    parameter_names = list(parameters.keys())
    parameter_values = [
        [value] if np.isscalar(value) else value for value in parameters.values()
    ]
    idx_product = product(*[range(len(v)) for v in parameter_values])
    values_product = product(*parameter_values)
    combinations = [(idx, value) for idx, value in zip(idx_product, values_product)]
    results = np.empty([len(value) for value in parameter_values], dtype=object)
    for combination in combinations:
        idx, parameter_values = combination
        suffix = "_".join(
            "".join(map(str, pair)) for pair in zip(parameter_names, parameter_values)
        )
        filename = f"{name}_{suffix}.pkl"
        full_path = os.path.join(path, filename)
        with open(full_path, "rb") as f:
            result = pickle.load(f)
        results[idx] = result
    return results


def rename_pkl_fields(parameters: dict, name: str, path: str, mapping: dict) -> None:
    parameter_names = list(parameters.keys())
    parameter_values = [
        [value] if np.isscalar(value) else value for value in parameters.values()
    ]
    idx_product = product(*[range(len(v)) for v in parameter_values])
    values_product = product(*parameter_values)
    combinations = [(idx, value) for idx, value in zip(idx_product, values_product)]

    for combination in combinations:
        _, parameter_values = combination
        suffix = "_".join(
            "".join(map(str, pair)) for pair in zip(parameter_names, parameter_values)
        )
        filename = f"{name}_{suffix}.pkl"
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            with open(full_path, "rb") as file:
                data = pickle.load(file)
            for old_field, new_field in mapping.items():
                if old_field in data:
                    data[new_field] = data.pop(old_field)
            with open(full_path, "wb") as file:
                print(f"Modifying {full_path}")
                pickle.dump(data, file)


def read_field(results: np.ndarray, field: str) -> np.ndarray:
    return np.array([result[field] for result in results.reshape(-1)])
