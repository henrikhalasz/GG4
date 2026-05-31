from pathlib import Path
from typing import Callable, Dict, Tuple, Union
import importlib.util
import sys
import numpy as np


EstimatorFunction = Callable[
    [np.ndarray, int, int],
    Tuple[np.ndarray, np.ndarray]
]


def load_estimator_functions(folder: Union[str, Path]) -> Dict[str, EstimatorFunction]:
    """
    Load all .py files in a folder that contain a top-level function named
    estimate_latent_and_input.

    Files without this function are ignored.

    Parameters
    ----------
    folder:
        Folder containing estimator .py files.

    Returns
    -------
    Dict[str, EstimatorFunction]
        key:
            filename stem, for example "kalman_em"
        value:
            function object estimate_latent_and_input
    """
    folder = Path(folder).resolve()

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")

    estimators: Dict[str, EstimatorFunction] = {}
    folder_str = str(folder)
    added_to_path = False

    if folder_str not in sys.path:
        sys.path.insert(0, folder_str)
        added_to_path = True

    try:
        for py_file in folder.glob("*.py"):
            if py_file.name.startswith("__") or py_file.name.startswith("test_"):
                continue

            module_name = f"_loaded_estimator_{py_file.stem}"

            spec = importlib.util.spec_from_file_location(module_name, py_file)

            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)

            try:
                spec.loader.exec_module(module)
            except Exception:
                # Ignore files that cannot be imported.
                continue

            func = getattr(module, "estimate_latent_and_input", None)

            if callable(func):
                estimators[py_file.name] = func
    finally:
        if added_to_path:
            try:
                sys.path.remove(folder_str)
            except ValueError:
                pass

    return estimators
