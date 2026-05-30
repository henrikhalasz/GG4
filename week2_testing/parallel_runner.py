"""
parallel_runner.py

Utility for running a dictionary of independent functions in parallel.

Typical notebook usage
----------------------

from parallel_runner import run_functions_parallel

results = run_functions_parallel(
    functions,
    observation,
    latent_dim=8,
    input_dim=2,
    max_workers="auto",
    native_threads_per_worker=1,
)

Each function should have signature like:

    func(observation, LatentDim=8, InputDim=2)

and should return:

    latent_states, inputs

Important:
- For best multiprocessing compatibility, define your functions in an importable .py file,
  not only inside a Jupyter cell.
- Put this import near the start of the notebook, before importing heavy numerical libraries
  if possible.
"""

from __future__ import annotations

import os

# These environment variables must be set before NumPy / SciPy / sklearn / JAX / Torch
# are imported in each process. They reduce CPU oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Useful for JAX CPU execution. Safe to leave even if JAX is not used.
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Tuple, Union


FunctionDict = Mapping[str, Callable[..., Tuple[Any, Any]]]
Backend = Literal["process", "thread"]


def _limit_native_threads_runtime(n_threads: int) -> None:
    """
    Best-effort runtime limiting of native numerical thread pools.

    This helps if NumPy / scipy / sklearn / torch have already been imported.
    It is not as reliable as setting environment variables before import,
    but it usually helps.
    """
    if n_threads < 1:
        n_threads = 1

    # threadpoolctl controls BLAS/OpenMP pools used by NumPy, scipy, sklearn, etc.
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=n_threads)
    except Exception:
        pass

    # Torch may use its own intra-op / inter-op thread pools.
    try:
        import torch

        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(max(1, n_threads))
    except Exception:
        pass


def _worker_initializer(native_threads_per_worker: int) -> None:
    """
    Initializer run once inside each worker process.
    """
    os.environ["OMP_NUM_THREADS"] = str(native_threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(native_threads_per_worker)
    os.environ["OPENBLAS_NUM_THREADS"] = str(native_threads_per_worker)
    os.environ["BLIS_NUM_THREADS"] = str(native_threads_per_worker)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(native_threads_per_worker)
    os.environ["NUMEXPR_NUM_THREADS"] = str(native_threads_per_worker)
    os.environ["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=false "
        f"intra_op_parallelism_threads={native_threads_per_worker}"
    )

    _limit_native_threads_runtime(native_threads_per_worker)


def _run_one_task(
    task: Tuple[str, Callable[..., Tuple[Any, Any]], Any, int, int, int]
) -> Tuple[str, Any, Any]:
    """
    Execute one user function.

    task = (name, func, observation, latent_dim, input_dim, native_threads_per_worker)
    """
    name, func, observation, latent_dim, input_dim, native_threads_per_worker = task

    _limit_native_threads_runtime(native_threads_per_worker)

    latent_states, inputs = func(
        observation,
        LatentDim=latent_dim,
        InputDim=input_dim,
    )

    return name, latent_states, inputs


def choose_worker_count(
    n_tasks: int,
    max_workers: Union[int, Literal["auto"]] = "auto",
) -> int:
    """
    Choose a sensible number of parallel workers.

    max_workers="auto" means:
        min(number_of_tasks, os.cpu_count())

    For 17 functions:
        - 8 logical CPU threads -> 8 workers
        - 32 logical CPU threads -> 17 workers
    """
    if n_tasks <= 0:
        return 0

    if max_workers == "auto":
        cpu_count = os.cpu_count() or 1
        return min(n_tasks, cpu_count)

    if not isinstance(max_workers, int):
        raise TypeError("max_workers must be an int or 'auto'.")

    if max_workers < 1:
        raise ValueError("max_workers must be >= 1.")

    return min(n_tasks, max_workers)


def run_functions_parallel(
    functions: FunctionDict,
    observation: Any,
    *,
    latent_dim: int = 8,
    input_dim: int = 2,
    max_workers: Union[int, Literal["auto"]] = "auto",
    native_threads_per_worker: int = 1,
    backend: Backend = "process",
    print_shapes: bool = True,
    return_ordered: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Run a dictionary of functions in parallel.

    Parameters
    ----------
    functions:
        Dictionary or mapping:
            name -> function

        Each function must accept:
            func(observation, LatentDim=latent_dim, InputDim=input_dim)

        Each function must return:
            latent_states, inputs

    observation:
        Input passed to every function.

    latent_dim:
        Passed as LatentDim.

    input_dim:
        Passed as InputDim.

    max_workers:
        "auto" or integer.

        "auto" uses min(len(functions), os.cpu_count()).

    native_threads_per_worker:
        Number of internal numerical-library threads allowed per worker.
        Usually use 1 when using process-level parallelism.

    backend:
        "process":
            Best for CPU-heavy pure Python or mixed Python/NumPy/JAX work.
            Requires functions to be pickleable.

        "thread":
            Less strict about pickling and often works better directly inside
            notebooks, but Python-level CPU work will be limited by the GIL.
            It can still be useful if the functions mainly call NumPy/JAX/PyTorch
            kernels that release the GIL.

    print_shapes:
        If True, print result shapes as each task finishes.

    return_ordered:
        If True, return dictionary sorted by function name.
        If False, return in completion order.

    Returns
    -------
    results:
        Dictionary:
            name -> {
                "latent_states": latent_states,
                "inputs": inputs,
            }
    """
    if native_threads_per_worker < 1:
        raise ValueError("native_threads_per_worker must be >= 1.")

    items = sorted(functions.items())
    n_tasks = len(items)

    if n_tasks == 0:
        return {}

    n_workers = choose_worker_count(n_tasks, max_workers)

    tasks = [
        (
            name,
            func,
            observation,
            latent_dim,
            input_dim,
            native_threads_per_worker,
        )
        for name, func in items
    ]

    results: Dict[str, Dict[str, Any]] = {}

    if backend == "process":
        executor_cls = ProcessPoolExecutor
        executor_kwargs = {
            "max_workers": n_workers,
            "initializer": _worker_initializer,
            "initargs": (native_threads_per_worker,),
        }
    elif backend == "thread":
        executor_cls = ThreadPoolExecutor
        executor_kwargs = {
            "max_workers": n_workers,
        }

        # For thread backend, limit native pools in the parent process too.
        _limit_native_threads_runtime(native_threads_per_worker)
    else:
        raise ValueError("backend must be either 'process' or 'thread'.")

    with executor_cls(**executor_kwargs) as executor:
        future_to_name = {
            executor.submit(_run_one_task, task): task[0]
            for task in tasks
        }

        for future in as_completed(future_to_name):
            name = future_to_name[future]

            try:
                finished_name, latent_states, inputs = future.result()
            except Exception as exc:
                raise RuntimeError(f"Function {name!r} failed during parallel execution.") from exc

            results[finished_name] = {
                "latent_states": latent_states,
                "inputs": inputs,
            }

            if print_shapes:
                latent_shape = getattr(latent_states, "shape", None)
                input_shape = getattr(inputs, "shape", None)

                print(finished_name)
                print("  latent_states:", latent_shape)
                print("  inputs:       ", input_shape)

    if return_ordered:
        return {name: results[name] for name, _ in items}

    return results


def run_functions_sequential(
    functions: FunctionDict,
    observation: Any,
    *,
    latent_dim: int = 8,
    input_dim: int = 2,
    print_shapes: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Sequential fallback with the same return format as run_functions_parallel.
    Useful for debugging.
    """
    results: Dict[str, Dict[str, Any]] = {}

    for name, func in sorted(functions.items()):
        latent_states, inputs = func(
            observation,
            LatentDim=latent_dim,
            InputDim=input_dim,
        )

        results[name] = {
            "latent_states": latent_states,
            "inputs": inputs,
        }

        if print_shapes:
            print(name)
            print("  latent_states:", getattr(latent_states, "shape", None))
            print("  inputs:       ", getattr(inputs, "shape", None))

    return results
