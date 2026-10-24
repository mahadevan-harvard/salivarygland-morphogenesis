import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import contextlib

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np
from generate_params import *

def run_one(args):
    path, outputPath = args

    log_dir = Path(outputPath) / "logs"
    log_dir.mkdir(exist_ok=True)

    stem = Path(path).stem
    log_file = log_dir / f"{stem}.log"

    print(f"Running {path}")
    with open(log_file, "w", buffering=1) as f:
        with contextlib.redirect_stdout(f), \
        contextlib.redirect_stderr(f):
            print(f"PID {os.getpid()}")
            
            from anchoredcircle_exponential import run_simulation
            run_simulation(path, outputPath=outputPath)

    print("Finished")

if __name__ == "__main__":

    # Sweep parameters
    inputFolder = "kappa_sweep"
    outputFolder = inputFolder
    kappa_array = np.array([0.0, 5e-6])
    z0 = 1.0
    k_P = 0.01

    # Generate the data structure
    inputPath = f"input/{inputFolder}"
    Path(inputPath).mkdir(exist_ok=True)
    outputPath = f"output/{outputFolder}"
    Path(outputPath).mkdir(exist_ok=True)

    paths = []
    for i,kappa in enumerate(kappa_array):
        filename = generate_params(z0=z0, k_P=k_P, kappa_n=kappa, inputFolder=inputPath)
        path = f"{inputPath}/{filename}"
        paths += [path]

    # ------------------------------------------------------------------
    # Run simulations in parallel
    # ------------------------------------------------------------------

    tasks = [(path, outputPath) for path in paths]

    with ProcessPoolExecutor(max_workers=4) as executor:
        list(executor.map(run_one, tasks))

    print("Finished all simulations")