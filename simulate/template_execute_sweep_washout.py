"""
Code for manuscript "*Basement membrane dynamics governs salivary gland budding morphogenesis" 
Soft Math Lab (Harvard University)
Author: Justin Tauber
"""

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
            
            from anchoredcircle_exponential_washout import run_simulation
            run_simulation(path, outputPath=outputPath)

if __name__ == "__main__":

    max_workers = 8

    # Sweep parameters
    inputFolder = "exponential_washout_k12"
    outputFolder = inputFolder
    z0_array = np.array([15.0])
    kp_array = np.array([0.01])

    # Generate the data structure
    inputPath = f"input/{inputFolder}"
    Path(inputPath).mkdir(exist_ok=True)
    outputPath = f"output/{outputFolder}"
    Path(outputPath).mkdir(exist_ok=True)

    paths = []
    for z0 in z0_array:
        for k_P in kp_array:

            filename = generate_params(z0=z0, k_P=k_P, kappa_n=5e-6, dt=0.01, V=1, seed=2358692411, inputFolder=inputPath)
            path = f"{inputPath}/{filename}"
            paths += [path]

    # ------------------------------------------------------------------
    # Run simulations in parallel
    # ------------------------------------------------------------------

    tasks = [(path, outputPath) for path in paths]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(run_one, tasks))

    print("Finished all simulations")    