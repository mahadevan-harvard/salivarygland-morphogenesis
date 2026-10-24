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
            
            from anchoredcircle_exponential import run_simulation
            run_simulation(path, outputPath=outputPath)

    print("Finished")

if __name__ == "__main__":

    max_workers = 8

    # Sweep parameters
    inputFolder = "tau_sweep_k12"
    outputFolder = inputFolder
    tau_array = np.array([1e3,2e3,3e3,5e3,1e4,2e4,3e4,5e4,1e5,2e5,3e5,5e5,1e6,2e6,3e6,5e6,1e7])
    dt_array = np.array([0.0001,0.001,0.001,0.001,0.001,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.1,0.1,0.1,0.1])
    V_array = np.array([1,2,3])
    seed_array = np.array([2358692411, 3137230304, 2766775195])    

    z0 = 1.0
    k_P = 0.01

    # Generate the data structure
    inputPath = f"input/{inputFolder}"
    Path(inputPath).mkdir(exist_ok=True)
    outputPath = f"output/{outputFolder}"
    Path(outputPath).mkdir(exist_ok=True)

    paths = []
    for j, version in enumerate(V_array):
        for i,tau in enumerate(tau_array):
            dt = dt_array[i]
            seed = int(seed_array[j])
            V = int(version)
            filename = generate_params(z0=z0, k_P=k_P, tau_g=tau, dt=dt, kappa_n=5e-6, motility=1e-7, seed=seed, V=V, inputFolder=inputPath)
            path = f"{inputPath}/{filename}"
            paths += [path]

    # ------------------------------------------------------------------
    # Run simulations in parallel
    # ------------------------------------------------------------------

    tasks = [(path, outputPath) for path in paths]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(run_one, tasks))

    print("Finished all simulations")