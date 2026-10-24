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
    inputFolder = "standard_sweep_pd_k12"
    outputFolder = inputFolder
    z0_array = np.array([3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0]) #np.array([0.5, 1.0, 2.0])
    kp_array = np.array([0.001, 0.002, 0.003, 0.005 ,0.008, 0.01, 0.02, 0.03, 0.05 ,0.08, 0.1])
    V_array = np.array([1,2,3])
    seed_array = np.array([2358692411, 3137230304, 2766775195]) 

    dt = 0.1

    # Generate the data structure
    inputPath = f"input/{inputFolder}"
    Path(inputPath).mkdir(exist_ok=True)
    outputPath = f"output/{outputFolder}"
    Path(outputPath).mkdir(exist_ok=True)

    paths = []
    for j, version in enumerate(V_array):
        for z0 in z0_array:
            for k_P in kp_array:
                seed = int(seed_array[j])
                V = int(version)
                filename = generate_params(z0=z0, k_P=k_P, dt=dt, kappa_n=5e-6, motility=1e-7,inputFolder=inputPath, seed=seed, V=V)
                path = f"{inputPath}/{filename}"
                paths += [path]

    # ------------------------------------------------------------------
    # Run simulations in parallel
    # ------------------------------------------------------------------

    tasks = [(path, outputPath) for path in paths]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(run_one, tasks))

    print("Finished all simulations")