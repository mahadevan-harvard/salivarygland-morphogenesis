# salivarygland-morphogenesis
Brownian dynamics model for bud formation in early morphogenesis of salivary gland.

> *Basement membrane dynamics governs salivary gland budding morphogenesis,*  
> J. Tauber, A. Morales, H. Li, J. Cui, C. Stringer, G.P.T. Choi, S. Wang, L. Mahadevan  
> [TBD].

## Installation

This project uses Python 3. The simulation code relies on JAX.

1. **Clone the repository:**
```bash
   git clone https://github.com/mahadevan-harvard/threshold-branching-erosion.git
   cd threshold-branching-erosion
```
   
2. **(Optional but recommended) Install jax and jax-md through conda environment:**
```bash
   conda create -n jaxmd-env
   conda activate jaxmd-env
   conda install -c conda-forge jax jaxmd
```

3. **Install the package in editable mode::**
```bash
    pip install -e .
```
  
## Repository structure

- `analysis/`: Scripts to plot simulation and segmentation output
- `simulate/`: Core simulation and solver code

## simulation scripts

The core of the simulations code is:
- `interactions.py`: definitions of interaction energies
- `anchoredcircle_exponential.py`: Brownian dynamics implementation and growth definitions

Variants of the simulations with different growth conditions use:
- `anchoredcircle_exponential_washout.py`: simulation with exponential thinning. 
- `anchoredcircle_exponential_perimeteronly.py`: simulation without area growth.

To run parameter sweeps the following scripts can be used:
- template_execute_sweep.py
- template_execute_sweep_washout.py
- template_execute_sweep_kappan.py
- template_execute_sweep_kappan_perimeter.py
- template_execute_sweep_tau.py

These scripts create an input folder where they store a `.json` file with the input parameters, an output folder where the positions are stored at specified intervals in a `.npz` file. This folder also contains a `log/` subfolder, that contains the `stdout` of the simulation. The code to generate the `.json` files is contained in `generate_params.py`.

## Analysis scripts
These are the scripts that are used for analysis of the simulation output (and sometimes experimental data). 
These scripts also create the basis for the figures in the corresponding paper.