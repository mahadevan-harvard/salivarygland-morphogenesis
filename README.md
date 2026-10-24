# salivarygland-morphogenesis
Brownian dynamics model for bud formation in early morphogenesis of salivary glands.

> *Basement membrane dynamics governs salivary gland budding morphogenesis,*  
> J. Tauber, A. Morales, H. Li, J. Cui, C. Stringer, G.P.T. Choi, S. Wang, L. Mahadevan  
> *In preparation*.

## Model Overview
This repository implements a Brownian dynamics model for epithelial bud formation in salivary gland morphogenesis. The model simulates:

- Particle-based representation of basement membrane
- Spring-like interactions between adjacent particles
- Growth dynamics (exponential, washout, perimeter-only)

For details we refer to the research paper.

## Installation
This project uses Python 3. The simulation code relies on JAX.

1. **Clone the repository:**
```bash
   git clone https://github.com/mahadevan-harvard/salivarygland-morphogenesis.git
   cd salivarygland-morphogenesis
```
   
2. **(Optional but recommended) Install jax and jax-md through conda environment:**
```bash
   conda create -n jaxmd-env
   conda activate jaxmd-env
   conda install -c conda-forge jax jaxmd
```

3. **Install the package in editable mode:**
```bash
    pip install -e .
```
  
## Required software and packages
- Python 3
- JAX
- jax-md
- numpy
- matplotlib
- scipy

## Repository structure
- `analysis/`: Scripts to plot simulation and segmentation output
- `simulate/`: Core simulation and solver code

### simulation scripts
The core of the simulations code is:
- `interactions.py`: definitions of interaction energies
- `anchoredcircle_exponential.py`: Brownian dynamics implementation and growth definitions

Variants of the simulations with different growth conditions use:
- `anchoredcircle_exponential_washout.py`: simulation with exponential thinning. 
- `anchoredcircle_exponential_perimeteronly.py`: simulation without area growth.

To run parameter sweeps the following scripts can be used:
- `template_execute_sweep.py`
- `template_execute_sweep_washout.py`
- `template_execute_sweep_kappan.py`
- `template_execute_sweep_kappan_perimeter.py`
- `template_execute_sweep_tau.py`

These scripts create an input folder where they store a `.json` file with the input parameters, an output folder where the positions are stored at specified intervals in a `.npz` file. This folder also contains a `log/` subfolder, that contains the `stdout` of the simulation. The code to generate the `.json` files is contained in `generate_params.py`.

### Analysis scripts
These are the scripts that are used for analysis of the simulation output (and sometimes experimental data). 
These scripts also create the basis for the figures in the corresponding paper.

A brief description of the scripts:
- `create_acircularity_plot_washout.py`: Create acircularity change plot for the thinning simulation, with respect to the starting point (Fig. 5D).
- `create_acircularity_plot_washoutcontrol.py`: Create acircularity change plot for the thinning simulation, with respect to the washout point (Fig. 5E).
- `create_acircularity_plot.py`: Create acircularity change plot for a range of interface thicknesses (Fig. 4A).
- `create_areavolumeratio_plot_experiment.py`: Create a plot of area and volume change for control glands (Fig. 2C).
- `create_asphericity_plot_experiment.py`: Create a plot of asphericity change for BB-94 treatments (Fig. 4B).
- `create_asphericity_plot_washout_experiment.py`: Create a plot of asphericity change of the washout experiment (Fig. 5F).
- `create_budsize_vs_tg_plot.py`: Create a plot of the mean budsize versus the growth time (Fig. 6B).
- `create_phasediagram.py`: Create phasediagram of the budding-to-ballooning transition (Fig. 6A).
- `create_still_curvature.py`: Plot outline of filament colored according to curvature (Fig. 3B, Fig. 5A).
- `create_still_line.py`: Plot outline of filament in a single color (Fig. 4A, Fig. 6A-C).
- `PlotLibrary.py`: Plotting class with information regarding figure dimensions for different journals.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you use this code, please cite:
> Tauber, J., et al. (2026). Basement membrane dynamics governs salivary gland budding morphogenesis. *In preparation*.
