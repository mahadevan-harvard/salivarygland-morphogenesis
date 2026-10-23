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