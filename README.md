# GALJ-MOF

**GALJ-MOF** (**G**aussian-**A**pproximated **L**ennard-**J**ones for **M**etal-**O**rganic **F**rameworks)

### Usage

1. Run `python gaussian_approximation.py`.

2. Paramaters of 8 Gaussians for each element will be saved in the **gaussian_params** directory.

3. To generate GI descriptor grids, execute the command `python get_gi.py`. Include the following arguments for customization:

- `--mof-dir`: path to the directory containing CIF files
- `--sigma`: a list of `sigma` parameters to generate GI descriptors
- `--grid-spacing`: grid spacing of GI grids (default: 0.2)

The generated GI descriptor grids will be stored in the **grids** directory under the filename `mofname.npy`. To execute the `get_gi.py` script, use the command like:

`python get_gi.py --mof-dir MOFs --sigma 0.1 0.2 0.3 --grid-spacing 0.5`

4. The first 3 columns in `mofname.npy` represent the x, y, and z coordinates of grid points. Subsequent columns contain GI descriptors with their corresponding `--sigma` values.

---

If you use GALJ-MOF in a scientific publication, please cite the following paper:

S. Choi, D. S. Sholl, and A. J. Medford, Gaussian Approximation of Dispersion Potentials for Efficient Featurization and Machine-Learning Predictions of Metal-Organic Frameworks, *J. Chem. Phys.* 2022, 156, 214108. DOI: https://doi.org/10.1063/5.0091405

---

### Dependencies
- NumPy
- SciPy
- Atomic Simulation Environment (ASE)
- PyTorch
- Skorch
- AMPTorch

### Acknowledgements
- This works was supported by the Department of Energy, Office of Science, Basic Energy Sciences, under Award #DE-SC0020306.
- The codes in `get_gi.py` is adapted from [**AMPtorch/CEMT**](https://github.com/ulissigroup/amptorch/tree/CEMT). For installation and instruction of the **AMPtorch** package, please refer to its official [GitHub repo](https://github.com/ulissigroup/amptorch).
