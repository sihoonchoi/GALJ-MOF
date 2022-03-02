# GALJ-MOF

**GALJ-MOF** (**G**aussian-**A**pproximated **L**ennard-**J**ones for **M**etal-**O**rganic **F**rameworks)

### Usage

1. Run `python gaussian_approximation.py`.

2. Paramaters of 8 Gaussians for each element will be saved in the **gaussian_params** directory.

3. `get_GI.py` generates GI descriptors based on GALJ. The code is adapted from [**AMPtorch/CEMT**](https://github.com/ulissigroup/amptorch/tree/CEMT). For installation and instruction of the **AMPtorch** package, please refer to its [GitHub repo](https://github.com/ulissigroup/amptorch).

---

If you use GALJ-MOF in a scientific publication, please cite the following paper:

S. Choi, D. S. Sholl, and A. J. Medford, Gaussian Approximation of Dispersion Potentials for Efficient Featurization and Machine-Learning Predictions of Metal-Organic Frameworks, *J. Chem. Phys.* (2022) (Submitted)

---

### Dependencies
- NumPy
- SciPy
- Atomic Simulation Environment (ASE)

### Acknowledgements
- This works was supported by the Department of Energy, Office of Science, Basic Energy Sciences, under Award #DE-SC0020306.
