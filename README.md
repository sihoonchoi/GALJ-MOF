# GI

**GI** (**G**aussian **I**ntegral) Descriptors

## Usage

### 1. Run Gaussian Approximation
To perform Gaussian approximation of the original Lennard-Jones (LJ) potential curves, run:
```bash
python gauss_approx_lj.py
```

To perform Gaussian approximation of the soft LJ curves, run:
```bash
python gauss_approx_soft_lj.py
```

The optimized Gaussian parameters for each element will be saved in:
- `gauss_params/lj` for the origianl LJ potential
- `gauss_params/soft_lj` for the soft LJ potential

### 2. Generate GI Descriptor Vectors
To compute the GI descriptors, run the following command:
```bash
python gi.py --system --name OPAGIX --sigmas 0.1 0.2 0.3 --mcsh-orde 2 --mof-pool mean --mol-pool com
```

**Required Arguments**

- `--system`: specifies the system type
    - `mof`: for metal-organic frameworks
    - `mol`: for molecules
- `--name`: name of the compound
    - If `--system` is `mof`, it will read `mof/{name}.cif`
    - If `--system` is `mol`, it will read `mol/{name}.xyz`
- `--sigmas`: space-separated list of `sigma` values used to generate the GI descriptors
- `--mcsh-order`: Maximum MCSH order to include in the descriptor
- `--mof-pool`: pooling strategy for MOFs
    - `min`: use the minimum value across the structure
    - `mean`: use the mean value across the structure
- `--mol-pool`:
    - `mean`: use the mean value across all atoms
    - `com`: compute a single descriptor vector at the center-of-mass

The resulting GI descriptors will be saved in the `descriptors/` directory with the filename `{system}.csv`.

---

If you use the GI descriptor scheme in a scientific publication, please cite the following paper:

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
- The codes in `gi.py` is adapted from [**AMPtorch/CEMT**](https://github.com/ulissigroup/amptorch/tree/CEMT). For installation and instruction of the **AMPtorch** package, please refer to its official [GitHub repo](https://github.com/ulissigroup/amptorch).
