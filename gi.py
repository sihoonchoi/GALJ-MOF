import numpy as np
import pandas as pd
import os
import pymatgen.io.zeopp as zeopp

from ase import Atoms
from ase.io import read
from amptorch.preprocessing import AtomsToData
from amptorch.descriptor.GMPOrderNorm import GMPOrderNorm
from pymatgen.io.ase import AseAtomsAdaptor

class suppress_stdout(object):
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

def get_voronoi_nodes(atoms, min_radius = 0.5):
    frac_positions = atoms.get_scaled_positions() * atoms.cell[:].max(axis = 0)
    structure = AseAtomsAdaptor.get_structure(Atoms(atoms.get_chemical_symbols(), frac_positions, cell = np.eye(3) * atoms.cell[:].max(axis = 0), pbc = [True, True, True]))
    with suppress_stdout():
        nodes = zeopp.get_high_accuracy_voronoi_nodes(structure)
    remove_idx = [
        idx for idx, site in enumerate(nodes.sites) if site.properties['voronoi_radius'] < min_radius
    ]
    nodes.remove_sites(remove_idx)

    voronoi_radii = np.array([site.properties['voronoi_radius'] for site in nodes.sites])
    points = nodes.cart_coords

    max_voronoi_radii = np.where(voronoi_radii == max(voronoi_radii))[0]
    points /= atoms.cell[:].max(axis = 0)
    points = np.dot(points, atoms.cell[:])
    return np.array(points[max_voronoi_radii])

def generate_gmp(system, name, sigmas, mcsh_order, mof_pool, mol_pool, target_dir):
    extension = 'cif' if system == 'mof' else 'xyz'
    atoms = read(f'./{system}/{name}.{extension}')
    elements = atoms.get_chemical_symbols()
    GMPs = {
        "MCSHs": {
            "orders": list(np.arange(mcsh_order + 1)),
            "sigmas": np.array(sigmas)
        },
        "atom_gaussians": {},
        "cutoff": max(20, 5 * max(np.array(sigmas)) + 5),
        "square": False
    }

    for element in set(elements):
        GMPs['atom_gaussians'][element] = f"./gauss_params/soft_lj/{element}.g"

    descriptor = GMPOrderNorm(MCSHs = GMPs, elements = set(elements))

    a2d = AtomsToData(
        descriptor = descriptor,
        r_energy = False,
        r_forces = False,
        save_fps = False,
        fprimes = False,
        cores = 1)

    if system == 'mof':
        ref_positions = get_voronoi_nodes(atoms)
        gi = a2d.convert(atoms, ref_positions = ref_positions, idx = 0)
        if mof_pool == 'min':
            gi_pool = gi.fingerprint.numpy().min(axis = 0)
        elif mof_pool == 'mean':
            gi_pool = gi.fingerprint.numpy().mean(axis = 0)
    elif system == 'mol':
        if mol_pool == 'mean':
            gi  = a2d.convert(atoms, idx = 0)
            gi_pool = gi.fingerprint.numpy().mean(axis = 0)
        elif mol_pool == 'com':
            ref_positions = atoms.get_center_of_mass().reshape(1, -1)
            gi = a2d.convert(atoms, ref_positions = ref_positions, idx = 0)
            gi_pool = gi.fingerprint.numpy().reshape(-1,)

    fps = np.concatenate((np.array([name]), gi_pool)).reshape(1, -1)
    columns = [system] + [f'{system}-{o}-{s}' for o in np.arange(mcsh_order + 1) for s in sigmas]
    df = pd.DataFrame(fps, columns = columns)

    if system == 'mof':
        df.to_csv(f'{target_dir}/descriptors/{system}.csv', index = False)
    elif system == 'mol':
        df.to_csv(f'{target_dir}/descriptors/{system}.csv', index = False)

def main(args):
    os.makedirs(f'{args.target_dir}/descriptors', exist_ok = True)
    generate_gmp(args.system, args.name, args.sigmas, args.mcsh_order, args.mof_pool, args.mol_pool, args.target_dir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required = True, type = str)
    parser.add_argument("--name", required = True, type = str)
    parser.add_argument("--sigmas", required = True, nargs = '+', type = float)
    parser.add_argument("--target-dir", default = '.', type = str)
    parser.add_argument("--mof-pool", default = 'mean', type = str)
    parser.add_argument("--mol-pool", default = 'com', type = str)
    parser.add_argument("--mcsh-order", default = 2, type = int)
    args = parser.parse_args()

    main(args)
