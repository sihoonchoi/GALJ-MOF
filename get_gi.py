import numpy as np
import glob
import os

from ase.io import read

from amptorch.preprocessing import AtomsToData
from amptorch.descriptor.GMPOrderNorm import GMPOrderNorm

def main(args):
    path = 'grids'
    if not os.path.isdir(path):
        os.makedirs(path)

    mofs = glob.glob(f'{args.mof_dir}/*.cif')
    for mof in mofs:
        atoms = read(mof)
        elements = atoms.get_chemical_symbols()
        elements = list(set(elements))

        print(f"Generating GMP descriptors for {mof[:-4]}")

        x = int(np.ceil(atoms.cell.cellpar()[0] / args.grid_spacing))
        y = int(np.ceil(atoms.cell.cellpar()[1] / args.grid_spacing))
        z = int(np.ceil(atoms.cell.cellpar()[2] / args.grid_spacing))

        positions = []
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    pos = atoms.cell[0] * i / x + atoms.cell[1] * j / y + atoms.cell[2] * k / z
                    positions.append(list(pos))
        positions = np.array(positions)

        GMPs = {
            "MCSHs": {
                "orders": [0],
                "sigmas": args.sigma
                },
            "atom_gaussians": {},
            "cutoff": max(20, 5 * max(np.array(args.sigma)) + 5),
            "square": False
        }

        for element in elements:
            GMPs['atom_gaussians'][element] = "./gaussian_params/{}.g".format(element)

        descriptor = GMPOrderNorm(MCSHs = GMPs, elements = elements)
        # descriptor_setup = ('gmpordernorm', GMPs, GMPs.get('cutoff_distance'), elements)

        a2d = AtomsToData(
            descriptor = descriptor,
            r_energy = False,
            r_forces = False,
            save_fps = False,
            fprimes = False)

        do = a2d.convert(atoms, ref_positions = positions, idx = 0)
        fps = do.fingerprint.numpy()
        descriptors = np.concatenate((positions, fps), axis = 1)

        with open(f'{path}/{mof.split("/")[-1][:-4]}.npy', 'wb') as f:
            np.save(f, descriptors)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mof-dir", required = True, type = str)
    parser.add_argument("--sigma", required = True, nargs = '+', type = float)
    parser.add_argument("--grid-spacing", default = 0.2, type = float)
    args = parser.parse_args()

    main(args)