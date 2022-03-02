import numpy as np
import sys
import os

from ase import Atom, Atoms
from ase.io import read

from amptorch.preprocessing import AtomsToData
from amptorch.descriptor.GMPOrderNorm import GMPOrderNorm

mof = read('./MOF/{}'.format(sys.argv[1]))

elements = mof.get_chemical_symbols()
elements = list(set(elements))

print("Generating GMP descriptors for {}".format(sys.argv[1][:-4]))

grid_spacing = 0.2

x = np.ceil(mof.cell.cellpar()[0] / 0.2)
y = np.ceil(mof.cell.cellpar()[1] / 0.2)
z = np.ceil(mof.cell.cellpar()[2] / 0.2)

positions = []
for i in range(x):
    for j in range(y):
        for k in range(z):
            pos = mof.cell[0] * i / grid.shape[0] + mof.cell[1] * j / grid.shape[1] + mof.cell[2] * k / grid.shape[2]
            positions.append(list(pos))
positions = np.array(positions)

sigmas = [0.000001, 0.25, 0.5] # sigma = 1e-6 is equivalent to 1-D energy histograms

GMPs = {
    "MCSHs": {
        "orders": [0],
        "sigmas": sigmas
        },
    "atom_gaussians": {},
    "cutoff": 10,
    "square": False
}

for element in elements:
    GMPs['atom_gaussians'][element] = "./gaussian_params/{}.g".format(element)

descriptor = GMPOrderNorm(MCSHs = GMPs, elements = elements)
descriptor_setup = ('gmpordernorm', GMPs, GMPs.get('cutoff_distance'), elements)

a2d = AtomsToData(
    descriptor = descriptor,
    r_energy = False,
    r_forces = False,
    save_fps = False,
    fprimes = False)

do = a2d.convert(mof, ref_positions = positions, idx = 0)
fps = do.fingerprint.numpy()
descriptors = np.concatenate((positions, fps), axis = 1)

path = './GI'

if not os.path.isdir(path):
    os.makedirs(path)

with open('{}/{}.npy'.format(path, sys.argv[1][:-4]), 'wb') as f:
    np.save(f, descriptors)