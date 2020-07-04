import os

from matplotlib import pyplot as plt

import microstructpy as msp

# Create domain
domain = msp.geometry.Square()

# Create list of seed points
factory = msp.seeding.Seed.factory
n = 50
seeds = msp.seeding.SeedList([factory('circle', r=0.01) for i in range(n)])
seeds.position(domain)

# Create Voronoi diagram
pmesh = msp.meshing.PolyMesh.from_seeds(seeds, domain)

# Plot Voronoi diagram and seed points
pmesh.plot(edgecolors='k', facecolors='gray')
seeds.plot(edgecolors='k', facecolors='none')

plt.axis('square')
plt.xlim(domain.limits[0])
plt.ylim(domain.limits[1])

file_dir = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(file_dir, 'standard_voronoi/voronoi_diagram.png')
dirs = os.path.dirname(filename)
if not os.path.exists(dirs):
    os.makedirs(dirs)
plt.savefig(filename, bbox_inches='tight', pad_inches=0)
