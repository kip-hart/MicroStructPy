from __future__ import division

import os

import numpy as np
import scipy.integrate
import scipy.stats
from matplotlib import pyplot as plt

import microstructpy as msp

# Define the domain
domain = msp.geometry.Square(corner=(0, 0), side_length=10)

# Define the material phases
a_dist = scipy.stats.lognorm(s=1, scale=0.1)
matrix_phase = {'fraction': 1,
                'material_type': 'matrix',
                'shape': 'circle',
                'area': a_dist}

neighborhood_phase = {'fraction': 1,
                      'material_type': 'solid',
                      'shape': 'ellipse',
                      'a': 1.5,
                      'b': 0.6,
                      'angle_deg': scipy.stats.uniform(0, 360)}

phases = [matrix_phase, neighborhood_phase]

# Create the seed list
seeds = msp.seeding.SeedList.from_info(phases, domain.area)
seeds.position(domain)

# Replace the neighborhood phase with materials
a = neighborhood_phase['a']
b = neighborhood_phase['b']
r = b / 3
n = 16

t_perim = np.linspace(0, 2 * np.pi, 201)
x_perim = (a - r) * np.cos(t_perim)
y_perim = (b - r) * np.sin(t_perim)
dx = np.insert(np.diff(x_perim), 0, 0)
dy = np.insert(np.diff(y_perim), 0, 0)
ds = np.sqrt(dx * dx + dy * dy)
arc_len = scipy.integrate.cumtrapz(ds, x=t_perim, initial=0)
eq_spaced = arc_len[-1] * np.arange(n) / n
x_pts = np.interp(eq_spaced, arc_len, x_perim)
y_pts = np.interp(eq_spaced, arc_len, y_perim)

repl_seeds = msp.seeding.SeedList()
geom = {'a': a - 2 * r, 'b': b - 2 * r}
for sn, seed in enumerate(seeds):
    if seed.phase == 0:
        repl_seeds.append(seed)
    else:
        center = seed.position
        theta = seed.geometry.angle_rad

        geom['angle_rad'] = theta
        geom['center'] = center
        core_seed = msp.seeding.Seed.factory('ellipse', phase=3,
                                             position=seed.position, **geom)
        repl_seeds.append(core_seed)

        x_ring = center[0] + x_pts * np.cos(theta) - y_pts * np.sin(theta)
        y_ring = center[1] + x_pts * np.sin(theta) + y_pts * np.cos(theta)
        for i in range(n):
            phase = 1 + (i % 2)
            center = (x_ring[i], y_ring[i])
            ring_geom = {'center': center, 'r': r}
            ring_seed = msp.seeding.Seed.factory('circle', position=center,
                                                 phase=phase, **ring_geom)
            if domain.within(center):
                repl_seeds.append(ring_seed)

# Create polygon and triangle meshes
pmesh = msp.meshing.PolyMesh.from_seeds(repl_seeds, domain)
phases = [{'material_type': 'solid'} for i in range(4)]
phases[0]['material_type'] = 'matrix'
tmesh = msp.meshing.TriMesh.from_polymesh(pmesh, phases, min_angle=20,
                                          max_volume=0.1)

# Plot triangle mesh
colors = ['C' + str(repl_seeds[att].phase) for att in tmesh.element_attributes]
tmesh.plot(facecolors=colors, edgecolors='k', linewidth=0.2)

plt.axis('square')
plt.xlim(domain.limits[0])
plt.ylim(domain.limits[1])

file_dir = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(file_dir, 'grain_neighborhoods/trimesh.png')
dirs = os.path.dirname(filename)
if not os.path.exists(dirs):
    os.makedirs(dirs)
plt.savefig(filename, bbox_inches='tight', pad_inches=0)
