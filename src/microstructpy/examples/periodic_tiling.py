from __future__ import division

import os

import numpy as np
import scipy.stats
from matplotlib import collections
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import microstructpy as msp

# ------------------------------------------------------------------------ #
#                                                                          #
# 2D: a periodic microstructure and its 2 x 2 tiling                       #
#                                                                          #
# ------------------------------------------------------------------------ #

# Create domain
domain_2d = msp.geometry.Square(side_length=2, corner=(0, 0))

# Create phases: a matrix of circular grains and elliptical inclusions
phases_2d = [
    {'shape': 'circle', 'size': scipy.stats.uniform(loc=0.2, scale=0.2),
     'material_type': 'crystalline', 'fraction': 2},
    {'shape': 'ellipse', 'size': scipy.stats.uniform(loc=0.25, scale=0.15),
     'aspect_ratio': 2, 'angle_deg': scipy.stats.uniform(loc=0, scale=180),
     'material_type': 'crystalline', 'fraction': 1},
]

# Create seeds and position them, periodic in x and y (the seeds fill 90%
# of the area, so that all of them can be placed). The margin keeps the
# seeds from ending within half a target edge length of a periodic face,
# or crossing one by less, which would leave thin pieces of grains on the
# opposite face and very small triangles there. The smallest grain needs
# about four elements across it, so the margin is at most an eighth of its
# smallest diameter (the CLI setting periodic_margin = auto does the same).
max_volume = 0.004
h_target = np.sqrt(4 * max_volume / np.sqrt(3))
seeds_2d = msp.seeding.SeedList.from_info(phases_2d, 0.9 * domain_2d.area,
                                          rng_seeds={'size': 1})
d_min = min([2 * min(getattr(s.geometry, 'axes', (s.geometry.size / 2,)))
             for s in seeds_2d])
margin = min(0.5 * h_target, d_min / 8)
seeds_2d.position(domain_2d, rng_seed=1, periodic=True,
                  periodic_margin=margin)

# Create the polygonal and triangular meshes. The edge optimization moves
# the seeds slightly to remove the shortest edges of the polygonal mesh,
# which would otherwise force very small triangles in the mesh; with the
# margin, it also thickens or removes the pieces of the grains at the
# periodic faces that are thinner than the margin, and opens the corners
# of the grains at the faces that are narrower than the minimum angle of
# the triangles, which the mesher would fill with very small triangles.
min_angle = 25
pmesh_2d = msp.meshing.PolyMesh.from_seeds(seeds_2d, domain_2d,
                                           periodic=True, edge_opt=True,
                                           n_iter=25, periodic_margin=margin,
                                           min_angle=min_angle)
tmesh_2d = msp.meshing.TriMesh.from_polymesh(pmesh_2d, phases_2d,
                                             min_angle=min_angle,
                                             max_volume=max_volume)

# Plot the tiled polygonal mesh, with each grain in one color, and the
# tiled triangular mesh, with the matching nodes on the periodic faces
pts = np.array(pmesh_2d.points)
cmap = plt.get_cmap('tab20')
colors = [cmap(s % 20) for s in pmesh_2d.seed_numbers]
tile_x, tile_y = domain_2d.side_length, domain_2d.side_length
offsets = [(i * tile_x, j * tile_y) for i in (0, 1) for j in (0, 1)]


def cell_polygon(region):
    # the points of a (convex) cell, in order around its center
    kps = sorted(set([kp for f in region for kp in pmesh_2d.facets[f]]))
    center = pts[kps].mean(axis=0)
    angles = np.arctan2(pts[kps, 1] - center[1], pts[kps, 0] - center[0])
    return pts[np.array(kps)[np.argsort(angles)]]


# the cells, and the grain boundaries (an elliptical grain is made of
# several cells, whose common facets are not grain boundaries)
polys = [cell_polygon(region) for region in pmesh_2d.regions]
seed_nums = np.array(pmesh_2d.seed_numbers)
boundaries = []
for facet, (n_1, n_2) in zip(pmesh_2d.facets, pmesh_2d.facet_neighbors):
    if min(n_1, n_2) < 0 or seed_nums[n_1] != seed_nums[n_2]:
        boundaries.append(pts[facet])
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for offset in offsets:
    axes[0].add_collection(collections.PolyCollection(
        [poly + offset for poly in polys], facecolors=colors,
        edgecolors=colors, linewidths=0.5))
    axes[0].add_collection(collections.LineCollection(
        [line + offset for line in boundaries], colors='k', linewidths=0.6))
axes[0].set_title('Polygonal mesh, tiled 2 x 2')

t_pts = np.array(tmesh_2d.points)
t_elems = np.array(tmesh_2d.elements)
for offset in offsets:
    axes[1].triplot(t_pts[:, 0] + offset[0], t_pts[:, 1] + offset[1],
                    t_elems, color='0.3', linewidth=0.25)
lows = [lo for lo, hi in tmesh_2d.periodic_nodes[0]]
highs = [hi for lo, hi in tmesh_2d.periodic_nodes[0]]
axes[1].plot(t_pts[lows, 0], t_pts[lows, 1], 'r.', markersize=4,
             label='nodes on x = 0')
axes[1].plot(t_pts[highs, 0], t_pts[highs, 1], 'b.', markersize=4,
             label='their images on x = 2')
axes[1].legend(loc='upper right')
axes[1].set_title('Triangular mesh, tiled 2 x 2')

for ax in axes:
    ax.axvline(tile_x, color='w', linewidth=1.2)
    ax.axhline(tile_y, color='w', linewidth=1.2)
    ax.set_aspect('equal')
    ax.set_xlim(0, 2 * tile_x)
    ax.set_ylim(0, 2 * tile_y)

file_dir = os.path.dirname(os.path.realpath(__file__))
out_dir = os.path.join(file_dir, 'periodic_tiling')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
plt.savefig(os.path.join(out_dir, 'tiled_2D.png'), bbox_inches='tight',
            pad_inches=0.1)
plt.close(fig)

# ------------------------------------------------------------------------ #
#                                                                          #
# 3D: a periodic microstructure and its 2 x 1 x 1 tiling                   #
#                                                                          #
# ------------------------------------------------------------------------ #

# Create domain
side = 4
domain_3d = msp.geometry.Cube(side_length=side, corner=(0, 0, 0))

# Create phases
phases_3d = [
    {'shape': 'sphere', 'size': scipy.stats.uniform(loc=0.8, scale=0.6),
     'fraction': 1},
    {'shape': 'sphere', 'size': 1.2, 'fraction': 1},
]

# Create seeds, position them and mesh the domain, periodic in x, y and z
# (the seeds fill 90% of the volume, so that all of them can be placed)
seeds_3d = msp.seeding.SeedList.from_info(phases_3d, 0.9 * domain_3d.volume,
                                          rng_seeds={'size': 2})
seeds_3d.position(domain_3d, rng_seed=2, periodic=True)
pmesh_3d = msp.meshing.PolyMesh.from_seeds(seeds_3d, domain_3d,
                                           periodic=True)
tmesh_3d = msp.meshing.TriMesh.from_polymesh(pmesh_3d, phases_3d,
                                             min_angle=15, max_volume=0.02)

# The nodes on opposite faces of the domain are images of each other
for axis, pairs in sorted(tmesh_3d.periodic_nodes.items()):
    print('axis ' + 'xyz'[axis] + ': ' + str(len(pairs)) + ' pairs of nodes')

# Plot the faces of the polyhedral mesh that are visible from the viewpoint
# (y = 0, x = side and z = side), for the domain and for a copy translated
# along x: the grains continue across the periodic face
pts = np.array(pmesh_3d.points)
colors = [cmap(s % 20) for s in pmesh_3d.seed_numbers]
visible = {-3: 0.85, -2: 0.7, -6: 1.0}  # wall number: shading


def wall_polygons(offset, walls):
    polys, facecolors = [], []
    for facet, neighs in zip(pmesh_3d.facets, pmesh_3d.facet_neighbors):
        wall = min(neighs)
        if wall in walls:
            region = max(neighs)
            polys.append(pts[facet] + offset)
            facecolors.append(np.array(colors[region]) * walls[wall])
    return polys, facecolors


fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(projection='3d')
ax.set_position([0, 0, 1, 1])
polys, facecolors = wall_polygons(np.zeros(3), {-3: 0.85, -6: 1.0})
polys_2, facecolors_2 = wall_polygons(np.array([side, 0, 0]), visible)
ax.add_collection3d(Poly3DCollection(polys + polys_2,
                                     facecolors=facecolors + facecolors_2,
                                     edgecolors='k', linewidths=0.3))
ax.set_xlim(0, 2 * side)
ax.set_ylim(0, side)
ax.set_zlim(0, side)
ax.set_box_aspect((2, 1, 1))
ax.view_init(elev=22, azim=-55)
ax.set_title('Polyhedral mesh, tiled twice along x')
plt.savefig(os.path.join(out_dir, 'tiled_3D.png'), bbox_inches='tight',
            pad_inches=0.1)
