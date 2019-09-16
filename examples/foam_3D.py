import os

import matplotlib.pyplot as plt
import microstructpy as msp
import numpy as np
import scipy.stats
from mpl_toolkits.mplot3d import Axes3D


def main():

    # Create Directory
    dirname = os.path.join(os.path.dirname(__file__), 'foam_3D')
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Define Domain
    domain = msp.geometry.Cube(side_length=10)

    # Create Void Tessellation
    void_mat = {'material_type': 'void',
                'shape': 'sphere',
                'size': 2
                }

    void_a = 0.57 * domain.volume
    void_seeds = msp.seeding.SeedList.from_info(void_mat, void_a)
    void_seeds.position(domain, rtol=0.1, verbose=True)
    void_tess = msp.meshing.PolyMesh.from_seeds(void_seeds, domain)

    # Add Foam to Edges in Void Tessellation
    foam_mat = {'material_type': 'amorphous',
                'shape': 'sphere',
                'size': 0.2
                }
    foam_r1 = 0.10
    foam_r2 = 0.05

    edge_seeds = {}
    kps_w_seeds = set()
    ext_kps = set()
    for neighs, facet in zip(void_tess.facet_neighbors, void_tess.facets):
        if np.min(neighs) < 0:
            ext_kps |= set(facet)


    for neighs, facet in zip(void_tess.facet_neighbors, void_tess.facets):
        if np.min(neighs) < 0:
            continue
        for i, kp_end in enumerate(facet):
            kp_beg = facet[i - 1]
            if kp_beg < kp_end:
                edge = (kp_beg, kp_end)
            else:
                edge = (kp_end, kp_beg)
            if edge in edge_seeds:
                continue
            if edge[0] in ext_kps or edge[1] in ext_kps:
                continue
                    
            pt1 = np.array(void_tess.points[edge[0]])
            pt2 = np.array(void_tess.points[edge[1]])
            edge_len = np.linalg.norm(pt2 - pt1)
            x, r = seed_edge(foam_r1, foam_r2, edge_len)

            if edge[0] in kps_w_seeds:
                x = x[1:]
                r = r[1:]
            if edge[1] in kps_w_seeds:
                x = x[:-1]
                r = r[:-1]
            kps_w_seeds |= set(edge)

            seeds = []
            for xi, ri in zip(x, r):
                f = xi / edge_len
                cen = (1 - f) * pt1 + f * pt2
                geom = msp.geometry.Sphere(r=ri, center=cen)
                seed = msp.seeding.Seed(geom, phase=1, position=cen)
                seeds.append(seed)
            edge_seeds[edge] = seeds

    all_edge_seeds = [s for edge in edge_seeds for s in edge_seeds[edge]]
    foam_seeds = msp.seeding.SeedList(all_edge_seeds)

    # Combine Results
    for seed in void_seeds:
        seed.geometry.r *= 1 - 0.0

    materials = [void_mat, foam_mat]
    seeds = void_seeds + foam_seeds
    print('polygonal meshing')
    pmesh = msp.meshing.PolyMesh.from_seeds(seeds, domain)

    # Triangular Mesh
    print('triangular meshing')
    tmesh = msp.meshing.TriMesh.from_polymesh(pmesh, materials)

    # Plot
    print('plotting')
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax._axis3don = False

    tmesh.plot(facecolor='aquamarine',
               edgecolor='gray',
               linewidth=0.1)

    lims = domain.limits
    for x in lims[0]:
        for y in lims[1]:
            for z in lims[2]:
                ax.plot([x], [y], [z])
    ax.set_aspect('equal')

    fname = os.path.join(dirname, 'trimesh.png')
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)

    # Remove Whitespace from Plot
    foam_image = plt.imread(fname)
    c_min = foam_image[:,:,:-1].min(axis=-1)
    min0 = c_min.min(axis=0)
    mask0 = min0 < 1
    ax0 = np.nonzero(mask0)[0][[0, -1]]
    
    min1 = c_min.min(axis=1)
    mask1 = min1 < 1
    ax1 = np.nonzero(mask1)[0][[0, -1]]

    cr_image = foam_image[ax1[0]:ax1[1], ax0[0]:ax0[1]]
    plt.imsave(fname.replace('.png', '_cropped.png'), cr_image)


def seed_edge(r1, r2, len_x):
    x_max = 0.5 * len_x
    xs = []
    rs = []
    x = 0
    r = r1
    while x + r < x_max:
        xs.append(x)
        rs.append(r)

        x += r * 1.05
        r = (r1 - r2) * np.exp(-x / r1) + r2
    x_arr = np.array(xs)
    r_arr = np.array(rs)

    xs = np.concatenate((x_arr, len_x - np.flip(x_arr)))
    rs = np.concatenate((r_arr, np.flip(r_arr)))

    if len(xs) > 0 and x_max - x_arr[-1] > r2:
        xs = np.insert(xs, -1, x_max)
        rs = np.insert(rs, -1, r2)

    return xs, rs


if __name__ == '__main__':
    main()
