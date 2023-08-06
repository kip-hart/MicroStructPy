import os

import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

import microstructpy as msp


def main():
    # Create Directory
    dirname = os.path.join(os.path.dirname(__file__), 'foam')
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Define Domain
    domain = msp.geometry.Square(side_length=8)

    # Create Void Tessellation
    void_mat = {'material_type': 'void',
                'shape': 'circle',
                'size': scipy.stats.lognorm(scale=1, s=0.2)
                }

    void_a = 0.7 * domain.area
    void_seeds = msp.seeding.SeedList.from_info(void_mat, void_a)
    void_seeds.position(domain, rtol=0.03, verbose=True)
    void_tess = msp.meshing.PolyMesh.from_seeds(void_seeds, domain)

    # Add Foam
    foam_mat = {'material_type': 'amorphous',
                'shape': 'circle',
                'size': scipy.stats.lognorm(scale=0.15, s=0.1)
                }

    foam_a = 0.15 * domain.area
    foam_seeds = msp.seeding.SeedList.from_info(foam_mat, foam_a)
    inds = np.flip(np.argsort([s.volume for s in foam_seeds]))
    foam_seeds = foam_seeds[inds]

    bkdwns = np.array([s.breakdown[0] for s in foam_seeds])
    np.random.seed(0)
    for i, seed in enumerate(foam_seeds):
        if i == 0:
            trial_pt = trial_position(void_tess)
        else:
            r = seed.geometry.r
            check_bkdwns = bkdwns[:i]
            good_pt = False
            while not good_pt:
                trial_pt = trial_position(void_tess)
                good_pt = check_pt(trial_pt, r, check_bkdwns)

        seed.position = trial_pt
        bkdwns[i] = seed.breakdown
        seed.phase = 1

    # Combine Results
    materials = [void_mat, foam_mat]
    seeds = void_seeds + foam_seeds
    pmesh = msp.meshing.PolyMesh.from_seeds(seeds, domain)

    # Triangular Mesh
    tmesh = msp.meshing.TriMesh.from_polymesh(pmesh,
                                              materials,
                                              min_angle=20,
                                              max_edge_length=0.1)

    # Plot
    tmesh.plot(facecolor='aquamarine',
               edgecolor='k',
               linewidth=0.2)

    plt.gca().set_position([0, 0, 1, 1])
    plt.axis('image')
    plt.gca().set_axis_off()
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)

    xlim, ylim = domain.limits
    plt.axis([xlim[0], xlim[1], ylim[0], ylim[1]])

    for ext in ['png', 'pdf']:
        fname = os.path.join(dirname, 'trimesh.' + ext)
        plt.savefig(fname, bbox_inches='tight', pad_inches=0)


def pick_edge(void_tess):
    f_neighs = void_tess.facet_neighbors
    i = -1
    neighs = [-1, -1]
    while any([n < 0 for n in neighs]):
        i = np.random.randint(len(f_neighs))
        neighs = f_neighs[i]
    facet = void_tess.facets[i]
    j = np.random.randint(len(facet))
    kp1 = facet[j]
    kp2 = facet[j - 1]
    return kp1, kp2


def trial_position(void_tess):
    kp1, kp2 = pick_edge(void_tess)
    pt1 = void_tess.points[kp1]
    pt2 = void_tess.points[kp2]

    f = np.random.rand()
    return [f * x1 + (1 - f) * x2 for x1, x2 in zip(pt1, pt2)]


def check_pt(point, r, breakdowns):
    pts = breakdowns[:, :-1]
    rads = breakdowns[:, -1]

    rel_pos = pts - point
    dist = np.sqrt(np.sum(rel_pos * rel_pos, axis=1))
    min_dist = rads + r - 0.3 * np.minimum(rads, r)
    return np.all(dist > min_dist)


if __name__ == '__main__':
    main()
