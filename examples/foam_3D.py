import os

import matplotlib.pyplot as plt
import microstructpy as msp
import numpy as np
import scipy.stats


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

    void_a = 0.4 * domain.volume
    void_seeds = msp.seeding.SeedList.from_info(void_mat, void_a)
    void_seeds.position(domain, rtol=0, verbose=True)
    void_tess = msp.meshing.PolyMesh.from_seeds(void_seeds, domain)

    # Add Foam
    foam_mat = {'material_type': 'amorphous',
                'shape': 'sphere',
                'size': 0.2
                }
    foam_r1 = 0.1
    foam_r2 = 0.05

    edge_seeds = {}
    kps_w_seeds = set()
    for neighs, facet in zip(void_tess.facet_neighbors, void_tess.facets):
        if np.min(neighs) < 0:
            continue
        for i, kp_end in enumeratet(facet):
            kp_beg = facet[i - 1]
            if kp_beg < kp_end:
                edge = (kp_beg, kp_end)
            else:
                edge = (kp_end, kp_beg)
            if edge in edge_seeds:
                continue
                    
            pt1 = np.array(void_tess.points[edge[0]])
            pt2 = np.array(void_tess.points[edge[1]])
            edge_len = np.linalg.norm(pt2 - pt1)
            x, r = seed_edge(foam_r1, foam_r2, edge_len)

            # TODO: continue adding foam seeds

            if edge_len < foam_r1:
                edge_seeds[edge] = msp.seeding.SeedList([])
                continue

            n_pts = int(edge_len / foam_r1) + 1
            f = np.linspace()


    foam_seeds = [msp.seeding.Seed(msp.geometry.Sphere(r=foam_r1, phase=1)) for
             pt in void_tess.points]
    for s, x in zip(foam_seeds, void_tess.points):
        s.position = x

    # Combine Results
    for seed in void_seeds:
        seed.geometry.r *= 0.7

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

    plt.show()
    '''

    plt.gca().set_position([0, 0, 1, 1])
    plt.axis('image')
    plt.gca().set_axis_off()
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)

    xlim, ylim = domain.limits
    plt.axis([xlim[0], xlim[1], ylim[0], ylim[1]])

    for ext in ['png', 'pdf']:
        fname = os.path.join(dirname, 'trimesh.' + ext)
        plt.savefig(fname)
    '''


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

    if len(xs) > 0:
        xs = np.append(xs, x_max)
        rs = np.append(rs, r2)

    return xs, rs
'''
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
    min_dist = rads + r - 0.2 * np.minimum(rads, r)
    return np.all(dist > min_dist)
'''


if __name__ == '__main__':
    main()
