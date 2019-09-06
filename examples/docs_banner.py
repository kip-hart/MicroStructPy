from __future__ import division

import os

import matplotlib.pyplot as plt
import microstructpy as msp
import numpy as np
import scipy.stats


def main():
    # Colors
    c1 = '#12C2E9'
    c2 = '#C471ED'
    c3 = '#F64F59'

    # Offset
    off = 1

    # Create Directory
    dirname = os.path.join(os.path.dirname(__file__), 'docs_banner')
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Create Domain
    domain = msp.geometry.Rectangle(width=10, length=20)

    # Create Unpositioned Seeds
    phase2 = {'color': c1}
    ell_geom = msp.geometry.Ellipse(a=8, b=3)
    ell_seed = msp.seeding.Seed(ell_geom, phase=2)

    mu = 1
    bnd = 0.5
    d_dist = scipy.stats.uniform(loc=mu-bnd, scale=2*bnd)
    phase0 = {'color': c2, 'shape': 'circle', 'd': d_dist}
    phase1 = {'color': c3, 'shape': 'circle', 'd': d_dist}
    circle_area = domain.area - ell_geom.area
    seeds = msp.seeding.SeedList.from_info([phase0, phase1], circle_area)

    seeds.append(ell_seed)
    hold = [False for seed in seeds]
    hold[-1] = True
    phases = [phase0, phase1, phase2]

    # Create Positioned Seeds
    seeds.position(domain, hold=hold, verbose=True)

    # Create Polygonal Mesh
    pmesh = msp.meshing.PolyMesh.from_seeds(seeds, domain)

    # Create Triangular Mesh
    tmesh = msp.meshing.TriMesh.from_polymesh(pmesh,
                                              min_angle=12,
                                              max_edge_length=0.2,
                                              max_volume=0.4)

    # Create Figure
    k = 0.2
    len_x = 3 * domain.length + 4 * off
    len_y = domain.width + 2 * off
    plt.figure(figsize=(k * len_x, k * len_y))

    # Plot Seeds 
    seed_colors = [phases[s.phase]['color'] for s in seeds]
    seeds.plot(color=seed_colors, alpha=0.8, edgecolor='k', linewidth=0.5)
    domain.plot(facecolor='none', edgecolor='k', linewidth=0.5)

    # Plot Polygonal Mesh
    pmesh.points = np.array(pmesh.points)
    pmesh.points[:, 0] += domain.length + off
    for region, phase_num in zip(pmesh.regions, pmesh.phase_numbers):
        if phase_num == 2:
            continue
        color = phases[phase_num]['color']

        facets = [pmesh.facets[f] for f in region]
        kps = ordered_kps(facets)
        x, y = zip(*[pmesh.points[kp] for kp in kps])
        plt.fill(x, y, color=color, alpha=0.8, edgecolor='none')

    ellipse_regions = set()
    for region_num, phase_num in enumerate(pmesh.phase_numbers):
        if phase_num == 2:
            ellipse_regions.add(region_num)

    ellipse_facets = []
    for facet, neighbors in zip(pmesh.facets, pmesh.facet_neighbors):
        common_regions = ellipse_regions & set(neighbors)
        if len(common_regions) == 1:
            ellipse_facets.append(facet)
    ellipse_kps = ordered_kps(ellipse_facets)
    x, y = zip(*[pmesh.points[kp] for kp in ellipse_kps])
    plt.fill(x, y, color=phases[-1]['color'], alpha=0.8, edgecolor='none')

    for facet, neighbors in zip(pmesh.facets, pmesh.facet_neighbors):
        common_regions = ellipse_regions & set(neighbors)
        if len(common_regions) < 2:
            x, y = zip(*[pmesh.points[kp] for kp in facet])
            plt.plot(x, y, color='k', linewidth=0.5)

    # Plot Triangular Mesh
    tmesh.points = np.array(tmesh.points)
    tmesh.points[:, 0] += 2 * off + 2 * domain.length
    tri_colors = [seed_colors[n] for n in tmesh.element_attributes]
    tmesh.plot(color=tri_colors, alpha=0.8, edgecolor='k', linewidth=0.3)

    # Set Up Axes
    plt.gca().set_position([0, 0, 1, 1])
    plt.axis('image')
    plt.axis('off')
    
    xlim, ylim = domain.limits
    xlim[0] -= off
    xlim[1] += 3 * off + 2 * domain.length

    ylim[0] -= off
    ylim[1] += off

    plt.axis(list(xlim) + list(ylim))

    fname = os.path.join(dirname, 'banner.png')
    plt.savefig(fname, bbox='tight', pad_inches=0)


def plot_seeds(seeds, phases, domain):
    plt.clf()
    colors = []
    for seed in seeds:
        colors.append(phases[seed.phase]['color'])
    seeds.plot(color=colors, alpha=0.8, edgecolor='none')
    seeds.plot(color='none', alpha=0.8, edgecolor='k', linewidth=0.1)

    show_plot(domain, 'seeds.pdf')


def plot_seeds_breakdown(seeds, phases, domain):
    plt.clf()
    colors = []
    for seed in seeds:
        colors.append(phases[seed.phase]['color'])
    seeds.plot_breakdown(color=colors, alpha=0.8, edgecolor='none')
    seeds.plot_breakdown(color='none', alpha=0.8, edgecolor='k', linewidth=0.1)

    show_plot(domain, 'breakdown.pdf')

def plot_polymesh(pmesh, phases, domain):
    plt.clf()
    colors = []
    for phase_num in pmesh.phase_numbers:
        colors.append(phases[phase_num]['color'])
    pmesh.plot(color=colors, alpha=0.8, edgecolor='none')
    pmesh.plot(color='none', alpha=0.8, edgecolor='k', linewidth=0.1)

    show_plot(domain, 'polymesh.pdf')

def plot_polymesh_without(pmesh, phases, domain):
    plt.clf()
    pmesh_facets = np.array(pmesh.facets)

    # Plot Surrounding
    for region, phase_num in zip(pmesh.regions, pmesh.phase_numbers):
        if phase_num == 1:
            continue
        color = phases[0]['color']

        facets = [pmesh.facets[f] for f in region]
        kps = ordered_kps(facets)
        x, y = zip(*[pmesh.points[kp] for kp in kps])
        plt.fill(x, y, color=color, alpha=0.8, edgecolor='none')

    # Plot Ellipse
    ellipse_regions = set()
    for region_num, phase_num in enumerate(pmesh.phase_numbers):
        if phase_num == 1:
            ellipse_regions.add(region_num)

    ellipse_facets = []
    for facet, neighbors in zip(pmesh.facets, pmesh.facet_neighbors):
        common_regions = ellipse_regions & set(neighbors)
        if len(common_regions) == 1:
            ellipse_facets.append(facet)
    ellipse_kps = ordered_kps(ellipse_facets)
    x, y = zip(*[pmesh.points[kp] for kp in ellipse_kps])
    plt.fill(x, y, color=phases[1]['color'], alpha=0.8, edgecolor='k',
             linewidth=0.1)

    # Plot Facets
    for facet, neighbors in zip(pmesh.facets, pmesh.facet_neighbors):
        common_regions = ellipse_regions & set(neighbors)
        if len(common_regions) < 2:
            x, y = zip(*[pmesh.points[kp] for kp in facet])
            plt.plot(x, y, color='k', linewidth=0.1)

    show_plot(domain, 'polymesh_without.pdf')


def plot_trimesh(tmesh, seeds, phases, domain):
    plt.clf()
    colors = []
    for seed_num in tmesh.element_attributes:
        phase_num = seeds[seed_num].phase
        color = phases[phase_num]['color']
        colors.append(color)
    tmesh.plot(color=colors, alpha=0.8, edgecolor='k', linewidth=0.1)

    show_plot(domain, 'trimesh.pdf')


def ordered_kps(pairs):
    t_pairs = [tuple(p) for p in pairs]
    kps = list(t_pairs.pop())
    while t_pairs:
        for i, pair in enumerate(t_pairs):
            if kps[-1] in pair:
                break
        assert kps[-1] in pair, pairs
        kps += [kp for kp in t_pairs.pop(i) if kp != kps[-1]]
    return kps[:-1]


def show_plot(domain, fname=None):
    plt.axis('square')

    xlim, ylim = domain.limits
    plt.xlim(xlim)
    plt.ylim(ylim)

    ax = plt.gca()
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    if not fname:
        plt.show()
    else:
        plt.savefig(fname, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    main()
