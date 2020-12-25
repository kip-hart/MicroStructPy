from __future__ import division

import os

import numpy as np
from matplotlib import collections
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.offsetbox import AnnotationBbox
from matplotlib.offsetbox import OffsetImage

import microstructpy as msp


def main(n_seeds, size_rng, pos_rng, k_lw):
    bkgrnd_color = 'black'
    line_color = (1, 1, 1, 1)  # white

    dpi = 300
    init_size = 2000
    logo_size = 1500
    favicon_size = 48

    logo_basename = 'logo.svg'
    favicon_basename = 'favicon.ico'
    social_basename = 'social.png'
    file_dir = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(file_dir, 'logo')
    if not os.path.exists(path):
        os.makedirs(path)
    logo_filename = os.path.join(path, logo_basename)
    pad_filename = os.path.join(path, 'pad_' + logo_basename)
    favicon_filename = os.path.join(path, favicon_basename)
    social_filename = os.path.join(path, social_basename)

    # Set Domain
    domain = msp.geometry.Circle()

    # Set Seed List
    np.random.seed(size_rng)
    rs = 0.3 * np.random.rand(n_seeds)

    factory = msp.seeding.Seed.factory
    seeds = msp.seeding.SeedList([factory('circle', r=r) for r in rs])
    seeds.position(domain, rng_seed=pos_rng)

    # Create the Poly Mesh
    pmesh = msp.meshing.PolyMesh.from_seeds(seeds, domain)

    # Create and Format the Figure
    plt.clf()
    plt.close('all')
    fig = plt.figure(figsize=(init_size / dpi, init_size / dpi), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.add_axes(ax)

    # Plot the Domain
    domain.plot(ec='none', fc=bkgrnd_color)

    # Plot the Facets
    facet_colors = []
    for neigh_pair in pmesh.facet_neighbors:
        if min(neigh_pair) < 0:
            facet_colors.append((0, 0, 0, 0))
        else:
            facet_colors.append(line_color)

    lw = k_lw * init_size / 100
    pmesh.plot_facets(index_by='facet', colors=facet_colors,
                      linewidth=lw, capstyle='round')

    pts = np.array(pmesh.points)
    rs = np.sqrt(np.sum(pts * pts, axis=1))
    mask = np.isclose(rs, 1)

    edges = []
    for facet in pmesh.facets:
        if np.sum(mask[facet]) != 1:
            continue

        edge = np.copy(pts[facet])
        if mask[facet[0]]:
            u = edge[0] - edge[1]
            u *= 1.1
            edge[0] = edge[1] + u
        else:
            u = edge[1] - edge[0]
            u *= 1.1
            edge[1] = edge[0] + u
        edges.append(edge)

    pc = collections.LineCollection(edges, color=line_color, linewidth=lw,
                                    capstyle='round')
    ax.add_collection(pc)

    # Format the Plot and Convert to Image Array
    plt.axis('square')
    plt.axis(1.01 * np.array([-1, 1, -1, 1]))
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    plt_im = np.array(canvas.buffer_rgba())
    mask = plt_im[:, :, 0] > 0.5 * 255

    # Create the Logo
    logo_im = np.copy(plt_im)

    xx, yy = np.meshgrid(*[np.arange(n) for n in logo_im.shape[:2]])
    zz = - 0.2 * xx + 0.9 * yy
    ss = (zz - zz.min()) / (zz.max() - zz.min())

    c1 = [67, 206, 162]
    c2 = [24, 90, 157]

    logo_im[mask, -1] = 0  # transparent background

    # gradient
    for i in range(logo_im.shape[-1] - 1):
        logo_im[~mask, i] = (1 - ss[~mask]) * c1[i] + ss[~mask] * c2[i]

    inds = np.linspace(0, logo_im.shape[0] - 1, logo_size).astype('int')
    logo_im = logo_im[inds]
    logo_im = logo_im[:, inds]

    pad_w = logo_im.shape[0]
    pad_h = 0.5 * logo_im.shape[1]
    pad_shape = np.array([pad_w, pad_h, logo_im.shape[2]]).astype('int')
    logo_pad = np.zeros(pad_shape, dtype=logo_im.dtype)
    pad_im = np.concatenate((logo_pad, logo_im, logo_pad), axis=1)
    doc_im = np.concatenate((logo_pad, pad_im, logo_pad), axis=1)

    plt.imsave(logo_filename, logo_im, dpi=dpi)
    plt.imsave(logo_filename.replace('.svg', '.png'), np.ascontiguousarray(logo_im), dpi=dpi)
    plt.imsave(pad_filename, pad_im, dpi=dpi)
    plt.imsave(pad_filename.replace('.svg', '.png'), np.ascontiguousarray(doc_im), dpi=dpi)

    # Create the Favicon
    fav_im = np.copy(logo_im)
    inds = np.linspace(0, fav_im.shape[0] - 1, favicon_size).astype('int')
    fav_im = fav_im[inds]
    fav_im = fav_im[:, inds]

    plt.imsave(favicon_filename, np.ascontiguousarray(fav_im), dpi=dpi, format='png')

    # Create the Social Banner
    fig_social, ax_social = plt.subplots()

    ax_social.set_xlim(0, 2)
    ax_social.set_ylim(0, 1)
    ax_social.set_aspect('equal')

    ax_social.set_axis_off()
    ax_social.get_xaxis().set_visible(False)
    ax_social.get_yaxis().set_visible(False)

    imagebox = OffsetImage(logo_im, zoom=0.05)
    ab = AnnotationBbox(imagebox, (1, 0.7), frameon=False)
    ax_social.add_artist(ab)
    ax_social.text(1, 0.35, 'MicroStructPy',
                   fontsize=20,
                   weight='bold',
                   horizontalalignment='center',
                   verticalalignment='center')
    ax_social.text(1, 0.23, 'Microstructure Mesh Generation in Python',
                   fontsize=10,
                   horizontalalignment='center',
                   verticalalignment='center')
    plt.draw()
    plt.savefig(social_filename, bbox_inches='tight')
    plt.close('all')


if __name__ == '__main__':
    n_seeds = 14
    size_rng = 4
    pos_rng = 7
    k_lw = 1.1

    main(n_seeds, size_rng, pos_rng, k_lw)
