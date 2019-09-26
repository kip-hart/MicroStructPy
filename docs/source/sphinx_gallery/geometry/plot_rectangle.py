"""
Plot figures relevant to :class:`microstructpy.geometry.Rectangle`

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections

import microstructpy as msp

def main():
    # Plot breakdown
    breakdown(2.5, 1, 0.3, (9, 4))  # rectangle_001.png - rect breakdown
    breakdown(1, 1, 0.2, (4, 4))  # rectangle_001.png - square breakdown


def breakdown(length, width, x1, figsize):
    r = msp.geometry.Rectangle(length=length, width=width)
    approx = r.approximate(x1=x1)

    # Plot rectangle
    plt.figure(figsize=figsize)
    r.plot(edgecolor='k', facecolor='none', lw=3)

    # Plot breakdown
    t = np.linspace(0, 2 * np.pi)
    xp = np.cos(t)
    yp = np.sin(t)
    for x, y, radius in approx:
        plt.plot(x + radius * xp, y + radius * yp, 'b')

    # Format Axes
    xtick = np.unique([circ[0] for circ in approx])
    ytick = np.unique([circ[1] for circ in approx])
    plt.xticks(xtick)
    plt.yticks(ytick)

    plt.axis('scaled')
    plt.grid(True, linestyle=':')


if __name__ == '__main__':
    plt.rc('savefig', dpi=300, bbox='tight', pad_inches=0)
    main()