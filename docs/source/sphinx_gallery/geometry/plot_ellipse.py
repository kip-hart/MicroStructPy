"""
Plot figures relevant to :class:`microstructpy.geometry.Ellipse`

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections
from matplotlib.ticker import FormatStrFormatter

import microstructpy as msp


def main():
    breakdown()  # ellipse_001.png - ellipse breakdown


def breakdown():
    a = 3
    b = 1
    x1 = 0.7

    plt.figure(figsize=(14, 6))
    ellipse = msp.geometry.Ellipse(a=a, b=b)
    approx = ellipse.approximate(x1)
    ellipse.plot(edgecolor='k', facecolor='none', lw=3)
    t = np.linspace(0, 2 * np.pi)
    for x, y, r in approx:
        plt.plot(x + r * np.cos(t), y + r * np.sin(t), 'b')
    
    xticks = np.unique(np.concatenate((approx[:, 0], (-a, a))))
    plt.xticks(xticks)
    plt.yticks(np.unique(np.concatenate((approx[:, 1], (-b, b)))))
    plt.gca().set_xticklabels([str(round(float(label), 1)) for label in xticks])
    plt.axis('scaled')
    plt.grid(True, linestyle=':')


if __name__ == '__main__':
    plt.rc('savefig', dpi=300, bbox='tight', pad_inches=0)
    main()