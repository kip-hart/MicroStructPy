from __future__ import division

import os
import sys

import numpy as np

from microstructpy import geometry
from microstructpy.meshing.polymesh import PolyMesh
from microstructpy.meshing.polymesh import kp_loop
from microstructpy.seeding import Seed
from microstructpy.seeding import SeedList

'''
Test Mesh

    D ------- C
    |         | \
    |         |  \
    |         |   \
    A ------- B -- E

    A = (0, 0)
    B = (1, 0)
    C = (1, 1)
    D = (0, 1)
    E = (1.5, 0)

    Facets:
        0 A B
        1 B C
        2 C D
        3 D A
        4 B E
        5 E C

    Regions:
        0, 1, 2, 3
        4, 5, 1

    Seed Numbers:
        0
        1

    Phase Numbers:
        2
        2

'''

A = (0, 0)
B = (1, 0)
C = (1, 1)
D = (0, 1)
E = (1.5, 0)
pts = [A, B, C, D, E]

facets = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (4, 2)]

regions = [(0, 1, 2, 3), (4, 5, 1)]

seed_nums = [0, 1]
phase_nums = [2, 2]

pmesh = PolyMesh(pts, facets, regions, seed_nums, phase_nums)


def test_eq():
    assert pmesh == pmesh

    r2 = [(0, 3, 2, 1), (4, 1, 5)]
    pmesh2 = PolyMesh(pts, facets, r2, seed_nums, phase_nums)
    assert pmesh == pmesh2
    assert pmesh2 == pmesh

    assert pmesh != pts
    assert pmesh != PolyMesh(pts, facets, regions, seed_nums, phase_nums[:-1])

    r3 = [(0, 3, 2), (4, 1, 5)]
    assert pmesh != PolyMesh(pts, facets, r3, seed_nums, phase_nums)

    pt2 = [(-1, 0), B, C, D, E]
    assert pmesh != PolyMesh(pt2, facets, regions, seed_nums, phase_nums)

    f2 = [(0, 1), (1, 3), (2, 3), (3, 0), (1, 4), (4, 2)]
    assert pmesh != PolyMesh(pts, f2, regions, seed_nums, phase_nums)

    pmesh3 = PolyMesh(pts, facets, regions)
    assert pmesh3 == pmesh3
    assert pmesh3 != pmesh


def test_read_write():
    pyver_str = sys.version.split(' ')[0]
    pyver_str = pyver_str.replace('.', '_')

    fname = 'tmp_' + pyver_str + '.txt'
    filepath = os.path.dirname(__file__)
    filename = filepath + '/' + fname

    pmesh.write(filename)
    rw_pmesh = PolyMesh.from_file(filename)
    os.remove(filename)
    assert pmesh == rw_pmesh

    alt_str = 'comments\n' + str(pmesh)
    with open(filename, 'w') as file:
        file.write(alt_str + '\n')
    rw_pmesh = PolyMesh.from_file(filename)
    os.remove(filename)
    assert pmesh == rw_pmesh


def test_repr():
    assert pmesh == eval(repr(pmesh))

    pmesh2 = PolyMesh(pts, facets, regions)
    assert pmesh2 == eval(repr(pmesh2))


def test_from_seeds():
    # d^2 = dx^2 + dy^2 - r^2
    # Let d = 3
    # And circle 1 has radius 4,
    # so dx = 5
    # And circle 2 has radius 2,
    # so dx = np.sqrt(13)
    # and in both cases, y= 0

    d = 3
    r1 = 4
    r2 = 2

    x1 = -np.sqrt(d * d + r1 * r1)
    x2 = np.sqrt(d * d + r2 * r2)

    p1 = 2
    p2 = 3

    x_len = 2 * 1.1 * max(-x1 + r1, x2 + r2)
    y_len = 2 * 1.1 * max(r1, r2)

    # create polymesh by hand
    A = (-0.5 * x_len, -0.5 * y_len)
    B = (0, -0.5 * y_len)
    C = (0.5 * x_len, -0.5 * y_len)
    D = (0.5 * x_len, 0.5 * y_len)
    E = (0, 0.5 * y_len)
    F = (-0.5 * x_len, 0.5 * y_len)

    pts = [A, B, C, D, E, F]
    facets = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4)]
    regions = [(0, 6, 4, 5), (1, 2, 3, 6)]
    poly_exp = PolyMesh(pts, facets, regions, [0, 1], [p1, p2])

    # create polymesh from seeds and domain
    s1 = Seed.factory('circle', r=r1, center=(x1, 0), phase=p1)
    s2 = Seed.factory('circle', r=r2, center=(x2, 0), phase=p2)
    slist = SeedList([s1, s2])

    lens = [x_len, y_len]
    dom = geometry.Rectangle(center=(0.0, 0.0), side_lengths=lens)

    poly_act = PolyMesh.from_seeds(slist, dom)
    assert poly_exp == poly_act


def test_kp_loop():
    p1 = [0, 1]
    p2 = [1, 2]
    p3 = [2, 0]
    pairs = [p1, p2, p3]
    loop = kp_loop(pairs)

    possible_pairs = [(0, 1, 2),
                      (1, 2, 0),
                      (2, 0, 1),
                      (2, 1, 0),
                      (1, 0, 2),
                      (0, 2, 1)]
    assert tuple(loop) in possible_pairs
