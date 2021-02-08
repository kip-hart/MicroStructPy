import numpy as np

from microstructpy import _misc


def test_from_str_int():
    pairs = [('0', 0),
             ('1', 1),
             ('2', 2),
             ('-1', -1),
             ('-2', -2)]

    for int_str, int_exp in pairs:
        int_act = _misc.from_str(int_str)
        a_str = 'Expected ' + str(int_exp) + ' and got ' + str(int_act)
        assert int_exp == int_act, a_str


def test_from_str_float():
    pairs = [('1.234', 1.234),
             ('0.214', 0.214),
             ('-0.4', -0.4),
             ('1.2e5', 1.2e5)]

    for flt_str, flt_exp in pairs:
        flt_act = _misc.from_str(flt_str)
        a_str = 'Expected ' + str(flt_exp) + ' and got ' + str(flt_act)
        assert flt_exp == flt_act, a_str


def test_from_str_bool():
    pairs = [('True', True),
             ('False', False),
             ('true', True),
             ('false', False)]

    for bool_str, bool_exp in pairs:
        bool_act = _misc.from_str(bool_str)
        a_str = 'Expected ' + str(bool_exp) + ' and got ' + str(bool_act)
        a_str += ' for string ' + repr(bool_str)
        assert bool_exp == bool_act, a_str


def test_from_str_list():
    pairs = [('[0]', [0]),
             # ('[1, 0, a]', [1, 0, 'a']),
             ('[-2.3, true]', [-2.3, True])]

    for list_str, list_exp in pairs:
        list_act = _misc.from_str(list_str)
        assert len(list_exp) == len(list_act)
        for act_val, exp_val in zip(list_act, list_exp):
            assert act_val == exp_val


'''
def test_from_str_list_of_lists():
    lol_str = '[[1, 0, 0, True, False], [2, 4, a, -2.3]]'
    lol_exp = [[1, 0, 0, True, False], [2, 4, 'a', -2.3]]

    lol_act = _misc.from_str(lol_str)

    assert len(lol_exp) == len(lol_act)
    for list_exp, list_act in zip(lol_exp, lol_act):
        assert len(list_exp) == len(list_act)
        for val_exp, val_act in zip(list_exp, list_act):
            assert val_exp == val_act
'''


def test_tangent_sphere_2D():
    pts = np.array([(0, 0), (4, 0), (3, 3)])
    rads = np.array([1, 0.5, 0.5])

    simps = [None, np.array([[0, 1, 2]])]
    for simp in simps:
        x, y, rad = _misc.tangent_sphere(pts, rads, simplices=simp)
        cen = np.array([x, y])

        rel_pos = pts - cen
        dist = np.sqrt(np.sum(rel_pos * rel_pos, axis=-1))
        assert np.all(np.isclose(dist, rads + rad))


def test_tangent_sphere_ND():
    n = 7
    np.random.seed(0)
    pts = np.random.rand(n + 1, n)
    nsphere = _misc.tangent_sphere(pts)
    cen = nsphere[:-1]
    rad = nsphere[-1]

    rel_pos = pts - cen
    dist = np.sqrt(np.sum(rel_pos * rel_pos, axis=-1))
    assert np.all(np.isclose(dist, rad))
