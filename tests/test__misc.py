import numpy as np
import pytest
import scipy.stats

from microstructpy import _misc


@pytest.mark.parametrize('i', [-90, -4, 0, 3, 16, 291])
def test_from_str_int(i):
    s = str(i)
    assert _misc.from_str(s) == i


@pytest.mark.parametrize('f', [1.234, 0.214, -0.4, 1.2e5])
def test_from_str_float_unformatted(f):
    s = str(f)
    assert _misc.from_str(s) == f


@pytest.mark.parametrize('f', [-4.251e3, 1.234, 0.214, -0.4, 1.2e5])
def test_from_str_float_formatted(f):
    s = '{: ^20.8e}'.format(f)
    assert _misc.from_str(s) == f


@pytest.mark.parametrize('b', [True, False])
@pytest.mark.parametrize('fmt', [None, 'capitalize', 'lower', 'swapcase',
                                 'title', 'upper'])
def test_from_str_bool(b, fmt):
    if fmt is None:
        s = str(b)
    else:
        s = getattr(str(b), fmt)()
    assert _misc.from_str(s) == b


@pytest.mark.parametrize('s', ['abc', 'def', '123a5', '{:2.4f}'])
def test_from_str_str(s):
    assert _misc.from_str(s) == s


@pytest.mark.parametrize('u', [u'A unicode \u018e string \xf1', u'abcdef'])
def test_from_str_unicode(u):
    assert _misc.from_str(u) == u


@pytest.mark.parametrize('l', [[], [0], ['abc', 2.34], [False, '2.3', 45]])
def test_from_str_list(l):
    s = str(l)
    assert _misc.from_str(s) == l


@pytest.mark.parametrize('inp,out', [('12, 49', (12, 49)),
                                     ('abc, def', ('abc', 'def'))])
def test_from_str_list_without(inp, out):
    act_out = _misc.from_str(inp)
    assert len(act_out) == len(out)
    for i, o in zip(_misc.from_str(inp), out):
        assert i == o


@pytest.mark.parametrize('t', [(2, 3), ((5.1, -1.2),), ('abc', 'def', '2')])
def test_from_str_tuple(t):
    s = str(t)
    assert _misc.from_str(s) == t


@pytest.mark.parametrize('v', [1.2, True, -3, [3.2, 4, 'a'], {1, 2}])
def test_from_str_bad_input(v):
    with pytest.raises(TypeError):
        _misc.from_str(v)


@pytest.fixture(scope='module', params=[2, -1.2, [53, 1e-3, 1.2e2]])
def dist_const(request):
    yield request.param


@pytest.fixture(scope='module', params=[('norm', {'loc': -1.2, 'scale': 2.3}),
                                        ('lognorm', {'scale': 0.2, 's': 1.2}),
                                        ('multivariate_normal',
                                         {'mean': [2, 3, 1],
                                          'cov': [[2, -3, 1],
                                                  [-3, 10, 2],
                                                  [1, 2, 200]]})])
def dist_distrib(request):
    dist_type, kwargs = request.param
    f = scipy.stats.__dict__[dist_type]
    yield f(**kwargs)


@pytest.fixture(scope='module', params=[1, 2, 5, 10, 100])
def n_samples(request):
    yield request.param


@pytest.fixture(scope='module', params=[1, 2, 3])
def order(request):
    yield request.param


def test_rvs_const_single(dist_const):
    assert _misc.rvs(dist_const) == dist_const


@pytest.mark.parametrize('kw', [None, 'size'])
def test_rvs_const_multi(kw, dist_const, n_samples):
    if kw is None:
        samples = _misc.rvs(dist_const, n_samples)
    else:
        samples = _misc.rvs(dist_const, **{kw: n_samples})

    try:
        iter(dist_const)
    except TypeError:
        assert np.isclose(samples, dist_const).all()
        if n_samples == 1:
            try:
                iter(samples)
            except TypeError:
                pass
            else:
                assert False
        else:
            assert len(samples) == n_samples
    else:
        assert len(samples) == len(dist_const)
        for group, value in zip(samples, dist_const):
            assert np.isclose(group, value).all()
            if n_samples == 1:
                try:
                    iter(group)
                except TypeError:
                    pass
                else:
                    assert False
            else:
                assert len(group) == n_samples


def test_rvs_distrib_single(dist_distrib):
    sample = _misc.rvs(dist_distrib)
    try:
        n_sample = len(sample)
    except TypeError:
        n_sample = 1

    if hasattr(dist_distrib, 'std'):
        n_distrib = 1
        assert n_sample == n_distrib


@pytest.mark.parametrize('kw', [None, 'size'])
def test_rvs_distrib_multi(kw, dist_distrib, n_samples):
    if kw is None:
        samples = _misc.rvs(dist_distrib, n_samples)
    else:
        samples = _misc.rvs(dist_distrib, **{kw: n_samples})

    try:
        iter(dist_distrib)
    except TypeError:
        if n_samples > 1:
            assert len(samples) == n_samples
    else:
        assert len(samples) == len(dist_distrib)
        for group in samples:
            if n_samples == 1:
                assert hasattr(group, 'std')
            else:
                assert len(group) == n_samples


def test_moment_const(order, dist_const):
    try:
        iter(dist_const)
    except TypeError:
        assert _misc.moment(dist_const, order) == dist_const ** order
    else:
        m = _misc.moment(dist_const, order)
        assert len(m) == len(dist_const)
        for v_distrib, v_moment in zip(dist_const, m):
            assert v_moment == v_distrib ** order


def test_moment_distrib(order, dist_distrib):
    try:
        iter(_misc.rvs(dist_distrib))
    except TypeError:
        is_multi = False
    else:
        is_multi = True

    print('sample', _misc.rvs(dist_distrib))

    if not is_multi:
        m = _misc.moment(dist_distrib, order)
        if order == 1:
            assert np.isclose(m, dist_distrib.mean())
        elif order == 2:
            mu = dist_distrib.mean()
            var = dist_distrib.var()
            assert np.isclose(m, mu * mu + var)


def test_mean_const(dist_const):
    assert np.isclose(_misc.mean(dist_const), dist_const).all()


def test_mean_distrib(dist_distrib):
    try:
        iter(dist_distrib)
    except TypeError:
        if hasattr(dist_distrib, 'mean'):
            out_mean = _misc.mean(dist_distrib)
            try:
                in_mean = dist_distrib.mean()
            except TypeError:
                in_mean = getattr(dist_distrib, 'mean')
            assert np.isclose(out_mean, in_mean).all()
