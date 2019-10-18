import pytest

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
def test_from_str_bool(b):
    s = str(b)
    assert _misc.from_str(s) == b
    for kw in ['capitalize', 'lower', 'swapcase', 'title', 'upper']:
        new_s = getattr(s, kw)()
        assert _misc.from_str(new_s) == b


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


@pytest.mark.parametrize('v', [1.2, True, -3, [3.2, 4, 'a'], {1, 2}])
def test_from_str_bad_input(v):
    with pytest.raises(TypeError):
        _misc.from_str(v)
