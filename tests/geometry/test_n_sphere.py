import microstructpy as msp

c1 = msp.geometry.n_sphere.NSphere(r=1, center=(0, 4, -1))
c2 = msp.geometry.n_sphere.NSphere(r=2, center=(3, 1))
c3 = msp.geometry.n_sphere.NSphere(d=4, center=(3, 1))


def test_eq():
    assert c1 == c1
    assert not c2 == c1

    assert c2 == c3
