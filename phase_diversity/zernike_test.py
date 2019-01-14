import numpy as np
from zernike import zernike, get_mn
import unittest
import scipy.misc as misc

def radial_polynomial(rho, m, n):
    R = 0.0
    for k in np.arange(0, (n-m)/2 + 1):
        R += np.power(rho, n - 2*k)*np.power(-1.0, k)*misc.factorial(n-k)/misc.factorial(k)/misc.factorial((n+m)/2 - k)/misc.factorial((n-m)/2 - k)
    return R
    
class test_zernike(unittest.TestCase):

    def test(self):
        for rho in np.linspace(0.0, 1.0, 10):
            for phi in np.linspace(0.0, 2.0*np.pi, 10):
                for n in np.arange(0, 27):
                    for m in np.arange(-n, 0):
                        z = zernike(m, n)
                        if (n-m) % 2 != 0:
                            expected = 0.0
                        else:
                            expected = radial_polynomial(rho, abs(m), n)*np.sin(phi*abs(m))
                        np.testing.assert_approx_equal(z.get_value(rho, phi), expected, significant=7)
                    for m in np.arange(0, n + 1):
                        z = zernike(m, n)
                        if (n-m) % 2 != 0:
                            expected = 0.0
                        else:
                            expected = radial_polynomial(rho, m, n)*np.cos(phi*m)
                        np.testing.assert_approx_equal(z.get_value(rho, phi), expected, significant=7)

class test_get_mn(unittest.TestCase):

    def test(self):
        np.testing.assert_equal(get_mn(1), (0, 0))
        np.testing.assert_equal(get_mn(2), (-1, 1))
        np.testing.assert_equal(get_mn(3), (1, 1))
        np.testing.assert_equal(get_mn(4), (-2, 2))
        np.testing.assert_equal(get_mn(5), (0, 2))
        np.testing.assert_equal(get_mn(6), (2, 2))
        np.testing.assert_equal(get_mn(7), (-3, 3))
        np.testing.assert_equal(get_mn(8), (-1, 3))
        np.testing.assert_equal(get_mn(9), (1, 3))
        np.testing.assert_equal(get_mn(10), (3, 3))
        np.testing.assert_equal(get_mn(11), (-4, 4))
        np.testing.assert_equal(get_mn(12), (-2, 4))
        np.testing.assert_equal(get_mn(13), (0, 4))
        np.testing.assert_equal(get_mn(14), (2, 4))
        np.testing.assert_equal(get_mn(15), (4, 4))
        np.testing.assert_equal(get_mn(16), (-5, 5))
        np.testing.assert_equal(get_mn(17), (-3, 5))
        np.testing.assert_equal(get_mn(18), (-1, 5))
        np.testing.assert_equal(get_mn(19), (1, 5))
        np.testing.assert_equal(get_mn(20), (3, 5))
        np.testing.assert_equal(get_mn(21), (5, 5))
        
        
if __name__ == '__main__':
    unittest.main()