import numpy as np
from zernike import zernike, get_nm, get_noll
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
                        z = zernike(n, m)
                        if (n-m) % 2 != 0:
                            expected = 0.0
                        else:
                            expected = radial_polynomial(rho, abs(m), n)*np.sin(phi*abs(m))
                        np.testing.assert_approx_equal(z.get_value(rho, phi), expected, significant=7)
                    for m in np.arange(0, n + 1):
                        z = zernike(n, m)
                        if (n-m) % 2 != 0:
                            expected = 0.0
                        else:
                            expected = radial_polynomial(rho, m, n)*np.cos(phi*m)
                        np.testing.assert_approx_equal(z.get_value(rho, phi), expected, significant=7)


class test_get_noll(unittest.TestCase):

    def test(self):
        np.testing.assert_equal(get_noll(0, 0), 1)
        np.testing.assert_equal(get_noll(1, 1), 2)
        np.testing.assert_equal(get_noll(1, -1), 3)
        np.testing.assert_equal(get_noll(2, 0), 4)
        np.testing.assert_equal(get_noll(2, -2), 5)
        np.testing.assert_equal(get_noll(2, 2), 6)
        np.testing.assert_equal(get_noll(3, -1), 7)
        np.testing.assert_equal(get_noll(3, 1), 8)
        np.testing.assert_equal(get_noll(3, -3), 9)
        np.testing.assert_equal(get_noll(3, 3), 10)
        np.testing.assert_equal(get_noll(4, 0), 11)
        np.testing.assert_equal(get_noll(4, 2), 12)
        np.testing.assert_equal(get_noll(4, -2), 13)
        np.testing.assert_equal(get_noll(4, 4), 14)
        np.testing.assert_equal(get_noll(4, -4), 15)
        np.testing.assert_equal(get_noll(5, 1), 16)
        np.testing.assert_equal(get_noll(5, -1), 17)
        np.testing.assert_equal(get_noll(5, 3), 18)
        np.testing.assert_equal(get_noll(5, -3), 19)
        np.testing.assert_equal(get_noll(5, 5), 20)

class test_get_nm(unittest.TestCase):

    def test(self):
        np.testing.assert_equal(get_nm(1), (0, 0))
        np.testing.assert_equal(get_nm(2), (1, 1))
        np.testing.assert_equal(get_nm(3), (1, -1))
        np.testing.assert_equal(get_nm(4), (2, 0))
        np.testing.assert_equal(get_nm(5), (2, -2))
        np.testing.assert_equal(get_nm(6), (2, 2))
        np.testing.assert_equal(get_nm(7), (3, -1))
        np.testing.assert_equal(get_nm(8), (3, 1))
        np.testing.assert_equal(get_nm(9), (3, -3))
        np.testing.assert_equal(get_nm(10), (3, 3))
        np.testing.assert_equal(get_nm(11), (4, 0))
        np.testing.assert_equal(get_nm(12), (4, 2))
        np.testing.assert_equal(get_nm(13), (4, -2))
        np.testing.assert_equal(get_nm(14), (4, 4))
        np.testing.assert_equal(get_nm(15), (4, -4))
        np.testing.assert_equal(get_nm(16), (5, 1))
        np.testing.assert_equal(get_nm(17), (5, -1))
        np.testing.assert_equal(get_nm(18), (5, 3))
        np.testing.assert_equal(get_nm(19), (5, -3))
        np.testing.assert_equal(get_nm(20), (5, 5))
        
        
if __name__ == '__main__':
    unittest.main()