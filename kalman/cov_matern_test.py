import numpy as np
from cov_matern import cov_matern
import unittest
from scipy.special import gamma

class test_cov_matern(unittest.TestCase):

    def test(self):
        p = 2.0
        nu = p+0.5
        k = cov_matern(p=p)
        ell = 1.7
        sigma = 1.5
        F, q = k.get_F_q(sigma = 1.5, ell = ell)
        lmbda = np.sqrt(2.0*nu)/ell
        print lmbda
        F_expected = np.array([
            [0.,          1.,          0.    ],
            [0.,          0.,          1.    ],
            [-lmbda**3, -3*lmbda**2, -3*lmbda]])
            
        np.testing.assert_allclose(F, F_expected, rtol=1e-7, atol=0)
        q_expected = 2.0*sigma*np.sqrt(np.pi)*lmbda**(2.0*nu)*gamma(nu+0.5)/gamma(nu)
        np.testing.assert_almost_equal(q, q_expected, decimal=9)
        
        #######################################################################
        
        nu = 9.5
        k = cov_matern(nu=nu)
        ell = 2.123
        sigma = 3.74
        lmbda = np.sqrt(2.0*nu)/ell
        for omega in np.linspace(0, 100, 100):

            F, q, P_plus_coefs, P_minus_coefs = k.get_F_q(sigma, ell, return_P=True)
            P_plus_coefs = np.flip(P_plus_coefs, axis=0)
            P_minus_coefs = np.flip(P_minus_coefs, axis=0)
            S = q/np.polynomial.polynomial.polyval(1.j*omega, P_plus_coefs)/np.polynomial.polynomial.polyval(1.j*omega, P_minus_coefs)
            np.testing.assert_almost_equal(S.imag, 0, decimal=12)
            S_expected = 2.0*sigma*np.sqrt(np.pi)*lmbda**(2.0*nu)*gamma(nu+0.5)/gamma(nu)*(lmbda**2+omega**2)**(-nu-0.5)
            np.testing.assert_almost_equal(S.real, S_expected, decimal=12)
            

if __name__ == '__main__':
    unittest.main()