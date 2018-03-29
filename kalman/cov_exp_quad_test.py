import numpy as np
from cov_exp_quad import cov_exp_quad
import unittest

class test_cov_exp_quad(unittest.TestCase):

    def test(self):
        k = cov_exp_quad(N=4)
        F, q = k.get_F_q(sigma = 1.0, ell = 1.0)
        F_expected = np.array([
            [ 0., 1.,         0.,          0.,          0.        ],
            [ 0., 0.,         1.,          0.,          0.        ],
            [ 0., 0.,         0.,          1.,          0.        ],
            [ 0., 0.,         0.,          0.,          1.        ],
            [-1., -1.6487038, -1.10911211, -0.36610328, -0.05103104]])
            
        np.testing.assert_allclose(F, F_expected, rtol=1e-7, atol=0)
        q_expected = 962.545257458
        np.testing.assert_almost_equal(q, q_expected, decimal=9)
        
        #######################################################################
        
        k = cov_exp_quad(N=20)
        ell = 2.123
        sigma = 3.74
        kappa = 1.0/(2.0*ell*ell)
        for omega in np.linspace(0, 100, 100):

            F, q, P, P_plus, P_minus = k.get_F_q(sigma, ell, return_P=True)
            
            m = len(P_plus)
            np.testing.assert_equal(m, len(P_minus))
            for i in np.arange(0, m):
                np.testing.assert_almost_equal(P_plus[i], P_minus[i]*(-1.0)**i, decimal=9)
                
            
            S = q/np.polynomial.polynomial.polyval(1.j*omega, P)
            np.testing.assert_equal(S.imag, 0)
            S_expected = sigma * np.sqrt(np.pi/kappa)*np.exp(-omega**2/4.0/kappa)
            np.testing.assert_almost_equal(S.real, S_expected, decimal=6)
            

if __name__ == '__main__':
    unittest.main()