import numpy as np
from cov_exp_quad import cov_exp_quad
import unittest

class test_cov_exp_quad(unittest.TestCase):

    def test(self):
        k = cov_exp_quad(N=4)
        F, q = k.get_F_q(sigma = 1.0, ell = 1.0)
        #F_expected = np.array([
        #    [ 0., 1.,         0.,          0.,          0.        ],
        #    [ 0., 0.,         1.,          0.,          0.        ],
        #    [ 0., 0.,         0.,          1.,          0.        ],
        #    [ 0., 0.,         0.,          0.,          1.        ],
        #    [-1., -1.6487038, -1.10911211, -0.36610328, -0.05103104]])
            
            
        F_expected = np.array([
            [ 0.,           1.,           0.,           0.        ],
            [ 0.,           0.,           1.,           0.        ],
            [ 0.,           0.,           0.,           1.        ],
            [-19.59591794, -32.30786441, -21.73406995,  -7.17412991]])

            
        np.testing.assert_allclose(F, F_expected, rtol=1e-7, atol=0)
        q_expected = 962.545257458
        np.testing.assert_almost_equal(q, q_expected, decimal=9)
        
        #######################################################################
        
        k = cov_exp_quad(N=20)
        sigma = 3.74
        for ell in np.linspace(0.1, 15, 11):
            kappa = 1.0/(2.0*ell*ell)
            for omega in np.linspace(-2.0*np.pi/ell, 2.0*np.pi/ell, 11):
    
                F, q, P, P_plus, P_minus = k.get_F_q(sigma, ell, return_P=True)
                
                m = len(P_plus)
                np.testing.assert_equal(m, len(P_minus))
                for i in np.arange(0, m):
                    np.testing.assert_approx_equal(P_plus[i], P_minus[i]*(-1.0)**i, significant=6)
                    
                
                S = q/np.polynomial.polynomial.polyval(1.j*omega, np.flip(P, axis=0))
                #S = q/np.polynomial.polynomial.polyval(1.j*omega, np.flip(P_plus, axis=0))/np.polynomial.polynomial.polyval(1.j*omega, np.flip(P_minus, axis=0))
                np.testing.assert_almost_equal(S.imag, 0, decimal=12)
                S_expected = sigma * np.sqrt(np.pi/kappa)*np.exp(-omega**2/4.0/kappa)
                np.testing.assert_almost_equal(S.real, S_expected, decimal=5)

if __name__ == '__main__':
    unittest.main()