import sys
sys.path.append('..')
import numpy as np
import tip_tilt
import unittest

class test_tip_tilt(unittest.TestCase):
    
    def test(self):
        prior_prec = 1.
        k = 2
        l = 10
        
        
        D = np.random.normal(size=(l, k, 20, 20)) + np.random.normal(size=(l, k, 20, 20))*1.j
        S = np.random.normal(size=(l, k, 20, 20)) + np.random.normal(size=(l, k, 20, 20))*1.j
        F = np.random.normal(size=(l, 1, 20, 20)) + np.random.normal(size=(l, 1, 20, 20))*1.j
        xs = np.linspace(-1., 1., D.shape[2])
        coords = np.dstack(np.meshgrid(xs, xs)[::-1])
        tt = tip_tilt.tip_tilt(D, S, F, coords, prior_prec=prior_prec)
        
        
        
        theta = np.random.normal(size=2*(l+1), scale=1./np.sqrt(prior_prec + 1e-10))#np.zeros(2*self.L)
        result = tt.lik(theta, None)

        a = theta[0:2*l].reshape((l, 2))
        a0 = theta[2*l:2*l+2]
        au = np.tensordot(a, coords, axes=(1, 2)) + np.tensordot(a0, coords, axes=(0, 2))

        C_T = np.transpose(np.absolute(S)*np.absolute(D)*np.absolute(F), axes=(1, 0, 2, 3)) # swap k and l
        D_T = np.transpose(np.angle(D)-np.angle(S)-np.angle(F), axes=(1, 0, 2, 3)) # swap k and l

        phi = D_T - au
        expected_result = np.sum(C_T*np.cos(phi))
        expected_result += np.sum(a*a)*prior_prec/2

        np.testing.assert_almost_equal(result, expected_result)
        
        #tt.optimize()
        
if __name__ == '__main__':
    unittest.main()
