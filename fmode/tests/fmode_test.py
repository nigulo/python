import sys
sys.path.append('..')
sys.path.append('../utils')
import numpy as np
import unittest
import fmode
import misc

class test_fmode(unittest.TestCase):
    
    def test_grad(self):
        x = np.linspace(0, 10, 10)
        coords = misc.meshgrid(x, x, x)#np.reshape([np.tile(x, len(x)), np.repeat(x, len(x))], (10, 10, 2))
        print(coords.shape)
        params = np.abs(np.random.normal(size=5*3))
        data_fitted = fmode.basis_func(coords, params)
        sigma = 1.0
        data = np.random.normal(size=(coords.shape[0], coords.shape[1]))
        data_mask = np.ones_like(data)
        #sys.exit()

        grads = fmode.calc_loglik_grad(coords, data_fitted, data, data_mask, sigma, params)
        

        #######################################################################
        # Check against values calculated using finite differences
        delta_params = np.ones_like(params)*1.0e-8

        loglik = fmode.calc_loglik(data_fitted, data, data_mask, sigma)

        liks = np.array([loglik])
        liks = np.repeat(liks, len(params), axis=0)
        liks1 = np.zeros_like(liks)
        #liks = np.tile(lik, (alphas.shape[0], alphas.shape[1]))
        #liks1 = np.zeros_like(alphas)
        for l in np.arange(len(params)):
            delta = np.zeros_like(params)
            delta[l] = delta_params[l]
            data_fitted = fmode.basis_func(coords, params+delta)
            liks1[l] = fmode.calc_loglik(data_fitted, data, data_mask, sigma)

        grads_expected = (liks1 - liks) / delta_params

        np.testing.assert_almost_equal(grads, grads_expected, 6)
        
if __name__ == '__main__':
    unittest.main()