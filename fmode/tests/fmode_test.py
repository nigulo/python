import sys
sys.path.append('..')
sys.path.append('../utils')
import numpy as np
import unittest
import fmode
import misc

class test_fmode(unittest.TestCase):
 
    def test_param_func_grad(self):
        x = np.linspace(0, 10, 10)
        coords = misc.meshgrid(x, x, x)#np.reshape([np.tile(x, len(x)), np.repeat(x, len(x))], (10, 10, 2))
        params = []
        mode_params = dict()
        
        for mode_index in range(3):
            if mode_index not in mode_params:
                mode_params[mode_index] = dict()
            for nu_index in range(2, 5):
                if nu_index not in mode_params[mode_index]:
                    mode_params[mode_index][nu_index] = []
                k1 = np.abs(np.random.normal())
                k2 = np.abs(np.random.normal())
                k = np.sqrt(k1**2+k2**2)
                beta = np.abs(np.random.normal())
                scale = np.abs(np.random.normal())
                mode_params[mode_index][nu_index].append([len(params), x[1], k1, k2, np.arctan2(k2, k1)])
                params.append(k)
                params.append(beta)
                params.append(scale)

        grads = fmode.basis_func_grad(coords, params, mode_params)
        

        #######################################################################
        # Check against values calculated using finite differences
        delta_params = np.ones_like(params)*1.0e-8

        f, _ = fmode.basis_func(coords, params, mode_params)
        print(f.shape)

        fs = np.tile(f[:, :, :, None], (1, 1, 1, len(params)))
        fs1 = np.empty_like(fs)
        #liks = np.tile(lik, (alphas.shape[0], alphas.shape[1]))
        #liks1 = np.zeros_like(alphas)
        for l in np.arange(len(params)):
            delta = np.zeros_like(params)
            delta[l] = delta_params[l]
            fs1[:, :, :, l], _ = fmode.basis_func(coords, params+delta, mode_params)

        grads_expected = (fs1 - fs) / np.tile(delta_params[None, None, None, :], (coords.shape[0], coords.shape[1], coords.shape[2], 1))

        np.testing.assert_almost_equal(grads, grads_expected, 6)
    

    def test_loglik_grad(self):
        x = np.linspace(0, 10, 10)
        coords = misc.meshgrid(x, x, x)#np.reshape([np.tile(x, len(x)), np.repeat(x, len(x))], (10, 10, 2))
        print(coords.shape)

        params = []
        mode_params = dict()
        
        for mode_index in range(3):
            if mode_index not in mode_params:
                mode_params[mode_index] = dict()
            for nu_index in range(2, 5):
                if nu_index not in mode_params[mode_index]:
                    mode_params[mode_index][nu_index] = []
                k1 = np.abs(np.random.normal())
                k2 = np.abs(np.random.normal())
                k = np.sqrt(k1**2+k2**2)
                beta = np.abs(np.random.normal())
                scale = np.abs(np.random.normal())
                mode_params[mode_index][nu_index].append([len(params), x[1], k1, k2, np.arctan2(k2, k1)])
                params.append(k)
                params.append(beta)
                params.append(scale)

        data_fitted, data_mask = fmode.basis_func(coords, params, mode_params)
        sigma = 1.0
        data = np.random.normal(size=(coords.shape[0], coords.shape[1]))

        grads = fmode.calc_loglik_grad(coords, data_fitted, data, data_mask, sigma, params, mode_params)
        

        #######################################################################
        # Check against values calculated using finite differences
        delta_params = np.ones_like(params)*1.0e-7

        loglik = fmode.calc_loglik(data_fitted, data, data_mask, sigma)

        liks = np.array([loglik])
        liks = np.repeat(liks, len(params), axis=0)
        liks1 = np.zeros_like(liks)
        #liks = np.tile(lik, (alphas.shape[0], alphas.shape[1]))
        #liks1 = np.zeros_like(alphas)
        for l in np.arange(len(params)):
            delta = np.zeros_like(params)
            delta[l] = delta_params[l]
            data_fitted, data_mask = fmode.basis_func(coords, params+delta, mode_params)
            liks1[l] = fmode.calc_loglik(data_fitted, data, data_mask, sigma)

        grads_expected = (liks1 - liks) / delta_params

        np.testing.assert_almost_equal(grads, grads_expected, 6)

    
if __name__ == '__main__':
    unittest.main()