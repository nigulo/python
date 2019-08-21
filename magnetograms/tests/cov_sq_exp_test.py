import sys
sys.path.append('..')
import numpy as np
import cov_sq_exp
import unittest

class test_cov_sq_exp(unittest.TestCase):

    def test_calc_cov(self):
        
        x1_range = 1.0
        x2_range = 1.0
        n1 = 2
        n2 = 2
        x1 = np.linspace(0, x1_range, n1)
        x2 = np.linspace(0, x2_range, n2)
        x_mesh = np.meshgrid(x1, x2)
        x = np.dstack(x_mesh).reshape(-1, 2)

        sig_var = 0.5
        ell = 0.2
        noise_var = 0.07        
        gp = cov_sq_exp.cov_sq_exp(sig_var, ell, noise_var, dim_out=2)
        
        K, K_grads = gp.calc_cov(x, x, data_or_test=True, calc_grad = True)
        
        #######################################################################        
        # Compare covariance
        
        K_expected = np.zeros((8, 8))
        for i in np.arange(0, 4):
            i_abs = 2*i
            for j in np.arange(0, 4):
                x_diff = x[i] - x[j]
                x_diff_sq = np.dot(x_diff, x_diff)
                j_abs = 2*j
                K_expected[i_abs:i_abs+2, j_abs:j_abs+2] = np.identity(2) * sig_var * np.exp(-0.5 * x_diff_sq / ell / ell)
        K_expected += np.identity(8)*noise_var
        
        np.testing.assert_almost_equal(K, K_expected, 7)
                
        #######################################################################        
        # Compare gradients of covariance

        K_grads_expected = np.zeros((3, 8, 8))
        for i in np.arange(0, 4):
            i_abs = 2*i
            for j in np.arange(0, 4):
                x_diff = x[i] - x[j]
                x_diff_sq = np.dot(x_diff, x_diff)
                j_abs = 2*j
                exp_fact = np.exp(-0.5 * x_diff_sq / ell / ell)
                K_grads_expected[1, i_abs:i_abs+2, j_abs:j_abs+2] = np.identity(2) * sig_var * exp_fact * x_diff_sq / ell / ell / ell
                K_grads_expected[0, i_abs:i_abs+2, j_abs:j_abs+2] = np.identity(2) * exp_fact
        K_grads_expected[2, :, :] += np.identity(8)

        np.testing.assert_almost_equal(K_grads, K_grads_expected, 7)

        #######################################################################        
        # Compare gradients with values calculate using finite difference
        delta_sig_var = sig_var*1.0e-5
        delta_ell = ell * 1.0e-10
        delta_noise_var = noise_var * 1.0e-5

        gp1 = cov_sq_exp.cov_sq_exp(sig_var + delta_sig_var, ell, noise_var, dim_out=2)
        K1, _ = gp1.calc_cov(x, x, data_or_test=True, calc_grad = True)
        K_grads_expected = (K1 - K) / delta_sig_var
        np.testing.assert_almost_equal(K_grads[0,:,:], K_grads_expected, 10)
        
        gp1 = cov_sq_exp.cov_sq_exp(sig_var - delta_sig_var, ell, noise_var, dim_out=2)
        K1, _ = gp1.calc_cov(x, x, data_or_test=True, calc_grad = True)
        K_grads_expected = -(K1 - K) /delta_sig_var
        np.testing.assert_almost_equal(K_grads[0,:,:], K_grads_expected, 10)

        gp1 = cov_sq_exp.cov_sq_exp(sig_var, ell + delta_ell, noise_var, dim_out=2)
        K1, _ = gp1.calc_cov(x, x, data_or_test=True, calc_grad = True)
        K_grads_expected = (K1 - K) / delta_ell
        np.testing.assert_almost_equal(K_grads[1,:,:], K_grads_expected, 9)
        
        gp1 = cov_sq_exp.cov_sq_exp(sig_var, ell - delta_ell, noise_var, dim_out=2)
        K1, _ = gp1.calc_cov(x, x, data_or_test=True, calc_grad = True)
        K_grads_expected = -(K1 - K) / delta_ell
        np.testing.assert_almost_equal(K_grads[1,:,:], K_grads_expected, 9)

        gp1 = cov_sq_exp.cov_sq_exp(sig_var, ell, noise_var + delta_noise_var, dim_out=2)
        K1, _ = gp1.calc_cov(x, x, data_or_test=True, calc_grad = True)
        K_grads_expected = (K1 - K) / delta_noise_var
        np.testing.assert_almost_equal(K_grads[2,:,:], K_grads_expected, 10)
        
        gp1 = cov_sq_exp.cov_sq_exp(sig_var, ell, noise_var - delta_noise_var, dim_out=2)
        K1, _ = gp1.calc_cov(x, x, data_or_test=True, calc_grad = True)
        K_grads_expected = -(K1 - K) / delta_noise_var
        np.testing.assert_almost_equal(K_grads[2,:,:], K_grads_expected, 10)


        
if __name__ == '__main__':
    unittest.main()
