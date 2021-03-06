import sys
sys.path.append('..')
import numpy as np
import cov_div_free
import unittest
import numpy.linalg as la

class test_cov_div_free(unittest.TestCase):

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
        gp = cov_div_free.cov_div_free(sig_var, ell, noise_var)
        
        K, K_grads = gp.calc_cov(x, x, data_or_test=True, calc_grad = True)
        
        #######################################################################        
        # Compare covariance
        
        K_expected = np.zeros((8, 8))
        for i in np.arange(0, 4):
            for j in np.arange(0, 4):
                x_diff = x[i] - x[j]
                x_diff_sq = np.dot(x_diff, x_diff)
                for i1 in np.arange(0, 2):
                    i_abs = 2*i + i1
                    for j1 in np.arange(0, 2):
                        j_abs = 2*j + j1
                        K_expected[i_abs, j_abs] = x_diff[i1] * x_diff[j1] / ell / ell
                        if (i1 == j1):
                            K_expected[i_abs, j_abs] += 1 - x_diff_sq / ell / ell
                        K_expected[i_abs, j_abs] *= sig_var * np.exp(-0.5 * x_diff_sq / ell / ell)
        K_expected += np.identity(8)*noise_var
        
        np.testing.assert_almost_equal(K, K_expected, 10)
                
        #######################################################################        
        # Compare gradients of covariance

        K_grads_expected = np.zeros((3, 8, 8))
        for i in np.arange(0, 4):
            for j in np.arange(0, 4):
                x_diff = x[i] - x[j]
                x_diff_sq = np.dot(x_diff, x_diff)
                for i1 in np.arange(0, 2):
                    i_abs = 2*i + i1
                    for j1 in np.arange(0, 2):
                        j_abs = 2*j + j1
                        K_i_j = x_diff[i1] * x_diff[j1] / ell / ell
                        K_grads_expected[1, i_abs, j_abs] = -2.0 * x_diff[i1] * x_diff[j1] / ell / ell / ell
                        if (i1 == j1):
                            K_i_j += 1 - x_diff_sq / ell / ell
                            K_grads_expected[1, i_abs, j_abs] += 2.0 * x_diff_sq / ell / ell / ell
                        exp_fact = np.exp(-0.5 * x_diff_sq / ell / ell)
                        K_grads_expected[0, i_abs, j_abs] = K_i_j * exp_fact
                        K_grads_expected[1, i_abs, j_abs] += K_i_j * x_diff_sq / ell / ell / ell
                        K_grads_expected[1, i_abs, j_abs] *= sig_var * exp_fact
        K_grads_expected[2, :, :] += np.identity(8)
                
        np.testing.assert_almost_equal(K_grads, K_grads_expected, 10)

        #######################################################################        
        # Compare gradients with values calculate using finite difference
        delta_sig_var = sig_var*1.0e-5
        delta_ell = ell * 1.0e-10
        delta_noise_var = noise_var * 1.0e-5

        gp1 = cov_div_free.cov_div_free(sig_var + delta_sig_var, ell, noise_var)
        K1, _ = gp1.calc_cov(x, x, data_or_test=True, calc_grad = True)
        K_grads_expected = (K1 - K) / delta_sig_var
        np.testing.assert_almost_equal(K_grads[0,:,:], K_grads_expected, 10)
        
        gp1 = cov_div_free.cov_div_free(sig_var - delta_sig_var, ell, noise_var)
        K1, _ = gp1.calc_cov(x, x, data_or_test=True, calc_grad = True)
        K_grads_expected = -(K1 - K) /delta_sig_var
        np.testing.assert_almost_equal(K_grads[0,:,:], K_grads_expected, 10)

        gp1 = cov_div_free.cov_div_free(sig_var, ell + delta_ell, noise_var)
        K1, _ = gp1.calc_cov(x, x, data_or_test=True, calc_grad = True)
        K_grads_expected = (K1 - K) / delta_ell
        np.testing.assert_almost_equal(K_grads[1,:,:], K_grads_expected, 9)
        
        gp1 = cov_div_free.cov_div_free(sig_var, ell - delta_ell, noise_var)
        K1, _ = gp1.calc_cov(x, x, data_or_test=True, calc_grad = True)
        K_grads_expected = -(K1 - K) / delta_ell
        np.testing.assert_almost_equal(K_grads[1,:,:], K_grads_expected, 9)

        gp1 = cov_div_free.cov_div_free(sig_var, ell, noise_var + delta_noise_var)
        K1, _ = gp1.calc_cov(x, x, data_or_test=True, calc_grad = True)
        K_grads_expected = (K1 - K) / delta_noise_var
        np.testing.assert_almost_equal(K_grads[2,:,:], K_grads_expected, 10)
        
        gp1 = cov_div_free.cov_div_free(sig_var, ell, noise_var - delta_noise_var)
        K1, _ = gp1.calc_cov(x, x, data_or_test=True, calc_grad = True)
        K_grads_expected = -(K1 - K) / delta_noise_var
        np.testing.assert_almost_equal(K_grads[2,:,:], K_grads_expected, 10)


    def test_calc_cov_ij(self):
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
        gp = cov_div_free.cov_div_free(sig_var, ell, noise_var)
        
        K, K_grads = gp.calc_cov(x, x, data_or_test=True, calc_grad = True)

        for i in np.arange(0, K.shape[0]):
            for j in np.arange(0, K.shape[1]):
                K_ij = gp.calc_cov_ij(x, x, i, j)
                np.testing.assert_almost_equal(K_ij, K[i, j])

    def test_loglik(self):
        x1_range = 1.0
        x2_range = 1.0
        n1 = 2
        n2 = 2
        x1 = np.linspace(0, x1_range, n1)
        x2 = np.linspace(0, x2_range, n2)
        x_mesh = np.meshgrid(x1, x2)
        x = np.dstack(x_mesh).reshape(-1, 2)
        
        y = np.random.normal(size=(n1, n2, 2))
        y = np.column_stack((np.reshape(y[:, :, 0], n1*n2), np.reshape(y[:, :, 1], n1*n2)))

        sig_var = 0.5
        ell = 0.2
        noise_var = 0.07        
        gp = cov_div_free.cov_div_free(sig_var, ell, noise_var)
        loglik = gp.loglik(x, y)
        
        
        n = np.shape(x)[0]
        if (len(np.shape(x)) > 1):
            k = np.shape(x)[1]
        else:
            k = 1
        y_flat = np.reshape(y, (k*n, -1))
        K = gp.calc_cov(x, x, True)
        L = la.cholesky(K)
        alpha = la.solve(L.T, la.solve(L, y_flat))
        loglik_expected = (-0.5 * np.dot(y_flat.T, alpha) - sum(np.log(np.diag(L))) - 0.5 * n * np.log(2.0 * np.pi)).item()
        np.testing.assert_almost_equal(loglik, loglik_expected)
        

                
    def test_loglik_approx(self):
        x1_range = 1.0
        x2_range = 1.0
        n1 = 2
        n2 = 2
        x1 = np.linspace(0, x1_range, n1)
        x2 = np.linspace(0, x2_range, n2)
        x_mesh = np.meshgrid(x1, x2)
        x = np.dstack(x_mesh).reshape(-1, 2)
        
        y = np.random.normal(size=(n1, n2, 2))
        y = np.column_stack((np.reshape(y[:, :, 0], n1*n2), np.reshape(y[:, :, 1], n1*n2))).flatten()

        sig_var = 0.5
        ell = 0.2
        noise_var = 0.07        
        gp = cov_div_free.cov_div_free(sig_var, ell, noise_var)
        
        loglik1 = gp.loglik_approx(x, y, use_vector_form = False)
        loglik2 = gp.loglik_approx(x, y, use_vector_form = True)
        np.testing.assert_almost_equal(loglik1, loglik2)
        #loglik = gp.init(x, y)
        #np.testing.assert_almost_equal(loglik1, loglik)
        
        
if __name__ == '__main__':
    unittest.main()
