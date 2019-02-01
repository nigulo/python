import numpy as np
import GPR_div_free
import unittest

class test_GPR_div_feee(unittest.TestCase):

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
        gp = GPR_div_free.GPR_div_free(sig_var, ell, noise_var)
        
        K, K_grads = gp.calc_cov(x, x, data_or_test=True, calc_grad = True)
        
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
                        exp_fact = np.exp(-0.5 * x_diff_sq / ell / ell)
                        K_expected[i_abs, j_abs] *= sig_var * exp_fact
        K_expected += np.identity(8)*noise_var
        
        np.testing.assert_almost_equal(K, K_expected, 10)
                
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
                            K_grads_expected[1, i_abs, j_abs] += 1 + 2.0 * x_diff_sq / ell / ell / ell
                        exp_fact = np.exp(-0.5 * x_diff_sq / ell / ell)
                        K_grads_expected[0, i_abs, j_abs] *= K_i_j * exp_fact
                        K_grads_expected[1, i_abs, j_abs] *= sig_var
                        K_grads_expected[1, i_abs, j_abs] += K_i_j * x_diff_sq / ell / ell / ell
                        K_grads_expected[1, i_abs, j_abs] *= exp_fact
        K_grads_expected[2, :, :] += np.identity(8)
                
        np.testing.assert_almost_equal(K_grads, K_grads_expected, 10)
        

        
if __name__ == '__main__':
    unittest.main()