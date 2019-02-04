import numpy as np
import scipy.linalg as la
import kiss_gp
import GPR_div_free
import unittest

def cov_func(sig_var, ell, noise_var, u):
    gp = GPR_div_free.GPR_div_free(sig_var, ell, noise_var)
    U, U_grads = gp.calc_cov(u, u, data_or_test=True, calc_grad = True)
    return  U, U_grads

class test_GPR_div_feee(unittest.TestCase):

    def test_kiss_gp(self):
        
        n1 = 10
        n2 = 10
        n = n1*n2
        x1_range = 1.0
        x2_range = 1.0
        x1 = np.linspace(0, x1_range, n1)
        x2 = np.linspace(0, x2_range, n2)
        x_mesh = np.meshgrid(x1, x2)
        x = np.dstack(x_mesh).reshape(-1, 2)
        
        m1 = 5
        m2 = 5
        u1 = np.linspace(0, x1_range, m1)
        u2 = np.linspace(0, x2_range, m2)
        u_mesh = np.meshgrid(u1, u2)
        u = np.dstack(u_mesh).reshape(-1, 2)

        sig_var_train = 0.2
        length_scale_train = 0.2
        noise_var_train = 0.000001
        mean_train = 0.0
        
        gp_train = GPR_div_free.GPR_div_free(sig_var_train, length_scale_train, noise_var_train)
        K = gp_train.calc_cov(x, x, True)
        
        L = la.cholesky(K)
        s = np.random.normal(0.0, 1.0, 2*n)
        
        y = np.repeat(mean_train, 2*n) + np.dot(L, s)
        
        #y = np.reshape(y, (n, 2))
        y = np.reshape(y, (2*n, -1))



        kgp = kiss_gp.kiss_gp(x, u_mesh, u, y, cov_func)

        sig_var = 0.5
        ell = 0.2
        noise_var = 0.07        

        loglik = kgp.likelihood([ell, sig_var], [noise_var, y])
        loglik_grads = kgp.likelihood_grad([ell, sig_var], [noise_var, y])
        
        delta_sig_var = sig_var*1.0e-5
        delta_ell = ell * 1.0e-10

        loglik1 = kgp.likelihood([ell + delta_ell, sig_var], [noise_var, y])
        loglik_grad_expected = (loglik1 - loglik) / delta_ell
        np.testing.assert_almost_equal(loglik_grads[0], loglik_grad_expected, 10)
        
        loglik1 = kgp.likelihood([ell - delta_ell, sig_var], [noise_var, y])
        loglik_grad_expected = -(loglik1 - loglik) / delta_ell
        np.testing.assert_almost_equal(loglik_grads[0], loglik_grad_expected, 10)

        loglik1 = kgp.likelihood([ell, sig_var + delta_sig_var], [noise_var, y])
        loglik_grad_expected = (loglik1 - loglik) / delta_sig_var
        np.testing.assert_almost_equal(loglik_grads[1], loglik_grad_expected, 10)

        loglik1 = kgp.likelihood([ell, sig_var - delta_sig_var], [noise_var, y])
        loglik_grad_expected = -(loglik1 - loglik) / delta_sig_var
        np.testing.assert_almost_equal(loglik_grads[1], loglik_grad_expected, 10)
        
if __name__ == '__main__':
    unittest.main()