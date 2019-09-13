import sys
sys.path.append('..')
import numpy as np
import scipy.linalg as la
import kiss_gp
import cov_div_free
import unittest

def cov_func(theta, data, u1, u2, data_or_test):
    sig_var = theta[0]
    ell = theta[1]
    noise_var = data[0]
    gp = cov_div_free.cov_div_free(sig_var, ell, noise_var)
    U, U_grads = gp.calc_cov(u1, u2, data_or_test=data_or_test, calc_grad = True)
    U_grads = U_grads[:-1] # Omit noise_var
    return  U, U_grads

class test_kiss_gp(unittest.TestCase):

    def test_likelihood_and_grad(self):
        
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
        
        gp_train = cov_div_free.cov_div_free(sig_var_train, length_scale_train, noise_var_train)
        K = gp_train.calc_cov(x, x, True)
        
        L = la.cholesky(K)
        s = np.random.normal(0.0, 1.0, 2*n)
        
        y = np.repeat(mean_train, 2*n) + np.dot(L, s)
        
        #y = np.reshape(y, (n, 2))
        y = np.reshape(y, (2*n, -1))



        kgp = kiss_gp.kiss_gp(x, u_mesh, cov_func, y)

        sig_var = 0.5
        ell = 0.2
        noise_var = 0.07        

        loglik = kgp.likelihood([sig_var, ell], [noise_var])
        loglik_grads = kgp.likelihood_grad([sig_var, ell], [noise_var])
        
        delta_sig_var = 1.0e-7
        delta_ell = 1.0e-7

        loglik1 = kgp.likelihood([sig_var + delta_sig_var, ell], [noise_var])
        loglik_grad_expected = (loglik1 - loglik) / delta_sig_var
        np.testing.assert_almost_equal(loglik_grads[0], loglik_grad_expected, 5)

        loglik1 = kgp.likelihood([sig_var - delta_sig_var, ell], [noise_var])
        loglik_grad_expected = -(loglik1 - loglik) / delta_sig_var
        np.testing.assert_almost_equal(loglik_grads[0], loglik_grad_expected, 5)

        loglik1 = kgp.likelihood([sig_var, ell + delta_ell], [noise_var])
        loglik_grad_expected = (loglik1 - loglik) / delta_ell
        np.testing.assert_approx_equal(loglik_grads[1], loglik_grad_expected, 5)
        
        loglik1 = kgp.likelihood([sig_var, ell - delta_ell], [noise_var])
        loglik_grad_expected = -(loglik1 - loglik) / delta_ell
        np.testing.assert_approx_equal(loglik_grads[1], loglik_grad_expected, 5)

    def test_asympthotics(self):
        
        n1 = 10
        n2 = 10
        n3 = 3
        n = n1*n2*n3
        x1_range = 1.0
        x2_range = 1.0
        x3_range = 0.33333333
        x1 = np.linspace(0, x1_range, n1)
        x2 = np.linspace(0, x2_range, n2)
        x3 = np.linspace(0, x3_range, n3)
        
        x1_mesh, x2_mesh, x3_mesh = np.meshgrid(x1, x2, x3, indexing='ij')
        x = np.stack((x1_mesh, x2_mesh, x3_mesh), axis=3)
        x = x.reshape(-1, 3)
        
        #x_mesh = np.meshgrid(x1, x2)
        #x = np.dstack(x_mesh).reshape(-1, 2)
        
        x1_test = np.linspace(0, x1_range, 7)
        x2_test = np.linspace(0, x2_range, 7)
        x3_test = np.linspace(0, x3_range, 2)
        x1_test_mesh, x2_test_mesh, x3_test_mesh = np.meshgrid(x1_test, x2_test, x3_test, indexing='ij')
        x_test = np.stack((x1_test_mesh, x2_test_mesh, x3_test_mesh), axis=3)
        x_test = x_test.reshape(-1, 3)


        #x_test_mesh = np.meshgrid(x1_test, x2_test)
        #x_test = np.dstack(x_test_mesh).reshape(-1, 2)
        
        m1 = 10
        m2 = 10
        m3 = 3
        u1 = np.linspace(0, x1_range, m1)
        u2 = np.linspace(0, x2_range, m2)
        u3 = np.linspace(0, x3_range, m3)
        #u_mesh = np.meshgrid(u1, u2)
        u_mesh = np.meshgrid(u1, u2, u3, indexing='ij')
        #u = np.dstack(u_mesh).reshape(-1, 2)

        sig_var_train = 1.
        length_scale_train = 0.5
        noise_var_train = 0.000001
        mean_train = 0.0
        
        #######################################################################
        # Generate some data
        gp_train = cov_div_free.cov_div_free(sig_var_train, length_scale_train, noise_var_train)
        K = gp_train.calc_cov(x, x, True)
        
        L = la.cholesky(K)
        s = np.random.normal(0.0, 1.0, 3*n)
        
        y = np.repeat(mean_train, 3*n) + np.dot(L, s)
        #y = np.reshape(y, (n1, n2, 2))
        #######################################################################
        #y_downsampled = np.reshape(np.reshape(y, (n1, n2, 2))[::2, ::2, :], (2*m1*m2))
        #print(y.shape, y_downsampled.shape)
        
        #y = np.reshape(y, (n, 2))
        #y = np.reshape(y, (2*n))

        #y_downsampled = np.reshape(y_downsampled, (n))
        #print(y.shape, y_downsampled.shape)


        kgp = kiss_gp.kiss_gp(x, u_mesh, cov_func, y, indexing_type=False)

        sig_var = .8
        ell = 12.5
        noise_var = 0.07        

        lik = kgp.likelihood([sig_var, ell], [noise_var])
        
        # Compare posterior predictive means
        f_mean, f_var = kgp.fit(x_test, calc_var = True)
        f_mean = np.reshape(f_mean, (-1, 3))

        gp = cov_div_free.cov_div_free(sig_var, ell, noise_var)
        gp.init(x, y)
        f_mean_expected, f_var_expected = gp.fit(x_test, calc_var = True)
        f_mean_expected = np.reshape(f_mean_expected, (-1, 3))

        np.testing.assert_almost_equal(f_mean, f_mean_expected, 7)
        np.testing.assert_almost_equal(f_var, f_var_expected, 7)

        # Compare likelihoods
        lik_expected = gp.calc_loglik(x, y)
        np.testing.assert_almost_equal(lik, lik_expected)

        
        
if __name__ == '__main__':
    unittest.main()
