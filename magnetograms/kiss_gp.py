import numpy as np
import scipy.sparse.linalg as sparse
import utils
import numpy.linalg as la

class kiss_gp:
    
    def __init__(self, x, u_mesh, cov_func, y, dim=2):
        self.n = np.size(x)/2
        self.u_mesh = u_mesh
        self.u = np.dstack(u_mesh).reshape(-1, 2)
        self.cov_func = cov_func
        self.W = utils.calc_W(u_mesh, x, dim=dim)#np.zeros((len(x1)*len(x2)*2, len(u1)*len(u2)*2))
        self.U = None
        self.y = y

    
    def likelihood(self, theta, data):
        self.theta = theta
        self.data = data
        U, U_grads = self.cov_func(self.theta, self.data, self.u, self.u, data_or_test=True)
        self.U_grads = U_grads

        (x, istop, itn, normr) = sparse.lsqr(self.W, self.y)[:4]#, x0=None, tol=1e-05, maxiter=None, M=None, callback=None)
        self.x = x
        #print x
        L = la.cholesky(U)
        self.L = L
        #print Lr_hks_a
        #v = la.solve(L, x)
        #v = la.solve(L, x)
        self.alpha = la.solve(self.L.T, la.solve(self.L, x))

        #return -0.5 * np.dot(v.T, v)

        #L_inv = la.inv(L)
        #U_inv = la.inv(U)
        #np.testing.assert_almost_equal(self.alpha, np.dot(U_inv, x), 5)
        #np.testing.assert_almost_equal(np.dot(v.T, v), np.dot(x.T, np.dot(U_inv, x)), 5)
        #np.testing.assert_almost_equal(np.dot(x.T, self.alpha), np.dot(x.T, np.dot(U_inv, x)), 5)
        #np.testing.assert_almost_equal(np.dot(x.T, self.alpha), np.dot(v.T, v), 5)
        
        return -0.5 * np.dot(x.T, self.alpha) - sum(np.log(np.diag(L))) - 0.5 * self.n * np.log(2.0 * np.pi)
        

    def likelihood_grad(self, theta, data):
        if not np.array_equal(self.theta, theta) or not np.array_equal(self.data, data):
            self.theta = theta
            self.data = data
            U, U_grads = self.cov_func(self.theta, self.data, self.u, self.u, data_or_test=True)
            self.U_grads = U_grads

            (x, istop, itn, normr) = sparse.lsqr(self.W, self.y)[:4]
            self.x = x
            L = la.cholesky(U)
            self.L = L
            self.alpha = la.solve(self.L.T, la.solve(self.L, x))
            
        L_inv = la.inv(self.L)
        #np.testing.assert_almost_equal(L_T_inv, la.inv(L.T), 10)
        
        U_inv = np.dot(L_inv.T, L_inv)
        #np.testing.assert_almost_equal(U_inv, la.inv(self.U), 4)
        
        assert(self.U_grads.shape[0] == len(theta))
        num_params = len(theta)
        ret_val = np.zeros(num_params)

        for param_index in np.arange(0, num_params):

            exp_part = 0.5 * np.dot(self.alpha.T, np.dot(self.U_grads[param_index,:,:], self.alpha))
            det_part = -0.5 * sum(np.diag(np.dot(U_inv, self.U_grads[param_index,:,:])))
            ret_val[param_index] = exp_part + det_part
        
        self.U = None

        return ret_val


    def fit(self, x_test, calc_var = True):
        K_test, _ = self.cov_func(self.theta, self.data, x_test, self.u, data_or_test=False)
        f_mean = np.dot(K_test, self.alpha)
        if calc_var:
            v = la.solve(self.L, K_test.T)
            covar, _ = self.cov_func(self.theta, self.data, x_test, x_test, data_or_test=False)
            covar -= np.dot(v.T, v)
            var = np.diag(covar)
            return (f_mean, var)
        else:
            return f_mean
