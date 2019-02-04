import numpy as np
import scipy.sparse.linalg as sparse
import utils
import numpy.linalg as la

class kiss_gp:
    
    def __init__(self, x, u_mesh, u, cov_func):
        self.n = np.size(x)/2
        self.u_mesh = u_mesh
        self.u = u
        self.cov_func = cov_func
        self.W = utils.calc_W(u_mesh, u, x)#np.zeros((len(x1)*len(x2)*2, len(u1)*len(u2)*2))
        self.U = None

    
    def likelihood(self, theta, data):
        ell = theta[0]
        sig_var = theta[1]
        noise_var = data[0]
        y = data[1]
        U, U_grads = self.cov_func(sig_var, ell, noise_var, self.u)
        self.U = U
        self.U_grads = U_grads

        (x, istop, itn, normr) = sparse.lsqr(self.W, y)[:4]#, x0=None, tol=1e-05, maxiter=None, M=None, callback=None)
        self.x = x
        #print x
        L = la.cholesky(U)
        self.L = L
        #print L
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
        assert(self.U is not None)
        L_inv = la.inv(self.L)
        #np.testing.assert_almost_equal(L_T_inv, la.inv(L.T), 10)
        
        U_inv = np.dot(L_inv.T, L_inv)
        #np.testing.assert_almost_equal(U_inv, la.inv(self.U), 4)
        ret_val = np.zeros(2)


        
        # The indices of parameters are different in U_grads that in the return value of this function        
        exp_part = 0.5 * np.dot(self.alpha.T, np.dot(self.U_grads[0,:,:], self.alpha))
        det_part = -0.5 * sum(np.diag(np.dot(U_inv, self.U_grads[0,:,:])))
        ret_val[1] = exp_part + det_part

        exp_part = 0.5 * np.dot(self.alpha.T, np.dot(self.U_grads[1,:,:], self.alpha))
        det_part = -0.5 * sum(np.diag(np.dot(U_inv, self.U_grads[1,:,:])))
        ret_val[0] = exp_part + det_part
        
        self.U = None

        return ret_val

