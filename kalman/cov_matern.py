import numpy as np
from scipy.special import gamma

class cov_matern():

    def __init__(self, nu = None, p = None):
        if nu is None:
            assert (not p is None)
            nu = p + 0.5
        if p is None:
            assert (not nu is None)
            p = nu - 0.5
        assert(float(int(p)) == p)
        self.nu = nu
        self.p = p

    def get_F_q(self, sigma, ell, return_P = False):
        lmbda = np.sqrt(2.0*self.nu)/ell
        m = int(self.p) + 1
        positive_roots = np.ones(m)*lmbda
        negative_roots = -np.ones(m)*lmbda
        P_plus_coefs = np.poly(positive_roots)
        P_minus_coefs = np.poly(negative_roots)
        
        F = np.zeros((m, m))
        
        for i in np.arange(0, m - 1):
            F[i, i+1] = 1.0
        
        for i in np.arange(0, m):
            F[m - 1,i] = -P_minus_coefs[m-i]
            
        q = 2.0*sigma*np.sqrt(np.pi)*lmbda**(2.0*self.nu)*gamma(self.nu+0.5)/gamma(self.nu)

        if return_P:
            return (F, q, P_plus_coefs, P_minus_coefs)
        else:
            return (F, q)
