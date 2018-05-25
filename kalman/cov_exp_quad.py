import numpy as np
import scipy.misc as misc

class cov_exp_quad():

    def __init__(self, N = 6):
        assert(N % 2 == 0)
        assert(N <= 20)
        self.N = N

        self.coefs = np.zeros(2*N + 1)
        self.coefs[2*N] = 1.0
        for n in np.arange(0, N):
            n2 = 2*n
            self.coefs[n2] = float(np.prod(np.arange(n + 1, N + 1)))
            if (n % 2 != 0):
                self.coefs[n2] = -self.coefs[n2]
        
        self.N_fact = misc.factorial(N)

    def get_F_q(self, sigma, ell, return_P = False):
        kappa = 1.0/(2.0*ell*ell)
        
        P_coefs = np.array(self.coefs)
        for n in np.arange(0, self.N + 1):
            P_coefs[2*n] *= (4.0 * kappa)**(self.N-n)
        
        P_coefs = P_coefs[::-1]#np.flip(P_coefs, axis=0) # np.roots takes the coefs in the opposite order

        roots = np.roots(P_coefs)
        positive_roots = []
        negative_roots = []
        for root in roots:
            assert(root.real != 0)
            if root.real > 0:
                positive_roots.append(root)
            else:
                negative_roots.append(root)
        
        P_plus_coefs = np.poly(positive_roots)
        P_minus_coefs = np.poly(negative_roots)
        
        m = len(P_minus_coefs) - 1
        
        F = np.zeros((m, m))
        
        for i in np.arange(0, m - 1):
            F[i, i+1] = 1.0
        
        for i in np.arange(0, m):
            F[m - 1,i] = -P_minus_coefs[m-i]
        #F[m - 1,:] = -P_minus_coefs
            
        q = sigma*self.N_fact*(4.0*kappa)**self.N*np.sqrt(np.pi/kappa)

        if return_P:
            return (F, q, P_coefs, P_plus_coefs, P_minus_coefs)
        else:
            return (F, q)
