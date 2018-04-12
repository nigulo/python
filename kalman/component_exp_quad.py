import numpy as np
from cov_exp_quad import cov_exp_quad

try:
    from scipy.linalg import solve_lyapunov as solve_continuous_lyapunov
except ImportError:  # pragma: no cover; github.com/scipy/scipy/pull/8082
    from scipy.linalg import solve_continuous_lyapunov

class component_exp_quad():
    
    def __init__(self, sig_var, ell, N=6):
        self.ell = ell
        self.sig_var = sig_var
        self.N = N

    def get_params(self):
    
        k = cov_exp_quad(N=self.N)
        F, q = k.get_F_q(self.sig_var, self.ell)

        n_dim = np.shape(F)[0]
        Q_c = np.ones((1, 1))*q
        L = np.zeros((n_dim, 1))
        L[n_dim - 1] = 1.0
        
        H = np.zeros(n_dim) # observatioanl matrix
        H[0] = 1.0
        
        m_0 = np.zeros(n_dim) # zeroth state mean
        #P_0 = np.diag(np.ones(n_dim))#*sig_var) # zeroth state covariance
        
        #P_0 = solve_continuous_lyapunov(F, -np.dot(L, np.dot(Q_c, L.T)))
        P_0 = solve_continuous_lyapunov(F, -np.dot(L, L.T)*Q_c[0,0])
        #print P_0
        
        return F, L, H, m_0, P_0, Q_c

