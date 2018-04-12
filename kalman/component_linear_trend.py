import numpy as np

class component_linear_trend():
    
    def __init__(self, slope, intercept, t):
        self.slope = slope
        self.intercept = intercept
        self.t = t
        self.delta_t = t[1:] - t[:-1]

    def get_params(self):
        
        Q_c = np.zeros((1, 1))
        L = np.zeros((2, 1))
        L[1] = 1.0
        
        H = np.array([0.0, 1.0]) # observatioanl matrix
        
        F = np.zeros((len(self.delta_t), 2, 2))
        i = 0
        for dt in self.delta_t:
            F[i] = np.array([[1, 0], [self.slope*dt,1]])
            i += 1
        #P_0 = np.diag(np.ones(n_dim))#*sig_var) # zeroth state covariance
        
        #P_0 = solve_continuous_lyapunov(F, -np.dot(L, np.dot(Q_c, L.T)))
        m_0 = np.array([1.0, self.slope*self.t[0] + self.intercept]) # zeroth state mean
        P_0 = np.diag(np.array([0.0, 0.0]))#slope**2*t[0]**2 + intercept**2]))
        #print P_0
        
        #Q_c[n_dim - 1, n_dim - 1] = q
    
        return F, L, H, m_0, P_0, Q_c
