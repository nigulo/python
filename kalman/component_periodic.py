import numpy as np
import scipy.special as special

class component_periodic():
    
    def __init__(self, j_max, sig_var, omega_0, ell):
        self.j_max = j_max
        self.sig_var = sig_var
        self.omega_0 = omega_0
        self.ell = ell

    def get_params_j(self, j):
        ell_inv_sq = 1.0/self.ell/self.ell
        I2 = np.diag(np.ones(2))
    
        Fpj = np.array([[0.0, -self.omega_0*j], [self.omega_0*j, 0.0]])
    
        Lpj = I2
    
        if j == 0:
            qj = special.iv(0, ell_inv_sq)/np.exp(ell_inv_sq)
        else:
            qj = 2.0 * special.iv(j, ell_inv_sq)/np.exp(ell_inv_sq)
    
        P0pj = I2 * qj
        #Qcpj = np.zeros((2, 2))
        Hpj = np.array([1.0, 0.0])
        
        return Fpj, Lpj, P0pj, Hpj, qj
    
    
    def get_params(self):
        
        F = np.zeros((2*self.j_max, 2*self.j_max))
        L = np.zeros((2*self.j_max, 2*self.j_max))
        Q_c = np.zeros((2*self.j_max, 2*self.j_max)) # process noise
        H = np.zeros(2*self.j_max) # observatioanl matrix
        
        m_0 = np.zeros(2*self.j_max) # zeroth state mean
        P_0 = np.zeros((2*self.j_max, 2*self.j_max)) # zeroth state covariance
        
        for j in np.arange(0, self.j_max+1):
            Fpj, Lpj, P0pj, Hpj, _ = self.get_params_j(j)
            F[2*j:2*j+2, 2*j:2*j+2] = Fpj
            L[2*j:2*j+2, 2*j:2*j+2] = Lpj
            P_0[2*j:2*j+2, 2*j:2*j+2] = P0pj*self.sig_var
            H[2*j:2*j+2] = Hpj
    
            #ell_inv_sq = 1.0/ell/ell
            #F[2*j, 2*j+1] = -omega_0*j
            #F[2*j+1, 2*j] = omega_0*j
            #L[2*j, 2*j] = 1.0
            #L[2*j+1, 2*j+1] = 1.0
            #if j == 0:
            #    q = special.iv(0, ell_inv_sq)/np.exp(ell_inv_sq)
            #else:
            #    q = 2.0 * special.iv(j, ell_inv_sq)/np.exp(ell_inv_sq)
            #P_0[2*j, 2*j] = q
            #P_0[2*j+1, 2*j+1] = q
            #H[2*j] = 1.0
        return F, L, H, m_0, P_0, Q_c
