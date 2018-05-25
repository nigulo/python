import numpy as np
import scipy.special as special

from component_exp_quad import component_exp_quad
from component_periodic import component_periodic

class component_quasiperiodic():
    
    def __init__(self, j_max, sig_var, omega_0, ellp, ellq):
        self.num_j = j_max+1
        self.omega_0 = omega_0
        self.ellp = ellp
        self.ellq = ellq
        self.sig_var = sig_var

        self.Nq = 6
        self.exp_quad = component_exp_quad(sig_var=sig_var, ell=ellq, N=self.Nq)
        self.periodic = component_periodic(j_max=j_max, sig_var=1.0, omega_0=omega_0, ell=ellp)
        
    def get_params(self):
        #ell_inv_sq = 1.0/ell/ell
        Fq, Lq, Hq, _, P0q, Qcq = self.exp_quad.get_params()
        
        #lmbda = 1.0/ellq
        #Fq = -np.ones(1) * lmbda
        #Lq = np.ones(1)
        #Qcq = np.ones(1) * 2.0*sig_var*np.sqrt(np.pi)*lmbda*special.gamma(1.0)/special.gamma(0.5)
        #Hq = np.ones(1) # observatioanl matrix
        #P0q = np.ones(1)
        
        F = np.zeros((2*self.Nq*self.num_j, 2*self.Nq*self.num_j))
        L = np.zeros((2*self.Nq*self.num_j, 2*self.num_j))
        Q_c = np.zeros((2*self.num_j, 2*self.num_j)) # process noise
        H = np.zeros(2*self.Nq*self.num_j) # observatioanl matrix
        
        m_0 = np.zeros(2*self.Nq*self.num_j) # zeroth state mean
        P_0 = np.zeros((2*self.Nq*self.num_j, 2*self.Nq*self.num_j)) # zeroth state covariance
        
        I2 = np.diag(np.ones(2))
        for j in np.arange(0, self.num_j):
            Fpj, Lpj, P0pj, Hpj, qj = self.periodic.get_params_j(j)
            #Fpj = np.array([[0.0, -omega_0*j], [omega_0*j, 0.0]])
        
            #Lpj = I2
        
            #if j == 0:
            #    q = special.iv(0, ell_inv_sq)/np.exp(ell_inv_sq)
            #else:
            #    q = 2.0 * special.iv(j, ell_inv_sq)/np.exp(ell_inv_sq)
    
            #P0pj = I2 * q
            ##Qcpj = np.zeros((2, 2))
            #Hpj = np.array([1.0, 0.0])
            
            F[2*self.Nq*j:2*self.Nq*j+2*self.Nq, 2*self.Nq*j:2*self.Nq*j+2*self.Nq] = np.kron(Fq, I2) + np.kron(np.diag(np.ones(self.Nq)), Fpj)
            L[2*self.Nq*j:2*self.Nq*j+2*self.Nq, 2*j:2*j+2] = np.kron(Lq, Lpj)
            Q_c[2*j:2*j+2, 2*j:2*j+2] = np.kron(Qcq, I2 * qj)
            P_0[2*self.Nq*j:2*self.Nq*j+2*self.Nq, 2*self.Nq*j:2*self.Nq*j+2*self.Nq] = np.kron(P0q, P0pj)
            H[2*self.Nq*j:2*self.Nq*j+2*self.Nq] = np.kron(Hq, Hpj)
            
        return F, L, H, m_0, P_0, Q_c
