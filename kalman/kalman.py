import numpy as np
import scipy.misc
import scipy.linalg as la

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 15:36:22 2018

@author: olspern1
"""

class kalman():
    
    def __init__(self, t, y, F, L, H, R, m_0, P_0, Q_c, F_is_A = False):
        assert(np.shape(t)[0] == np.shape(y)[0])
        y_dim_2 = 1
        if np.ndim(y) > 1:
            y_dim_2 = np.shape(y)[1]
        self.y_dim_2 = y_dim_2
        if np.ndim(m_0) == 0:
            m_0 = np.reshape(m_0, 1)
        if np.ndim(Q_c) == 0:
            Q_c = np.reshape(Q_c, (1,1))
        if np.ndim(P_0) == 0:
            P_0 = np.reshape(P_0, (1,1))
        if np.ndim(F) == 0:
            F = np.reshape(F, (1,1))
        if np.ndim(L) == 0:
            L = np.reshape(L, (1,1))
        if np.ndim(H) == 0:
            H = np.reshape(H, 1)
        if np.ndim(H) == 1:
            assert(np.shape(H)[0] == np.shape(F)[0])
            H = np.reshape(H, (1, np.shape(H)[0]))
        if np.ndim(R) == 0:
            R = np.reshape(R, (1,1))

        if y_dim_2 == 1:
            assert(np.shape(H)[1] == np.shape(F)[0])
        else:
            assert(np.ndim(H) == 2)
            assert(np.shape(H)[0] == y_dim_2)
            assert(np.shape(H)[1] == np.shape(F)[0])

        assert(np.shape(R)[0] == y_dim_2)
        assert(np.shape(R)[1] == y_dim_2)

        assert(np.shape(F)[0] == np.shape(L)[0])
        assert(np.shape(L)[1] == np.shape(L)[0])
        assert(np.shape(Q_c)[0] == np.shape(L)[1])
        assert(np.shape(Q_c)[1] == np.shape(L)[1])
        assert(np.shape(F)[1] == np.shape(m_0)[0])
        assert(np.shape(P_0)[0] == np.shape(m_0)[0])
        assert(np.shape(P_0)[1] == np.shape(m_0)[0])
        
        self.t = t
        self.y = y
        self.F = F
        self.L = L
        self.H = H
        self.R = R
        self.m_0 = m_0
        self.P_0 = m_0
        self.Q_c = Q_c
        self.Q_c_is_not_zero = np.count_nonzero(Q_c) > 0

        self.m = np.zeros((len(t), np.shape(m_0)[0]))
        self.P = np.zeros((len(t), np.shape(P_0)[0], np.shape(P_0)[1]))
        self.m[0] = m_0
        self.P[0] = P_0
        
        self.A = np.zeros((len(t)-1, np.shape(F)[0], np.shape(F)[1]))
        self.m_ = np.zeros((len(t)-1, np.shape(m_0)[0]))
        self.P_ = np.zeros((len(t)-1, np.shape(P_0)[0], np.shape(P_0)[1]))

        self.ms = np.zeros((len(t), np.shape(m_0)[0]))
        self.Ps = np.zeros((len(t), np.shape(P_0)[0], np.shape(P_0)[1]))

        self.F_is_A = F_is_A

    def phi(self, tau):
        return la.expm(self.F*tau)
    
        
    def filter(self):
        self.k = 1
        y_means = np.zeros(len(self.t)-1)
        loglik = 0.0
        
        for step in np.arange(0, len(self.t)-1):
            (y_mean, S, y_loglik) = self.filter_step()
            y_means[step] = y_mean
            loglik += y_loglik
        return y_means, loglik
    
    def filter_step(self):
        #print "step", self.k
        delta_t = self.t[self.k] - self.t[self.k-1]
        if self.F_is_A:
            A = self.F
        else:
            A = self.phi(delta_t)
        self.A[self.k-1] = A
        
        #print A[2,2] - np.cos(self.F[2,3]*delta_t)
        #print A[2,3] - np.sin(self.F[2,3]*delta_t)
        #print A[3,2] + np.sin(self.F[2,3]*delta_t)
        #print A[3,3] - np.cos(self.F[2,3]*delta_t)
        

        Q = np.zeros((np.shape(self.L)[1], np.shape(self.L)[1]))
        if self.Q_c_is_not_zero:
            d_tau = delta_t/1000
            for tau in np.linspace(0, delta_t, 1000):
                Phi = self.phi(delta_t-tau)
                Q += np.dot(Phi, np.dot(self.L, np.dot(self.Q_c, np.dot(self.L.T, Phi.T))))
            Q *= d_tau


        m = self.m[self.k-1]
        P = self.P[self.k-1]

        # prediction step
        m_ = np.dot(A, m)
        P_ = np.dot(A, np.dot(P, A.T)) + Q
        self.m_[self.k-1] = m_
        self.P_[self.k-1] = P_
        
        # update step
        y_mean = np.dot(self.H, m_)
        v = self.y[self.k] - y_mean

        S = np.dot(self.H, np.dot(P_, self.H.T)) + self.R
        S_inv = la.inv(S)

        loglik_y = -0.5 * np.dot(v.T, np.dot(S_inv, v)) - 0.5 * la.det(S) - 0.5 * self.y_dim_2 * np.log(2.0 * np.pi)
        #print S, S_inv

        K = np.dot(P_, np.dot(self.H.T, S_inv))
        self.m[self.k] = m_ + np.dot(K, v)
        self.P[self.k] = P_ - np.dot(K, np.dot(S, K.T))
        #self.P[self.k] = P_ - np.dot(K, np.dot(self.H, P_))

        self.k += 1

        return (y_mean, S, loglik_y)

    def smooth(self):
        self.k = len(self.t) - 2
        
        self.ms[-1] = self.m[-1]
        self.Ps[-1] = self.P[-1]
        
        y_means = np.zeros(len(self.t)-2)
        
        for step in np.arange(len(self.t)-2, 0, step=-1):
            print step
            (y_mean, S) = self.smooth_step()
            y_means[step-1] = y_mean
        return y_means

    def smooth_step(self):
        m_ = self.m_[self.k]
        P_ = self.P_[self.k]
        A = self.A[self.k]
        
        P = self.P[self.k]
        
        G = np.dot(P, np.dot(A, la.inv(P_)))
        ms = self.m[self.k] + np.dot(G, self.ms[self.k + 1] - m_)
        Ps = P + np.dot(G, np.dot(self.Ps[self.k + 1] - P_, G.T))

        print ms
        self.ms[self.k] = ms
        self.Ps[self.k] = Ps
        
        y_mean = np.dot(self.H, ms)
        S = np.dot(self.H, np.dot(Ps, self.H.T)) + self.R
        
        self.k -= 1
        return (y_mean, S)
