# -*- coding: utf-8 -*-
"""
Created on Tue May 16 10:54:30 2017

@author: olspern1
"""

import numpy as np
import numpy.linalg as la

class cov_func:
    
    def __init__(self, toeplitz = False):
        self.toeplitz = toeplitz

    def calc_cov(self, x1, x2, data_or_test, calc_grad = False):
        raise Exception('Not implemented')

    def calc_cov_ij(self, x1, x2, i, j):
        raise Exception('Not implemented')
        

    def init(self, x, y):
        self.n = np.shape(x)[0]
        self.x = x
        self.y = y
        self.y_flat = y.flatten()
        self.K = self.calc_cov(x, x, True)
        if self.toeplitz:
            #L1 = toeplitz_cholesky_lower(np.shape(self.K)[0], self.K)
            #L2 = toeplitz_cholesky_upper(np.shape(self.K)[0], self.K)
            self.L = la.cholesky(self.K)
            #print "Comparison:"
            #print self.K
            #print self.L
            #print np.dot(self.L, self.L.T)-self.K
            #print np.dot(L1, L1.T)/2-self.K
            #print L2
        else:
            self.L = la.cholesky(self.K)
        self.alpha = la.solve(self.L.T, la.solve(self.L, self.y_flat))
        self.loglik = -0.5 * np.dot(self.y_flat, self.alpha) - sum(np.log(np.diag(self.L))) - 0.5 * self.n * np.log(2.0 * np.pi)
        return self.loglik
    
    def calc_loglik(self, x, y):
        return self.init(x, y)

    
    '''
        This is pairwise factor graph approximation to the log-likelihood
    '''
    def calc_loglik_approx(self, x, y, use_vector_form = False, subsample = 0):
        if use_vector_form:
            if not hasattr(self, 'K'):
                self.K=self.calc_cov(x, x, True)
            if not hasattr(self, 'G1'):
                self.calc_G(x)
            ones = np.ones(len(y))
            return -np.dot(ones, np.dot(self.G1, y*y)) + np.dot(y.T, np.dot(self.G2, y)) - self.G_log
        else:
            loglik = 0.
            sigma = self.calc_cov_ij(x, x, 0, 0)
            sigma2 = sigma*sigma
            inds = np.arange(0, np.size(x))
            if subsample > 0 and subsample < np.size(x)*np.size(x):
                subsample = max(np.size(x) - 1, subsample)
                ijs = np.column_stack((inds[:-1], inds[1:]))
                subsample -= len(ijs)
                i_s = np.random.choice(np.arange(0, np.size(x)), size=int(np.sqrt(subsample)))
                j_s = (np.random.choice(np.arange(2, np.size(x) + 2), size=int(np.sqrt(subsample))) + i_s) % np.size(x)
                print("ijs", ijs.shape)
                ijs = np.concatenate((ijs, np.transpose([np.tile(i_s, len(j_s)), np.repeat(j_s, len(i_s))])))
                print("ijs", ijs.shape)
            else:
                ijs = np.transpose([np.tile(inds, len(inds)), np.repeat(inds, len(inds))])
                #i_s = np.arange(0, np.size(x))
                #j_s = np.arange(0, np.size(x))
            print(ijs)
            for i, j in ijs:
                if j < i:
                    continue
                K_ij = self.calc_cov_ij(x, x, i, j)
                if i == j:
                    assert(K_ij == sigma) # In this approximation we assume constant variance
                else:
                    K_ij2 = K_ij*K_ij
                    val = sigma2 - K_ij2
                    loglik += -(sigma*(y[i]*y[i] + y[j]*y[j]) - 2.*K_ij*y[i]*y[j])/val - np.log(val) - np.log(2.*np.pi)
            
            if subsample > 0:
                loglik *= np.size(x)*np.size(x)/(subsample)
            return loglik                        
                    

    def calc_G(self, x):
        G1 = np.zeros_like(self.K)
        G2 = np.zeros_like(self.K)
        G_log = 0.
        sigma = self.K[0, 0]
        sigma2 = sigma*sigma
        for i in np.arange(0, self.K.shape[0]):
            assert(self.K[i, i] == sigma) # In this approximation we assume constant variance
            for j in np.arange(i + 1, self.K.shape[1]):
                K_ij2 = self.K[i, j]*self.K[i, j]
                val = sigma2 - K_ij2
                G_log += 2*np.log(val) 
                G1[i, j] = sigma/val
                G2[i, j] = self.K[i, j]/val

                G1[j, i] = G1[i, j]
                G2[j, i] = G2[i, j]
        self.G1 = G1
        self.G2 = G2
        self.G_log = 0.5*G_log + (self.K.shape[0]*(self.K.shape[0]-1)/2)*np.log(2.*np.pi)
        

    def fit(self, x_test, calc_var = True):
        K_test = self.calc_cov(x_test, self.x, False)
        f_mean = np.dot(K_test, self.alpha)
        if calc_var:
            v = la.solve(self.L, K_test.T)
            covar = self.calc_cov(x_test, x_test, False) - np.dot(v.T, v)
            var = np.diag(covar)
            return (f_mean, var)
        else:
            return f_mean
    
    def cv(self, x_test, y_test, noise_test):
        if np.isscalar(x_test):
            x_test = np.array([x_test])
            y_test = np.array([y_test])
            noise_test = np.array([noise_test])
        n_test = np.shape(x_test)[0]
        if (len(np.shape(x_test)) > 1):
            k_test = np.shape(x_test)[1]
        else:
            k_test = 1
        y_test_flat = np.reshape(y_test, (k_test*n_test, -1))
        
        K_test = self.calc_cov(x_test, self.t, False)
        f_mean = np.dot(K_test, self.alpha)
        v = la.solve(self.L, K_test.T)
        covar = self.calc_cov(x_test, x_test, False) - np.dot(v.T, v)
        covar += np.diag(noise_test)
        #covar = np.diag(np.diag(covar))
        #print self.noise_var[0:3]
        #print self.calc_cov(t_test, t_test, False)[0:3,0:3]
        #print np.dot(v.T, v)[0:3,0:3]
        #print covar
        #print np.linalg.eigvalsh(covar)
        #inv_covar = la.inv(covar)
        #print np.dot(inv_covar.T, covar)
        #print inv_covar
        var = np.diag(covar)
        L_test_covar = la.cholesky(covar)
        alpha_test_covar = la.solve(L_test_covar.T, la.solve(L_test_covar, (y_test_flat-f_mean)))
        loglik = -0.5 * np.dot((y_test_flat-f_mean).T, alpha_test_covar) - sum(np.log(np.diag(L_test_covar))) - 0.5 * n_test * np.log(2.0 * np.pi)
        #det_covar = la.det(covar)
        #print det_covar
        #loglik = -0.5 * np.dot((y_test-f_mean).T, np.dot(inv_covar, y_test-f_mean)) - 0.5 * np.log(det_covar) - 0.5 * n_test * np.log(2.0 * np.pi)
        #print loglik
        return (f_mean, var, loglik)


    def add(self, other_cov_func):
        
        self.old_calc_cov = self.calc_cov
        
        def new_cov(x1, x2, data_or_test, calc_grad = False):
            if calc_grad:
                K, K_grads = self.old_calc_cov(x1, x2, data_or_test, calc_grad)
                K1, K1_grads = other_cov_func.calc_cov(x1, x2, data_or_test, calc_grad)
                return K + K1, np.concatenate((K_grads, K1_grads))
            else:
                K = self.old_calc_cov(x1, x2, data_or_test, calc_grad)
                K1 = other_cov_func.calc_cov(x1, x2, data_or_test, calc_grad)
                return K + K1
        
        self.calc_cov = new_cov