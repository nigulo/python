# -*- coding: utf-8 -*-
"""
Created on Tue May 16 10:54:30 2017

@author: olspern1
"""

import numpy as np
import numpy.linalg as la
#from toeplitz_cholesky import toeplitz_cholesky_lower
#from toeplitz_cholesky import toeplitz_cholesky_upper

class GPR_div_free:
    
    def __init__(self, sig_var, length_scale, noise_var, toeplitz = False):
        self.sig_var = sig_var
        self.length_scale = length_scale
        self.inv_length_scale_sq = 1.0/(length_scale*length_scale)
        self.inv_length_scale_qb = self.inv_length_scale_sq/length_scale
        self.noise_var = noise_var
        self.toeplitz = toeplitz

    def calc_cov(self, x1, x2, data_or_test, calc_grad = False):
        if calc_grad:
            # We only calculate gradients in the constant noise case
            assert(np.isscalar(self.noise_var))
        K = np.zeros((np.size(x1), np.size(x2)))
        # Gradients for sigvar, length_scale and noise_var
        K_grads = np.zeros((3, np.size(x1), np.size(x2)))
        for i in np.arange(0, np.shape(x1)[0]):
            for j in np.arange(0, np.shape(x2)[0]):
                x_diff = x1[i] - x2[j]
                x_diff_sq = np.dot(x_diff, x_diff)
                for i1 in np.arange(0, np.shape(x1)[1]):
                    i_abs = 2*i + i1
                    for j1 in np.arange(0, np.shape(x2)[1]):
                        j_abs = 2*j + j1
                        K[i_abs, j_abs] = x_diff[i1] * x_diff[j1] * self.inv_length_scale_sq
                        K_grads[1, i_abs, j_abs] = -2.0 * x_diff[i1] * x_diff[j1] * self.inv_length_scale_qb
                        if (i1 == j1):
                            K[i_abs, j_abs] += 1 - x_diff_sq * self.inv_length_scale_sq
                            K_grads[1, i_abs, j_abs] += 2.0 * x_diff_sq * self.inv_length_scale_qb
                        exp_fact = np.exp(-0.5 * self.inv_length_scale_sq * x_diff_sq)
                        K_grads[0, i_abs, j_abs] = K[i_abs, j_abs] * exp_fact
                        K_grads[1, i_abs, j_abs] += K[i_abs, j_abs] * x_diff_sq * self.inv_length_scale_qb
                        K_grads[1, i_abs, j_abs] *= self.sig_var * exp_fact
                        K[i_abs, j_abs] *= self.sig_var * exp_fact
                
        if data_or_test:
            assert(np.size(x1) == np.size(x2))
            if (np.isscalar(self.noise_var)):
                K += np.identity(np.size(x1))*self.noise_var
            else:
                K += np.diag(self.noise_var)
            K_grads[2, :, :] += np.identity(np.size(x1))
        if calc_grad:
            return K, K_grads
        else:
            return K

    def init(self, x, y):
        self.n = np.shape(x)[0]
        if (len(np.shape(x)) > 1):
            self.k = np.shape(x)[1]
        else:
            self.k = 1
        self.x = x
        self.y = y
        self.y_flat = np.reshape(y, (self.k*self.n, -1))
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
        self.loglik = np.asscalar(-0.5 * np.dot(self.y_flat.T, self.alpha) - sum(np.log(np.diag(self.L))) - 0.5 * self.n * np.log(2.0 * np.pi))
        return self.loglik

    def fit(self, x_test):
        K_test = self.calc_cov(x_test, self.x, False)
        f_mean = np.dot(K_test, self.alpha)
        v = la.solve(self.L, K_test.T)
        covar = self.calc_cov(x_test, x_test, False) - np.dot(v.T, v)
        var = np.diag(covar)
        return (f_mean, var)
    
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
