# -*- coding: utf-8 -*-
"""
Created on Tue May 16 10:54:30 2017

@author: olspern1
"""

import numpy as np
import numpy.linalg as la

class GPR_QP2:
    
    def __init__(self, sig_vars, length_scales, freqs, noise_var, trend_var=0.0, c=0.0):
        self.sig_vars = sig_vars
        self.length_scales = length_scales
        self.freqs = freqs
        self.noise_var = noise_var
        self.trend_var = trend_var
        self.c=c
        assert(len(sig_vars) == len(length_scales))
        assert(len(sig_vars) == len(freqs))

    def calc_cov(self, t1, t2, data_or_test):
        K = np.zeros((len(t1), len(t2)))
        for i in np.arange(0, len(t1)):
            for j in np.arange(0, len(t2)):
                for sig_var, length_scale, freq in zip(self.sig_vars, self.length_scales, self.freqs):
                    if sig_var > 0 and freq > 0:
                        K[i, j] += sig_var * np.exp(-0.5 * (t1[i] - t2[j])**2/length_scale/length_scale) * np.cos(2.0 * np.pi*freq*(t1[i] - t2[j]))
                    elif sig_var > 0 and freq == 0:
                        K[i, j] += sig_var * np.exp(-0.5 * (t1[i] - t2[j])**2/length_scale/length_scale)
                if self.trend_var > 0:
                    K[i, j] += self.trend_var * (t1[i] - self.c)*(t2[j] - self.c)
        if data_or_test:
            assert(len(t1) == len(t2))
            if (np.isscalar(self.noise_var)):
                K += np.identity(len(t1))*self.noise_var
            else:
                K += np.diag(self.noise_var)
        return K

    def init(self, t, y):
        self.n = len(t)
        self.t = t
        self.y = y
        self.K = self.calc_cov(t, t, True)
        self.L = la.cholesky(self.K)
        self.alpha = la.solve(self.L.T, la.solve(self.L, y))

    def fit(self, t_test):
        K_test = self.calc_cov(t_test, self.t, False)
        f_mean = np.dot(K_test, self.alpha)
        v = la.solve(self.L, K_test.T)
        covar = self.calc_cov(t_test, t_test, False) - np.dot(v.T, v)
        var = np.diag(covar)
        loglik = -0.5 * np.dot(self.y.T, self.alpha) - sum(np.log(np.diag(self.L))) - 0.5 * self.n * np.log(2.0 * np.pi)
        return (f_mean, var, loglik)
    
    def cv(self, t_test, y_test, noise_test):
        if np.isscalar(t_test):
            t_test = np.array([t_test])
            y_test = np.array([y_test])
            noise_test = np.array([noise_test])
        n_test = len(t_test)
        K_test = self.calc_cov(t_test, self.t, False)
        f_mean = np.dot(K_test, self.alpha)
        v = la.solve(self.L, K_test.T)
        covar = self.calc_cov(t_test, t_test, False) - np.dot(v.T, v)
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
        alpha_test_covar = la.solve(L_test_covar.T, la.solve(L_test_covar, (y_test-f_mean)))
        loglik = -0.5 * np.dot((y_test-f_mean).T, alpha_test_covar) - sum(np.log(np.diag(L_test_covar))) - 0.5 * n_test * np.log(2.0 * np.pi)
        #det_covar = la.det(covar)
        #print det_covar
        #loglik = -0.5 * np.dot((y_test-f_mean).T, np.dot(inv_covar, y_test-f_mean)) - 0.5 * np.log(det_covar) - 0.5 * n_test * np.log(2.0 * np.pi)
        print loglik
        return (f_mean, var, loglik)
