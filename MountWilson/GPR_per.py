import numpy as np
import numpy.linalg as la

class GPR_per:
    
    def __init__(self, sig_var, length_scale, freq, noise_var, rot_freq=0.0, rot_amplitude=0.0, trend_var=0.0, c=0.0):
        self.sig_var = sig_var
        self.length_scale = length_scale
        self.freq = freq
        self.noise_var = noise_var
        self.rot_freq = rot_freq
        self.rot_amplitude = rot_amplitude
        self.trend_var = trend_var
        self.c=c

    def calc_cov(self, t1, t2, data_or_test):
        K = np.zeros((len(t1), len(t2)))
        for i in np.arange(0, len(t1)):
            for j in np.arange(0, len(t2)):
                if self.sig_var > 0 and self.freq > 0:
                    K[i, j] = self.sig_var * np.exp(-2.0/self.length_scale /self.length_scale * np.sin(np.pi*self.freq*(t1[i] - t2[j]))**2)
                if self.trend_var > 0:
                    K[i, j] += self.trend_var * (t1[i] - self.c)*(t2[j] - self.c)
                if self.rot_freq > 0 and self.rot_amplitude > 0:
                    K[i, j] += self.rot_amplitude * np.cos(2.0 * np.pi*self.rot_freq*(t1[i] - t2[j]))
        if data_or_test:
            assert(len(t1) == len(t2))
            if (np.isscalar(self.noise_var)):
                K += np.identity(len(t1))*self.noise_var
            else:
                K += np.diag(self.noise_var)
        return K
    
    def fit(self, t, y, t_test, y_test=None):
        n = len(t)
        K = self.calc_cov(t, t, True)
        L = la.cholesky(K)
        alpha = la.solve(L.T, la.solve(L, y))
        K_test = self.calc_cov(t_test, t, False)
        f_mean = np.dot(K_test, alpha)
        v = la.solve(L, K_test.T)
        covar = self.calc_cov(t_test, t_test, False) - np.dot(v.T, v)
        var = np.diag(covar)
        loglik = -0.5 * np.dot(y.T, alpha) - sum(np.log(np.diag(L))) - 0.5 * n * np.log(2.0 * np.pi)
        return (f_mean, var, loglik)