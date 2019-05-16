import numpy as np
import scipy.signal as signal
import cov_sq_exp
import kiss_gp

class infer_dbz():
    
    def __init__(self, x, u_mesh, u, bx, by, bz):
        self.x = x
        n1 = x.shape[0]
        n2 = x.shape[1]
        n = n1*n2
        #bx = np.reshape(bxy[:, 0], (n1, n2))
        #by = np.reshape(bxy[:, 1], (n1, n2))
        bx_smooth = signal.convolve2d(bx, np.ones((5,5)), mode = 'same') #Smooth it a little
        by_smooth = signal.convolve2d(by, np.ones((5,5)), mode = 'same') #Smooth it a little
        dbxy = bx_smooth[1:,:-1]-bx_smooth[:-1,:-1]
        dbyy = by_smooth[:-1,1:]-by_smooth[:-1,:-1]
        div_xy = dbxy + dbyy
        
        self.div_xy_flat = np.reshape(div_xy, (n))
        self.bz_flat = np.reshape(bz, (n))

        self.sig_var = 1.
        self.length_scale = 2.
        self.noise_var = 0.01
        
        self.gp = cov_sq_exp.cov_sq_exp(self.sig_var, self.length_scale, self.noise_var)

        self.kgp = kiss_gp.kiss_gp(x, u_mesh, u, self.calc_cov)

    def calc_cov(self, sig_var, length_scale, noise_var, u1, u2, data_or_test):
        # Arguments are actually unused
        assert(sig_var == self.sig_var)
        assert(length_scale == self.length_scale)
        assert(noise_var == self.noise_var)
        K, K_grads = self.gp.calc_cov(u1, u2, data_or_test=data_or_test, calc_grad = False)
        K1 = self.div_xy_flat*(self.div_xy_flat*K.T) + K
        np.testing.assert_almost_equal(np.dot(self.div_xy_flat, np.dot(K, np.diag(self.div_xy_flat))), K1)
        
        K1_grads = np.zeros_like(K_grads)
        for i in K1_grads.shape[0]:
            K1_grads[i] = self.div_xy_flat*(self.div_xy_flat*K_grads[i].T) + K_grads[i]
        
        return K1, K1_grads


    def calc(self):

        _ = self.kgp.likelihood([self.length_scale, self.sig_var], [self.noise_var, self.bz_flat])
        bz_mean, f_var = self.kgp.fit(self.x, calc_var = False)
        
        return self.bz_flat - bz_mean
        
