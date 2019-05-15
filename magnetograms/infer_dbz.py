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

        self.bz1 = np.zeros_like(self.bz_flat)
        

        self.sig_var = 1.
        self.length_scale = 2.
        self.noise_var = 0.01
        
        gp = cov_sq_exp.cov_sq_exp(self.sig_var, self.length_scale, self.noise_var)

        self.kgp = kiss_gp.kiss_gp(x, u_mesh, u, lambda _, _, _, u1, u2, data_or_test: gp.calc_cov(u, u2, data_or_test=data_or_test, calc_grad = True))

    def calc(self):
        y1 = (self.bz_flat - self.bz1)/ (self.div_xy_flat + 1e-15)

        _ = self.kgp.likelihood([self.length_scale, self.sig_var], [self.noise_var, y1])
        dz_mean, f_var = self.kgp.fit(self.x, calc_var = False)
        
        
        y1 = self.bz_flat - self.div_xy_flat * dz_mean
        _ = self.kgp.likelihood([self.length_scale, self.sig_var], [self.noise_var, y1])
        bz1,  _ = self.kgp.fit(self.x, calc_var = False)
        self.bz1 = bz1
        
        
        
