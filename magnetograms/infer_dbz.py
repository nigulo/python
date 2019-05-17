import numpy as np
import scipy.signal as signal
import cov_sq_exp
import kiss_gp
import utils

class infer_dbz():
    
    def __init__(self, bx, by, bz):
        
        #bx = np.reshape(bxy[:, 0], (n1, n2))
        #by = np.reshape(bxy[:, 1], (n1, n2))
        bx_smooth = bx#signal.convolve2d(bx, np.ones((5,5)), mode = 'same') #Smooth it a little
        by_smooth = by#signal.convolve2d(by, np.ones((5,5)), mode = 'same') #Smooth it a little
        dbxy = bx_smooth[1:,:-1]-bx_smooth[:-1,:-1]
        dbyy = by_smooth[:-1,1:]-by_smooth[:-1,:-1]
        div_xy = dbxy + dbyy
        
        n1 = div_xy.shape[0]
        n2 = div_xy.shape[1]
        n = n1*n2        
        
        x1_range = 1.0
        x2_range = 1.0
        n = n1*n2
        
        x1 = np.linspace(0, x1_range, n1)
        x2 = np.linspace(0, x2_range, n2)
        x_mesh = np.meshgrid(x2, x1)
        x = np.dstack(x_mesh).reshape(-1, 2)
        
        m1 = 10
        m2 = 10
        
        u1 = np.linspace(0, x1_range, m1)
        u2 = np.linspace(0, x2_range, m2)
        u_mesh = np.meshgrid(u1, u2)
        u = np.dstack(u_mesh).reshape(-1, 2)
        
        self.x = x

        self.div_xy_flat = np.reshape(div_xy, (n))
        self.bz_flat = np.reshape(bz[:-1,:-1], (n))

        self.sig_var1 = np.var(bz)*.9
        self.sig_var2 = np.var(bz)*.1
        self.length_scale = .2
        self.noise_var = 0.0001
        
        self.gp1 = cov_sq_exp.cov_sq_exp(self.sig_var1, self.length_scale, self.noise_var, dim_out=1)
        self.gp2 = cov_sq_exp.cov_sq_exp(self.sig_var2, self.length_scale, self.noise_var, dim_out=1)

        self.kgp = kiss_gp.kiss_gp(x, u_mesh, u, self.calc_cov, dim=1)

        self.div_uv = np.dot(self.kgp.W.T, self.div_xy_flat)

    def calc_cov(self, sig_var, length_scale, noise_var, u1, u2, data_or_test):
        # Arguments are actually unused
        assert(sig_var == self.sig_var1)
        assert(length_scale == self.length_scale)
        assert(noise_var == self.noise_var)
        K1, K1_grads = self.gp1.calc_cov(u1, u2, data_or_test=data_or_test, calc_grad = True)
        K2, K2_grads = self.gp2.calc_cov(u1, u2, data_or_test=data_or_test, calc_grad = True)
        #TODO: optimize
        #K1 = (self.div_uv*(self.div_uv*K.T).T) + K
        if u2.shape[0] == self.x.shape[0]:
            A1 = np.diag(self.div_xy_flat)
        else:
            A1 = np.diag(self.div_uv)

        if u1.shape[0] == self.x.shape[0]:
            A2 = np.diag(self.div_xy_flat)
        else:
            A2 = np.diag(self.div_uv)

        K_out = np.dot(A2, np.dot(K1, A1)) + K2
        #np.testing.assert_almost_equal(np.dot(self.div_uv, np.dot(K, np.diag(self.div_uv))) + K, K1)
        
        K_out_grads = np.zeros_like(K1_grads)
        for i in np.arange(0, K1_grads.shape[0]):
            #TODO: optimize
            #K1_grads[i] = self.div_uv*(self.div_uv*K_grads[i].T) + K_grads[i]
            K_out_grads[i] = np.dot(A2, np.dot(K1_grads[i], A1)) + K2_grads[i]
        
        return K_out, K_out_grads


    def calc(self):

        _ = self.kgp.likelihood([self.length_scale, self.sig_var1], [self.noise_var, self.bz_flat])
        bz_mean, f_var = self.kgp.fit(self.x, calc_var = True)
        
        #return self.bz_flat - bz_mean
        return bz_mean
        
