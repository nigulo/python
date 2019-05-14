import numpy as np
import scipy.signal as signal
import cov_sq_exp
import kiss_gp

class infer_dbz():
    
    def __init__(self, x, u_mesh, u, bx, by, bz):
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
        
        div_xy_flat = np.reshape(div_xy, (n))
        bz_flat = np.reshape(bz, (n))

        
        self.y1 = bz_flat / (div_xy_flat + 1e-15)

        sig_var = 1.
        length_scale = 2.
        noise_var = 0.01
        gp = cov_sq_exp.cov_sq_exp(sig_var, length_scale, noise_var)

        self.kgp = kiss_gp.kiss_gp(x, u_mesh, u, lambda sig_var, ell, noise_var, u: gp.calc_cov(u, u, data_or_test=True, calc_grad = True))

    def cov_func(self, sig_var, ell, noise_var, u):
        U, U_grads = self.gp.calc_cov(u, u, data_or_test=True, calc_grad = True)
        return  U, U_grads
        
    def calc(self):
        
        

loglik = gp.init(x, y)
y_test_mean1 = gp.fit(x_test, calc_var = False)
y_test_mean1 = np.reshape(y_test_mean1, (n_test, -1))

kgp = kiss_gp.kiss_gp(x, u_mesh, u, cov_func)
kgp.likelihood2(params, [y])


