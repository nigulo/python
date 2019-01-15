import numpy as np
import numpy.fft as fft
import zernike
import utils

class phase_aberration():
    
    def __init__(self, coefs):
        self.terms = []
        for index, coef in coefs:
            m, n = zernike.get_mn(index)
            z = zernike.zernike(m, n)
            self.terms.append((coef, z))
            
    def get_value(self, u):
        val = 0.0
        rho, phi = utils.cart_to_polar(u[0], u[1])
        for coef, z in self.terms:
            val += coef * z.get_value(rho, phi)
        return val


class coh_trans_func():

    def __init__(self, pupil_func, phase_aberr, phase_div):
        self.pupil_func = pupil_func
        self.phase_aberr = phase_aberr
        self.phase_div = phase_div
       
    def get_value(self, u):
        return self.pupil_func(u)*np.exp(1.j * (self.phase_aberr.get_value(u) + self.phase_div(u)))

class psf():
    def __init__(self, coh_trans_func, nx, ny):
        coh_vals = np.zeros((nx, ny))
        for x in np.arange(0, nx):
            for y in np.arange(0, ny):
                coh_vals[x, y] = coh_trans_func.get_value([np.sqrt(2)*(float(x) - nx/2) / nx, np.sqrt(2)*(float(y) - ny/2) / ny])
        vals = np.roll(np.roll(fft.fft2(coh_vals), nx/2, axis=0), ny/2, axis=1)
        self.incoh_vals = vals.real**2 + vals.imag**2
        
    def get_incoh_vals(self):
        return self.incoh_vals

