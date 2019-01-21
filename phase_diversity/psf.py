import numpy as np
import numpy.fft as fft
import zernike
import utils
import scipy.special as special

class phase_aberration():
    
    def __init__(self, coefs):
        self.terms = []
        for index, coef in coefs:
            n, m = zernike.get_nm(index)
            z = zernike.zernike(n, m)
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
        #if self.pupil_func(u) == 0:
        #    return 0.0
        #else:
        #    return np.exp(1.j * (self.phase_aberr.get_value(u) + self.phase_div(u)))
        #a = np.exp(1.j * (self.phase_aberr.get_value(u) + self.phase_div(u)))
        #b = np.cos(self.phase_aberr.get_value(u) + self.phase_div(u)) + 1.j * np.sin(self.phase_aberr.get_value(u) + self.phase_div(u))
        return self.pupil_func(u)*np.exp(1.j * (self.phase_aberr.get_value(u) + self.phase_div(u)))

class psf():
    def __init__(self, coh_trans_func, nx, ny):
        coh_vals = np.zeros((nx, ny))
        for x in np.arange(0, nx):
            for y in np.arange(0, ny):
                norm_x = np.sqrt(2.0)*(float(x) - nx/2) / nx
                norm_y = np.sqrt(2.0)*(float(y) - ny/2) / ny
                coh_vals[x, y] = coh_trans_func.get_value(np.array([norm_x, norm_y]))
        vals = fft.fft2(coh_vals)
        vals = vals.real**2 + vals.imag**2
        vals = np.roll(np.roll(vals, nx/2, axis=0), ny/2, axis=1)
        #vals = fft.ifft2(vals)
        #vals = fft.ifft2(vals)
        self.incoh_vals = vals
        #assert(np.all(self.incoh_vals == np.conjugate(vals)*vals))
        
    def get_incoh_vals(self):
        return self.incoh_vals


def aperture_circ(u, r=1.0, coef=5.0):
    if coef > 0.0:
        return 0.5+0.5*special.erf(coef*(r-np.sqrt(np.sum(u**2))))
    else:
        if np.sum(u**2) <= r*r: 
            return 1.0
        else: 
            return 0.0
    
