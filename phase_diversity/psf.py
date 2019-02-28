import numpy as np
import numpy.fft as fft
from scipy.signal import correlate2d as correlate
import zernike
import utils
import sys

class phase_aberration():
    
    def __init__(self, coefs):
        self.terms = []
        for index, coef in coefs:
            n, m = zernike.get_nm(index)
            z = zernike.zernike(n, m)
            self.terms.append((coef, z))
            
    def __call__(self, xs):
        scalar = False
        if len(np.shape(xs)) == 1:
            scalar = True
            xs = np.array([xs])
        vals = np.zeros(np.shape(xs)[:-1])
        rhos_phis = utils.cart_to_polar(xs)
        for coef, z in self.terms:
            #TODO vectorize zernike
            vals += z.get_value(rhos_phis) * coef
        if scalar:
            vals = vals[0]
        return vals


class wavefront():
    
    def __init__(self, data):
        self.data = data
        
    def __call__(self, xs):
        # Ignore the coordinates, just return data
        # We assume that the wavefront array is properly 
        # aligned with the coordinates
        return self.data


class coh_trans_func():

    def __init__(self, pupil_func, phase_aberr, phase_div = None):
        self.pupil_func = pupil_func
        self.phase_aberr = phase_aberr
        self.phase_div = phase_div
       
    def __call__(self, xs, defocus = False):
        #if self.pupil_func(u) == 0:
        #    return 0.0
        #else:
        #    return np.exp(1.j * (self.phase_aberr.get_value(u) + self.phase_div(u)))
        #a = np.exp(1.j * (self.phase_aberr.get_value(u) + self.phase_div(u)))
        #b = np.cos(self.phase_aberr.get_value(u) + self.phase_div(u)) + 1.j * np.sin(self.phase_aberr.get_value(u) + self.phase_div(u))
        phase = self.phase_aberr(xs)
        if defocus and self.phase_div is not None:
            phase += self.phase_div(xs)
        return self.pupil_func(xs)*np.exp(1.j * phase)

class psf():
    def __init__(self, coh_trans_func, nx, ny):
        self.nx = nx
        self.ny = ny
        #coh_vals = np.zeros((nx, ny))
        xs = np.linspace(-1.0, 1.0, nx)/np.sqrt(2)
        ys = np.linspace(-1.0, 1.0, ny)/np.sqrt(2)
        assert(len(xs) == nx)
        assert(len(ys) == ny)
        self.coords = np.dstack(np.meshgrid(xs, ys))
        self.incoh_vals = dict()
        self.otf_vals = dict()
        self.coh_trans_func = coh_trans_func
        
    def calc(self, defocus = True):
        coh_vals = self.coh_trans_func(self.coords, defocus)
        
        auto = correlate(coh_vals, coh_vals.conjugate(), mode='full')
        #vals = fft.ifft2(fft.ifftshift(auto)).real
        vals = fft.fftshift(fft.ifft2(fft.ifftshift(auto))).real
        vals /= vals.sum()
        self.incoh_vals[defocus] = vals
        return vals

    def calc_otf(self, defocus = True):
        if defocus not in self.incoh_vals:
            self.calc(defocus)
        vals = fft.fft2(self.incoh_vals[defocus])
        #vals = fft.fftshift(vals)
        self.otf_vals[defocus] = vals
        return vals

    def multiply(self, dat_F, defocus = True):
        if False not in self.otf_vals:
            self.calc_otf(False)
        ret_val = dat_F * self.otf_vals[False]
        if not defocus:
            return ret_val
        else:
            if True not in self.otf_vals:
                self.calc_otf(True)
            ret_val_d = dat_F * self.otf_vals[True]
            return (ret_val, ret_val_d)
            
    def convolve(self, dat, betas, defocus = True):
        dat_F = fft.fft2(dat)
        ret_val = []
        for m_F in self.multiply(dat_F, defocus):
            m = fft.fftshift(fft.ifft2(m_F).real)
            ret_val.append(m) 
        if defocus:
            return (ret_val[0], ret_val[1])
        else:
            return ret_val[0]

    def deconvolve(self, D, D_d, gamma, do_fft = True):
        #P = np.roll(np.roll(P, int(self.nx/2), axis=0), int(self.nx/2), axis=1)
        #P_d = np.roll(np.roll(P_d, int(self.nx/2), axis=0), int(self.nx/2), axis=1)
        
        S = self.otf_vals[False]
        S_conj = S.conjugate()
        S_d = self.otf_vals[True]
        S_d_conj = S_d.conjugate()
        
        F_image = D * S_conj + gamma * D_d * S_d_conj
        F_image /= S*S_conj + gamma * S_d * S_d_conj
        
        if not do_fft:
            return F_image

        image = fft.ifft2(F_image).real
        #image = np.roll(np.roll(image, int(self.nx/2), axis=0), int(self.nx/2), axis=1)
        return image


class psf_():
    def __init__(self, coh_trans_func, nx, ny, repeat_factor = 2):
        assert((repeat_factor % 2) == 0)
        self.nx = nx
        self.ny = ny
        #coh_vals = np.zeros((nx, ny))
        xs = np.linspace(-1.0, 1.0, nx)/np.sqrt(2)
        ys = np.linspace(-1.0, 1.0, ny)/np.sqrt(2)
        assert(len(xs) == nx)
        assert(len(ys) == ny)
        coords = np.dstack(np.meshgrid(xs, ys))
        
        #us = np.transpose([np.tile(xs, ny), np.repeat(ys, nx)])
        #coh_vals = coh_trans_func.get_value(us)
        #coh_vals = np.reshape(coh_vals, (nx, ny))

        coh_vals = coh_trans_func(coords)
        
        vals = fft.ifft2(coh_vals)
        #vals = fft.fftshift(vals)
        #vals = vals.real**2 + vals.imag**2
        vals = (vals*vals.conjugate()).real
        #vals = fft.ifft2(vals)
        #vals = fft.ifft2(vals).real
        vals = np.roll(np.roll(vals, int(nx/2), axis=0), int(ny/2), axis=1)
        #vals /= vals.sum()
        self.incoh_vals = vals
        #assert(np.all(self.incoh_vals == np.conjugate(vals)*vals))
        
    def get_incoh_vals(self):
        return self.incoh_vals

    def get_otf_vals(self):
        vals = fft.fft2(self.incoh_vals)
        #vals = np.roll(np.roll(vals, int(self.nx/2), axis=0), int(self.ny/2), axis=1)
        vals = fft.fftshift(vals)
        return np.array([vals.real, vals.imag])
