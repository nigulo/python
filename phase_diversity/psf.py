import numpy as np
import numpy.fft as fft
import zernike
import utils

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

    def __init__(self, pupil_func, phase_aberr, phase_div):
        self.pupil_func = pupil_func
        self.phase_aberr = phase_aberr
        self.phase_div = phase_div
       
    def get_value(self, xs):
        #if self.pupil_func(u) == 0:
        #    return 0.0
        #else:
        #    return np.exp(1.j * (self.phase_aberr.get_value(u) + self.phase_div(u)))
        #a = np.exp(1.j * (self.phase_aberr.get_value(u) + self.phase_div(u)))
        #b = np.cos(self.phase_aberr.get_value(u) + self.phase_div(u)) + 1.j * np.sin(self.phase_aberr.get_value(u) + self.phase_div(u))
        return self.pupil_func(xs)*np.exp(1.j * (self.phase_aberr(xs) + self.phase_div(xs)))

class psf():
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

        coh_vals = coh_trans_func.get_value(coords)
        coh_vals = np.tile(coh_vals, (repeat_factor + 1, repeat_factor + 1))
        print(np.shape(coh_vals))
        middle_x = int(nx*(repeat_factor+1)/2)
        middle_y = int(ny*(repeat_factor+1)/2)
        delta_x = int(nx*repeat_factor/2)
        delta_y = int(ny*repeat_factor/2)
        coh_vals = coh_vals[middle_x-delta_x:middle_x+delta_x,middle_y-delta_y:middle_y+delta_y]
        print(np.shape(coh_vals))
        for i in np.arange(0, np.shape(coh_vals)[0]):
            print(coh_vals[i,:])
        
        
        vals = fft.fft2(coh_vals)
        vals = vals.real**2 + vals.imag**2
        #vals = fft.ifft2(vals)
        #vals = fft.ifft2(vals).real
        vals = np.roll(np.roll(vals, int(nx/2), axis=0), int(ny/2), axis=1)
        vals /= vals.sum()
        self.incoh_vals = vals
        #assert(np.all(self.incoh_vals == np.conjugate(vals)*vals))
        
    def get_incoh_vals(self):
        return self.incoh_vals

    def get_otf_vals(self):
        vals = fft.fft2(self.incoh_vals)
        vals = np.roll(np.roll(vals, int(self.nx/2), axis=0), int(self.ny/2), axis=1)
        return np.array([vals.real, vals.imag])


