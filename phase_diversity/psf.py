import numpy as np
import numpy.fft as fft
from scipy.signal import correlate2d as correlate
import zernike
import utils
import copy

class phase_aberration():
    
    def __init__(self, alphas):
        if len(np.shape(alphas)) == 0:
            self.create_pols(alphas)
        else:
            self.create_pols(len(alphas))
            self.set_alphas(alphas)
    
    def create_pols(self, num):
        self.pols = []
        for i in np.arange(1, num + 1):
            n, m = zernike.get_nm(i)
            z = zernike.zernike(n, m)
            self.pols.append(z)

    def calc_terms(self, xs):
        self.terms = np.zeros(np.concatenate(([len(self.pols)], np.shape(xs)[:-1])))
        i = 0
        rhos_phis = utils.cart_to_polar(xs)
        for z in self.pols:
            self.terms[i] = z.get_value(rhos_phis)
            i += 1

    def set_alphas(self, alphas):
        if len(self.pols) != len(alphas):
            self.create_pols(len(alphas))
        self.alphas = np.array(alphas)
        
            
    def __call__(self):
        vals = np.zeros(np.shape(self.terms)[1:])
        for i in np.arange(0, len(self.terms)):
            vals += self.terms[i] * self.alphas[i]
        return vals
    
    def get_pol_values(self):
        vals = np.zeros(np.concatenate((np.array([len(self.terms)]), np.shape(self.terms[0]))))
        for i in np.arange(0, len(self.terms)):
            vals[i] = self.terms[i]
        return vals


class wavefront():
    
    def __init__(self, data):
        self.data = data
        
    def __call__(self, xs):
        # Ignore the coordinates, just return data
        # We assume that the wavefront array is properly 
        # aligned with the coordinates
        return self.data


'''
Coherent transfer function, also called as generalized pupil function
'''
class coh_trans_func():

    def __init__(self, pupil_func, phase_aberr, defocus_func = None):
        self.pupil_func = pupil_func
        self.phase_aberr = phase_aberr
        self.defocus_func = defocus_func
        
    def calc(self, xs):
        self.phase_aberr.calc_terms(xs)
        self.pupil = self.pupil_func(xs)
        self.defocus = self.defocus_func(xs)
            
    def __call__(self, defocus = False):
        phase = self.phase_aberr()
        if defocus:
            phase += self.defocus
        return self.pupil*np.exp(1.j * phase)


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
        self.coh_trans_func.calc(self.coords)
        
        # Repeat the same for bigger grid
        xs1 = np.linspace(-1.0, 1.0, nx*2-1)/np.sqrt(2)
        ys1 = np.linspace(-1.0, 1.0, ny*2-1)/np.sqrt(2)
        coords1 = np.dstack(np.meshgrid(xs1, ys1))

        self.coh_trans_func1 = copy.deepcopy(self.coh_trans_func)
        self.coh_trans_func1.calc(coords1)
        
        
    def calc(self, defocus = True, normalize = True):
        coh_vals = self.coh_trans_func(defocus)
        
        auto = correlate(coh_vals, coh_vals.conjugate(), mode='full')
        #vals = fft.ifft2(fft.ifftshift(auto)).real
        vals = fft.fftshift(fft.ifft2(fft.ifftshift(auto))).real
        if normalize:
            vals /= vals.sum()
        #vals = scipy.misc.imresize(vals, (vals.shape[0]+1, vals.shape[1]+1))
        self.incoh_vals[defocus] = vals
        return vals
    
    def calc_otf(self, defocus = True, recalc_psf=True):
        if recalc_psf:
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
            
    def convolve(self, dat, defocus = True):
        dat_F = fft.fft2(dat)
        ret_val = []
        for m_F in self.multiply(dat_F, defocus):
            m = fft.ifftshift(fft.ifft2(m_F).real)
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


    '''
        Actually this is negative log likelihood
    '''
    def likelihood(self, theta, data):
        D = data[0]
        D_d = data[1]
        gamma = data[2] # Not used
        alphas = theta
        
        pa = self.coh_trans_func.phase_aberr
        pa.set_alphas(alphas)
        
        self.calc_otf(defocus = False)
        self.calc_otf(defocus = True)
        
        S = self.otf_vals[False]
        S_d = self.otf_vals[True]
        nzi = np.nonzero(np.abs(S) + np.abs(S_d))
        
        num = D[nzi]*S[nzi].conjugate() + D_d[nzi]*S_d[nzi].conjugate()
        num *= num.conjugate()
        den = S[nzi]*S[nzi].conjugate()+ S_d[nzi]*S_d[nzi].conjugate()

        lik = -np.sum((num/den).real) + np.sum((D*D.conjugate() + D_d*D_d.conjugate()).real)
        

        return lik
        

    def likelihood_grad(self, theta, data):
        #regularizer_eps = 1e-10

        D = data[0]
        D_d = data[1]
        gamma = data[2] # Not used
        alphas = theta
        
        pa = self.coh_trans_func.phase_aberr
        pa.set_alphas(alphas)
        self.calc_otf(defocus = False)
        self.calc_otf(defocus = True)
        
        S = self.otf_vals[False]
        S_d = self.otf_vals[True]
        
        nzi = np.where(np.abs(S)+np.abs(S_d) != 0.)
        S_nzi = S[nzi]
        S_d_nzi = S[nzi]
        S_nzi_conj = S_nzi.conjugate()
        S_d_nzi_conj = S_d_nzi.conjugate()
        D_nzi = D[nzi]
        D_d_nzi = D_d[nzi]
        
        Z = np.zeros_like(S)
        Z_d = np.zeros_like(S)

        #S_conj = S.conjugate()
        #S_d_conj = S_d.conjugate()
        SD = D_nzi*S_nzi_conj + D_d_nzi*S_d_nzi_conj
        SD2 = SD*SD.conjugate()
        den = 1./(S_nzi*S_nzi_conj + S_d_nzi*S_d_nzi_conj)**2
        
        SDS = (S_nzi*S_nzi_conj + S_d_nzi*S_d_nzi_conj)*SD
        
        Z[nzi] = (SDS*D_nzi.conjugate()-SD2*S_nzi_conj)*den
        Z_d[nzi] = (SDS*D_d_nzi.conjugate()-SD2*S_d_nzi_conj)*den
        
        self.coh_trans_func1.phase_aberr.set_alphas(alphas)
        H = self.coh_trans_func1(defocus = False)
        H_d = self.coh_trans_func1(defocus = True)

        Z_conv_H = fft.ifft2(fft.fft2(Z)*fft.fft2(H.conjugate()))
        Z_conv_H_d = fft.ifft2(fft.fft2(Z_d)*fft.fft2(H_d.conjugate()))
        
        zs = self.coh_trans_func1.phase_aberr.get_pol_values()
        print("zs", np.shape(zs))
        grads = np.zeros_like(alphas)
        coef = 4./(self.nx*self.ny)
        for i in np.arange(0, len(alphas)):
            grads[i] = coef*np.sum(zs[i]*(Z_conv_H + Z_conv_H_d).imag)

        return grads


'''
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
'''