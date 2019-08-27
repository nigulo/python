import numpy as np
import numpy.fft as fft
import scipy.signal as signal
import zernike
import utils
import copy

__DEBUG__ = True
if __DEBUG__:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))
    #sys.path.append('../utils')
    import plot

class phase_aberration():
    
    def __init__(self, alphas, start_index = 3):
        self.start_index= start_index
        if len(np.shape(alphas)) == 0:
            # alphas is an integer, representing jmax
            self.create_pols(alphas)
            self.jmax = alphas
        else:
            self.create_pols(len(alphas))
            self.set_alphas(alphas)
    
    def create_pols(self, num):
        self.pols = []
        for i in np.arange(self.start_index+1, self.start_index+num+1):
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
        self.jmax = len(self.alphas)
    
            
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

    def calc_terms(self, xs):
        return
        
    def __call__(self):
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

        if __DEBUG__:
            my_plot = plot.plot(nrows=1, ncols=2)
            my_plot.colormap(self.pupil, [0])
            my_plot.colormap(self.pupil*phase, [1])
            
            my_plot.save("aperture_test.png")
            my_plot.close()

        if defocus:
            phase += self.defocus
        return self.pupil*np.exp(1.j * phase)

def deconvolve_(D, D_d, S, S_d, gamma, do_fft = True, fft_shift = True):
    regularizer_eps = 1e-10
    assert(gamma == 1.0) # Because in likelihood we didn't involve gamma
    #P = np.roll(np.roll(P, int(self.nx/2), axis=0), int(self.nx/2), axis=1)
    #P_d = np.roll(np.roll(P_d, int(self.nx/2), axis=0), int(self.nx/2), axis=1)
    
    S_conj = S.conjugate()
    S_d_conj = S_d.conjugate()
    
    F_image = D * S_conj + gamma * D_d * S_d_conj + regularizer_eps
    np.set_printoptions(threshold=np.inf)
    F_image /= (S*S_conj + gamma * S_d * S_d_conj + regularizer_eps)

    if fft_shift:
        F_image = fft.ifftshift(F_image)
    
    if not do_fft:
        return F_image

    image = fft.ifft2(F_image).real
    #image = np.roll(np.roll(image, int(self.nx/2), axis=0), int(self.nx/2), axis=1)
    return image

class psf():

    '''
        diameter in centimeters
        wavelength in Angstroms
    '''
    def __init__(self, coh_trans_func, nx, arcsec_per_px, diameter, wavelength, corr_or_fft=True):
        self.nx= nx
        coords, rc, x_limit = utils.get_coords(nx, arcsec_per_px, diameter, wavelength)
        self.coords = coords
        x_min = np.min(self.coords, axis=(0,1))
        x_max = np.max(self.coords, axis=(0,1))
        print("psf_coords", x_min, x_max, np.shape(self.coords))
        np.testing.assert_array_almost_equal(x_min, -x_max)
        self.incoh_vals = dict()
        self.otf_vals = dict()
        self.corr = dict() # only for testing purposes
        self.coh_trans_func = coh_trans_func
        self.coh_trans_func.calc(self.coords)
        
        # Repeat the same for bigger grid
        xs1 = np.linspace(-x_limit, x_limit, self.nx*2-1)
        coords1 = np.dstack(np.meshgrid(xs1, xs1)[::-1])
        self.coords1 = coords1

        self.nx1 = self.nx * 2 - 1

        self.coh_trans_func1 = copy.deepcopy(self.coh_trans_func)
        self.coh_trans_func1.calc(coords1)

        xs2 = np.linspace(-x_limit, x_limit, (self.nx*2-1)*2-1)
        coords2 = np.dstack(np.meshgrid(xs2, xs2)[::-1])
        self.coh_trans_func2 = copy.deepcopy(self.coh_trans_func)
        self.coh_trans_func2.calc(coords2)
        
        self.corr_or_fft = corr_or_fft
        
        
    def calc(self, defocus = True, normalize = True):
        coh_vals = self.coh_trans_func(defocus)
        
        if self.corr_or_fft:
            #corr = signal.correlate2d(coh_vals, coh_vals, mode='full')/(self.nx*self.nx)
            corr = signal.fftconvolve(coh_vals, coh_vals[::-1, ::-1].conj(), mode='full')/(self.nx*self.nx)
            vals = fft.fftshift(fft.ifft2(fft.ifftshift(corr))).real
        else:
            vals = fft.ifft2(coh_vals)
            vals = (vals*vals.conjugate()).real
            vals = fft.ifftshift(vals)
            vals = utils.upsample(vals)
            # In principle there shouldn't be negative values, but ...
            vals[vals < 0] = 0. # Set negative values to zero
            corr = fft.fftshift(fft.fft2(fft.ifftshift(vals)))
            

        
        if normalize:
            vals /= vals.sum()
        self.incoh_vals[defocus] = vals
        self.otf_vals[defocus] = corr

        return vals
    
    def calc_otf(self, defocus = True, recalc_psf=True):
        if recalc_psf:
            self.calc(defocus)
            #vals = fft.fft2(fft.ifftshift(self.incoh_vals[defocus]))
            #np.testing.assert_almost_equal(fft.fftshift(vals), self.otf_vals[defocus])
        #vals = fft.fftshift(vals)
        #self.otf_vals[defocus] = vals
        #return vals
        return self.otf_vals[defocus]

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
        dat_F = fft.fftshift(fft.fft2(dat))
        #dat_F = fft.fft2(fft.fftshift(dat))
        ret_val = []
        for m_F in self.multiply(dat_F, defocus):
            m = fft.ifft2(fft.ifftshift(m_F))
            #m = fft.ifftshift(fft.ifft2(m_F))
            ret_val.append(m.real)
        if defocus:
            return (ret_val[0], ret_val[1])
        else:
            return ret_val[0]

    def deconvolve(self, D, D_d, alphas, gamma, do_fft = True):
        self.coh_trans_func.phase_aberr.set_alphas(alphas)
        self.calc(defocus=False)
        self.calc(defocus=True)
        S = self.otf_vals[False]
        S_d = self.otf_vals[True]
        return deconvolve_(D, D_d, S, S_d, gamma, do_fft = do_fft)

    '''
        Actually this is negative log likelihood
    '''
    def likelihood(self, theta, data):
        regularizer_eps = 1e-10
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
        den = S[nzi]*S[nzi].conjugate()+ S_d[nzi]*S_d[nzi].conjugate()+regularizer_eps

        lik = -np.sum((num/den).real) + np.sum((D*D.conjugate() + D_d*D_d.conjugate()).real)
        

        return lik
        

    def likelihood_grad(self, theta, data):
        #print("likelihood_grad")
        regularizer_eps = 1e-10

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
        
        nzi = np.nonzero(np.abs(S)+np.abs(S_d))
        S_nzi = S[nzi]
        S_d_nzi = S_d[nzi]
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
        den = 1./((S_nzi*S_nzi_conj + S_d_nzi*S_d_nzi_conj)**2 + regularizer_eps)
        
        SDS = (S_nzi*S_nzi_conj + S_d_nzi*S_d_nzi_conj)*SD
        
        Z[nzi] = (SDS*D_nzi.conjugate()-SD2*S_nzi_conj)*den
        Z_d[nzi] = (SDS*D_d_nzi.conjugate()-SD2*S_d_nzi_conj)*den
        
        self.coh_trans_func1.phase_aberr.set_alphas(alphas)
        H = self.coh_trans_func1(defocus = False)
        H_d = self.coh_trans_func1(defocus = True)
        zs = self.coh_trans_func1.phase_aberr.get_pol_values()

        self.coh_trans_func2.phase_aberr.set_alphas(alphas)
        H2 = self.coh_trans_func2(defocus = False)
        H2_d = self.coh_trans_func2(defocus = True)
        zs2 = self.coh_trans_func2.phase_aberr.get_pol_values()

        H1 = self.coh_trans_func(defocus = False)
        H1_d = self.coh_trans_func(defocus = True)
        zs1 = self.coh_trans_func.phase_aberr.get_pol_values()


        grads1 = np.zeros_like(alphas)
        for i in np.arange(0, len(alphas)):
            zsH = zs1[i]*H1
            zsH_d = zs1[i]*H1_d
            S_primes = -1.j*(signal.correlate2d(zsH, H1) - signal.correlate2d(H1, zsH))
            S_d_primes = -1.j*(signal.correlate2d(zsH_d, H1_d) - signal.correlate2d(H1_d, zsH_d))
            a = np.sum(Z*S_primes + Z_d*S_d_primes)
            grads1[i] = (a + a.conjugate()).real
        
        '''
        O = np.ones_like(H)+1.j
        O1 = np.ones_like(H1)+1.j
        O2 = np.ones_like(H2)+1.j
        
        I = np.diag(np.ones(O.shape[0]))
        I1 = np.diag(np.ones(O1.shape[0]))
        I2 = np.diag(np.ones(O2.shape[0]))

        my_plot = plot.plot_map(nrows=1, ncols=2)
        my_plot.plot(S.real, [0])
        my_plot.plot(utils.downscale(S).real, [1])
            
        my_plot.save("scaling_test.png")
        my_plot.close()

        grads1_test = np.zeros_like(alphas, dtype='complex')
        for i in np.arange(0, len(alphas)):
            #S_primes = -1.j*signal.correlate2d(zs1[i]*H1, H1)
            S_primes = -1.j*signal.correlate2d(O1, O1)
            a = I*S_primes
            grads1_test[i] = np.sum(a)/a.shape[0]/a.shape[1]

        I_down = utils.downscale(I)
        Z_conv_H = signal.convolve2d(I_down*np.sum(I)/np.sum(I_down), O1.conj(), mode='full')
        Z_conv_H_d = signal.convolve2d(Z_d, H_d.conj(), mode='full')
        
        Z_conv_H_1 = signal.correlate2d(H, Z.conj(), mode='full')
        Z_conv_H_d_1 = signal.correlate2d(H_d, Z_d.conj(), mode='full')


        test1a = -1.j*O*Z_conv_H
        grads_test = np.zeros_like(alphas, dtype='complex')
        for i in np.arange(0, len(alphas)):
            #for k in np.arange(0, S.shape[0]):
            #    for l in np.arange(0, S.shape[1]):
            #        print((H*Z_conv_H + H_d*Z_conv_H_d)[k, l])
            #grads[i] = coef*np.sum(zs[i]*(H*Z_conv_H + H_d*Z_conv_H_d).imag)
            #grads_test[i] = np.sum(zs2[i]*test1a)/test1a.shape[0]/test1a.shape[1]
            grads_test[i] = np.sum(test1a)/test1a.shape[0]/test1a.shape[1]
            print(grads_test[i], grads1_test[i])

        np.testing.assert_almost_equal(grads_test, grads1_test)

        test1 = 1.j*(H2*Z_conv_H + H2_d*Z_conv_H_d)# - (H.conj()*Z_conv_H_conj + H_d.conj()*Z_conv_H_d_conj)
        test1 += test1.conj()
        test2 = 1.j*(H2.conj()*Z_conv_H_1 + H2_d.conj()*Z_conv_H_d_1)
        test2 += test2.conj()
        test = test1-test2
        for k in np.arange(0, S.shape[0]):
            for l in np.arange(0, S.shape[1]):
                print(test[k, l])
        
        
        #test_a = 4.*(H2*Z_conv_H + H2_d*Z_conv_H_d)
        #np.testing.assert_almost_equal(test_a, test)

        print("zs", np.shape(zs))
        grads = np.zeros_like(alphas)
        coef = 1.#-4./(self.nx*self.ny)
        for i in np.arange(0, len(alphas)):
            #for k in np.arange(0, S.shape[0]):
            #    for l in np.arange(0, S.shape[1]):
            #        print((H*Z_conv_H + H_d*Z_conv_H_d)[k, l])
            #grads[i] = coef*np.sum(zs[i]*(H*Z_conv_H + H_d*Z_conv_H_d).imag)
            grads[i] = coef*np.sum(zs[i]*test.real)
            print(grads[i], grads1[i])

        np.testing.assert_almost_equal(grads, grads1)
        '''

        #print("likelihood_grad_end")
        return grads1/(self.nx*self.nx)

    def S_prime(self, theta, data):
        #regularizer_eps = 1e-10

        D = data[0]
        D_d = data[1]
        gamma = data[2] # Not used
        alphas = theta
        
        pa = self.coh_trans_func.phase_aberr
        pa.set_alphas(alphas)
        

        H = self.coh_trans_func(defocus = False)
        H_d = self.coh_trans_func(defocus = True)
        zs = self.coh_trans_func.phase_aberr.get_pol_values()


        grads = np.zeros((len(alphas), H.shape[0]*2-1, H.shape[1]*2-1), dtype='complex')
        for i in np.arange(0, len(alphas)):
            #S_primes = -1.j/(self.nx*self.ny)*(signal.convolve2d(zs[i]*H, H) - signal.convolve(H, zs[i]*H))
            S_primes = 1.j*(signal.correlate2d(zs[i]*H, H) - signal.correlate2d(H, zs[i]*H))
            S_primes1 = 1.j*(signal.correlate2d(zs[i]*H, H) - signal.convolve2d(H, (zs[i]*H).conj()[::-1,::-1]))
            #a = signal.correlate2d(zs[i]*H, H, mode='full')
            #b = signal.convolve2d((zs[i]*H), H.conj()[::-1,::-1], mode='full')
            #np.testing.assert_almost_equal(a, b)

            np.testing.assert_almost_equal(S_primes, S_primes1)
            
            #print("AAAAAAAAA", np.abs(S_primes-S_primes1))
            #S_primes = -1.j/(self.nx*self.ny)*(signal.convolve2d(zs[i]*H, H.conjugate()) - signal.convolve(H, zs[i]*H.conjugate()))
            #S_primes = -1.j/(self.nx*self.ny)*(signal.convolve2d(zs[i]*H, H.conjugate()) - signal.convolve(H, zs[i]*H.conjugate()))
            #S_d_primes = -1.j/(self.nx*self.ny)*(signal.convolve2d(zs1[i]*H1_d, H1_d.conjugate()) - signal.convolve(H1_d, zs1[i]*H1_d.conjugate()))
            grads[i] = S_primes

        return grads/(self.nx*self.nx)


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