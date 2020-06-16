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
            self.jmax = len(alphas)
    
    def create_pols(self, num):
        self.pols = []
        for i in np.arange(self.start_index+1, self.start_index+num+1):
            n, m = zernike.get_nm(i)
            z = zernike.zernike(n, m)
            self.pols.append(z)

    def calc_terms(self, xs=None, nx=None):
        if xs is None:
            assert(nx is not None)
            xs = np.linspace(-1./np.sqrt(2.), 1./np.sqrt(2.), nx)
            #print("PSF x_limit", xs[0], xs[-1])
            xs = np.dstack(np.meshgrid(xs, xs)[::-1])
            
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
    
    # TODO: why this method?
    def get_pol_values(self):
        vals = np.zeros(np.concatenate((np.array([len(self.terms)]), np.shape(self.terms[0]))))
        for i in np.arange(0, len(self.terms)):
            vals[i] = self.terms[i]
        return vals
    
    def get_terms(self):
        return self.terms

    def set_terms(self, terms):
        self.terms = terms


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

    def __init__(self, pupil_func = None, phase_aberr = None, defocus_func = None):
        self.pupil_func = pupil_func
        self.phase_aberr = phase_aberr
        self.defocus_func = defocus_func
        
    def set_phase_aberr(self, phase_aberr):
        self.phase_aberr = phase_aberr
        
        
    def calc(self, xs):
        if self.phase_aberr is not None:
            self.phase_aberr.calc_terms(xs)
        if self.pupil_func is not None:
            self.pupil = self.pupil_func(xs)
        if self.defocus_func is not None:
            self.defocus = self.defocus_func(xs)
            if np.isscalar(self.defocus):
                self.defocus = np.tile(self.defocus, (xs.shape[0], xs.shape[1]))
            self.diversity = np.zeros((2, self.defocus.shape[0], self.defocus.shape[1]))
            self.diversity[1] = self.defocus;
        else:
            self.defocus = 0.
            self.diversity = np.zeros(2)
    
    def get_pupil(self):
        return self.pupil

    def set_pupil(self, pupil):
        self.pupil = pupil
        
    def get_defocus(self):
        return self.defocus

    def set_defocus(self, defocus):
        self.defocus = defocus

    def get_diversity(self):
        if hasattr(self, 'diversity'):
            return self.diversity
        else:
            return None

    def set_diversity(self, diversity):
        self.diversity = diversity
        
    def __call__(self):
        self.phase = self.phase_aberr()

        if __DEBUG__:
            my_plot = plot.plot(nrows=1, ncols=2)
            my_plot.colormap(self.pupil, [0])
            my_plot.colormap(self.pupil*self.phase, [1])
            
            my_plot.save("aperture_test.png")
            my_plot.close()

        return np.array([self.pupil*np.exp(1.j * self.phase + self.diversity[0]), self.pupil*np.exp(1.j * (self.phase + self.diversity[1]))])

    '''
        Returns the coefficients for given expansion
    '''
    def dot(self, pa, normalize=True):
        if not hasattr(self, 'phase'):
            self.phase = self.phase_aberr()
        ret_val = np.sum(self.phase*pa.terms, axis=(1, 2))
        #ret_val1 = np.sum(np.tile(np.reshape(self.phase, (1, self.phase.shape[0], self.phase.shape[1])), (len(pa.terms), 1, 1))*pa.terms, axis=(1, 2))
        #np.testing.assert_array_almost_equal(ret_val, ret_val1)
        if normalize:
            ret_val /= np.sum(pa.terms*pa.terms, axis=(1, 2))
        # Check orthogonality
        #for i in np.arange(len(pa.terms)):
        #    for j in np.arange(i + 1, len(pa.terms)):
        #        print(np.sum(pa.terms[i]*pa.terms[j]))
        return ret_val

class psf():

    '''
        diameter in centimeters
        wavelength in Angstroms
    '''
    def __init__(self, coh_trans_func, nx=None, arcsec_per_px=None, diameter=None, wavelength=None, corr_or_fft=True, tip_tilt=None):
        self.coh_trans_func = coh_trans_func
        if nx is None:
            # Everything is precalculated
            self.nx = coh_trans_func.get_pupil().shape[0]
        else:
            self.nx= nx
            coords, rc, x_limit = utils.get_coords(nx, arcsec_per_px, diameter, wavelength)
            self.coords = coords
            x_min = np.min(self.coords, axis=(0,1))
            x_max = np.max(self.coords, axis=(0,1))
            #print("psf_coords", x_min, x_max, np.shape(self.coords))
            np.testing.assert_array_almost_equal(x_min, -x_max)
            self.coh_trans_func.calc(self.coords)
        
            # Repeat the same for bigger grid
            xs1 = np.linspace(-x_limit, x_limit, self.nx*2-1)
            coords1 = np.dstack(np.meshgrid(xs1, xs1)[::-1])
            self.coords1 = coords1
        
            self.coh_trans_func1 = copy.deepcopy(self.coh_trans_func)
            self.coh_trans_func1.calc(coords1)
    
            xs2 = np.linspace(-x_limit, x_limit, (self.nx*2-1)*2-1)
            coords2 = np.dstack(np.meshgrid(xs2, xs2)[::-1])
            self.coh_trans_func2 = copy.deepcopy(self.coh_trans_func)
            self.coh_trans_func2.calc(coords2)
        
        self.nx1 = self.nx * 2 - 1

        self.incoh_vals = dict()
        self.otf_vals = dict()
        self.corr = dict() # only for testing purposes
        self.corr_or_fft = corr_or_fft
        self.tip_tilt = tip_tilt
        
        
    def calc(self, alphas=None, normalize = True):
        if alphas is None:
            l = 1
        else:
            l = alphas.shape[0]
            
        if self.corr_or_fft:
            self.incoh_vals = np.zeros((l, 2, self.nx1, self.nx1))
            self.otf_vals = np.zeros((l, 2, self.nx1, self.nx1), dtype='complex')
        else:
            self.incoh_vals = np.zeros((l, 2, self.nx, self.nx))
            self.otf_vals = np.zeros((l, 2, self.nx, self.nx), dtype='complex')
        
        for i in np.arange(0, l):
            if alphas is not None:
                self.coh_trans_func.phase_aberr.set_alphas(alphas[i])
            coh_vals = self.coh_trans_func()
        
            if self.corr_or_fft:
                #corr = signal.correlate2d(coh_vals, coh_vals, mode='full')/(self.nx*self.nx)
                corr = signal.fftconvolve(coh_vals, coh_vals[:, ::-1, ::-1].conj(), mode='full', axes=(-2, -1))/(self.nx*self.nx)
                vals = fft.fftshift(fft.ifft2(fft.ifftshift(corr, axes=(-2, -1))), axes=(-2, -1)).real
            else:
                vals = fft.ifft2(coh_vals, axes=(-2, -1))
                vals = (vals*vals.conjugate()).real
                vals = fft.ifftshift(vals, axes=(-2, -1))
                if False: 
                    vals = np.array([utils.upsample(vals[0]), utils.upsample(vals[1])])
                # In principle there shouldn't be negative values, but ...
                vals[vals < 0] = 0. # Set negative values to zero
                corr = fft.fftshift(fft.fft2(fft.ifftshift(vals, axes=(-2, -1))), axes=(-2, -1))
    
            if normalize:
                corr /= np.sum(self.coh_trans_func.pupil)
                norm = np.sum(vals, axis = (1, 2)).repeat(vals.shape[1]*vals.shape[2]).reshape((vals.shape[0], vals.shape[1], vals.shape[2]))
                vals /= norm
            self.incoh_vals[i] = vals
            self.otf_vals[i] = corr
        return self.incoh_vals

    '''
    dat_F.shape = [l, 2, nx, nx]
    alphas.shape = [l, jmax]
    '''
    def multiply(self, dat_F, alphas=None, a=None):
        if len(dat_F.shape) < 3:
            dat_F = np.array([[dat_F, dat_F]])
        elif len(dat_F.shape) < 4:
            dat_F = np.tile(dat_F, (2,1)).reshape((dat_F.shape[0],2,dat_F.shape[-2],dat_F.shape[-1]))
        if alphas is not None and len(alphas.shape) < 2:
            alphas = np.array([alphas])

        if a is not None:
            assert(self.tip_tilt is not None)

        if len(self.otf_vals) == 0:
            self.calc(alphas=alphas)
        if a is not None:
            return self.tip_tilt.multiply(dat_F * self.otf_vals, a)
        else:
            return dat_F * self.otf_vals
            
    '''
    dat.shape = [l, 2, nx, nx]
    alphas.shape = [l, jmax]
    '''
    def convolve(self, dat, alphas=None, a=None):
        if len(dat.shape) < 3:
            dat = np.array([[dat, dat]])
        elif len(dat.shape) < 4:
            dat = np.tile(dat, (2,1)).reshape((dat.shape[0],2,dat.shape[-2],dat.shape[-1]))
        if alphas is not None and len(alphas.shape) < 2:
            alphas = np.array([alphas])
            
        dat_F = fft.fftshift(fft.fft2(dat), axes=(-2, -1))
        dat_F = self.multiply(dat_F, alphas, a)
        
        return fft.ifft2(fft.ifftshift(dat_F, axes=(-2, -1))).real
        

    def deconvolve(self, Ds, alphas, gamma, do_fft = True, fft_shift_before = False, ret_all=False, a_est=None, normalize=False, fltr=None):
        self.calc(alphas=alphas)
        Ps = self.otf_vals
        if not fft_shift_before:
            Ps = fft.ifftshift(Ps, axes=(-2, -1))
        if normalize:
            Ds = utils.normalize_(Ds, Ps)
        return utils.deconvolve_(Ds, Ps, gamma, do_fft = do_fft, fft_shift_before = fft_shift_before, ret_all=ret_all, tip_tilt=self.tip_tilt, a_est=a_est, fltr=fltr)


    def encode_params(self, alphas, a = None):
        theta = alphas.flatten()
        if self.tip_tilt is not None:
            theta = np.concatenate((theta, self.tip_tilt.encode(a)))
        return theta

    def encode_data(self, Ds, gamma):
        return [Ds, gamma]
    
    def encode(self, alphas, Ds, gamma, a = []):
        return self.encode_params(alphas, a), self.encode_data(Ds, gamma)

    def decode(self, theta, data):
        Ds = data[0]
        gamma = data[1]
        L = Ds.shape[0]
        jmax = self.coh_trans_func.phase_aberr.jmax
        alphas = np.zeros((L, jmax))
        #print("theta.shape", theta.shape, L, self.jmax, theta)
        for l in np.arange(0, L):
            begin_index = l*jmax
            alphas[l] = theta[begin_index:begin_index+jmax]
        return alphas, Ds, gamma, theta[L*jmax:]

    '''
        Actually this is negative log likelihood
    '''
    def likelihood(self, theta, data):
        alphas, Ds, gamma, other = self.decode(theta, data)
        regularizer_eps = 0.#1e-10
        D = Ds[:,0,:,:]
        D_d = Ds[:,1,:,:]
        
        self.calc(alphas=alphas)
        
        S = self.otf_vals[:,0,:,:]
        S_d = self.otf_vals[:,1,:,:]
        nzi = np.nonzero(np.abs(S) + np.abs(S_d))
        
        num = D[nzi]*S[nzi].conjugate() + D_d[nzi]*S_d[nzi].conjugate()
        num *= num.conjugate()
        den = S[nzi]*S[nzi].conjugate()+ S_d[nzi]*S_d[nzi].conjugate()+regularizer_eps

        lik = -np.sum((num/den).real) + np.sum((D*D.conjugate() + D_d*D_d.conjugate()).real)
        
        #######################################################################
        # Tip-tilt estimation
        #######################################################################
        if self.tip_tilt is not None:
            Ps = np.ones((D.shape[0], 2, self.nx1, self.nx1), dtype='complex')
            self.tip_tilt.set_data(fft.ifftshift(Ds, axes=(-2, -1)), Ps)#, F)
            lik += self.tip_tilt.lik(other)

        return lik
        

    def likelihood_grad(self, theta, data):
        alphas, Ds, gamma, other = self.decode(theta, data)
        #print("likelihood_grad")
        regularizer_eps = 0.#1e-10

        L = Ds.shape[0]
        jmax = self.coh_trans_func.phase_aberr.jmax

        D = Ds[:,0,:,:]
        D_d = Ds[:,1,:,:]
        
        self.calc(alphas=alphas)
        
        S = self.otf_vals[:,0,:,:]*np.sum(self.coh_trans_func.pupil)
        S_d = self.otf_vals[:,1,:,:]*np.sum(self.coh_trans_func.pupil)
        
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
        
        jmax = alphas.shape[1]
        grads = np.zeros(L*jmax)#, np.shape(D)[0], np.shape(D)[1]), dtype='complex')
        
        for l in np.arange(0, L):
            self.coh_trans_func.phase_aberr.set_alphas(alphas[l])
        
            ctf1 = self.coh_trans_func()
            H1 = ctf1[0]
            H1_d = ctf1[1]
            zs1 = self.coh_trans_func.phase_aberr.get_pol_values()
    
            for i in np.arange(0, jmax):
                zsH = zs1[i]*H1
                zsH_d = zs1[i]*H1_d
                S_primes = -1.j*(signal.correlate2d(zsH, H1) - signal.correlate2d(H1, zsH))
                S_d_primes = -1.j*(signal.correlate2d(zsH_d, H1_d) - signal.correlate2d(H1_d, zsH_d))
                a = np.sum(Z[l]*S_primes + Z_d[l]*S_d_primes)
                grads[l*jmax + i] = (a + a.conjugate()).real # This is just 2*a.real
        grads /= (self.nx*self.nx)
        
        
        #######################################################################
        # Tip-tilt estimation
        #######################################################################
        if self.tip_tilt is not None:
            Ps = np.ones((L, 2, self.nx1, self.nx1), dtype='complex')
            self.tip_tilt.set_data(fft.ifftshift(Ds, axes=(-2, -1)), Ps)#, F)
            grads = np.concatenate((grads, self.tip_tilt.lik_grad(other)))

        return grads

    def S_prime(self, theta, data):
        #regularizer_eps = 1e-10

        Ds = data[0]
        L = Ds.shape[0]
        D = Ds[:,0,:,:]
        D_d = Ds[:,1,:,:]

        gamma = data[1] # Not used
        alphas = theta.reshape(L, -1)
        
        grads = np.zeros((L, alphas.shape[1], Ds.shape[-2], Ds.shape[-1]), dtype='complex')
        for l in np.arange(0, L):
            self.coh_trans_func.phase_aberr.set_alphas(alphas[l])
        
            ctf = self.coh_trans_func()
            H = ctf[0]
            H_d = ctf[1]
            zs = self.coh_trans_func.phase_aberr.get_pol_values()
            
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
                grads[l, i] = S_primes

        return grads/(self.nx*self.nx)


    def critical_sampling(self, image, threshold=1e-3):
    
        pa_copy = self.coh_trans_func.phase_aberr
        pa = phase_aberration([])
        pa.calc_terms(nx = self.nx)
        self.coh_trans_func.phase_aberr = pa
        diversity_copy = self.coh_trans_func.get_diversity()
        self.coh_trans_func.set_diversity(np.zeros((2, self.nx, self.nx)))
        self.calc()
        self.coh_trans_func.phase_aberr = pa_copy
        self.coh_trans_func.set_diversity(diversity_copy)
        
        fimage = fft.fft2(fft.fftshift(image))
        #_, coefs = psf_.multiply(np.array([[fimage, fimage]]), np.array([], dtype='complex'))
        #coefs = coefs[0, 0, :, :]
        #coefs = np.abs(coefs)
        otf_vals = fft.ifftshift(self.otf_vals, axes=(-2, -1))
        coefs = np.abs(otf_vals[0, 0, :, :])
        
        if __DEBUG__:
            my_plot = plot.plot()
            my_plot.hist(coefs, bins=100)
            my_plot.save("transfer_func_hist.png")
            my_plot.close()
        
        mask = np.ones_like(coefs)
        indices = np.where(coefs < threshold)
        mask[indices] = 0.
        
        if __DEBUG__:
            my_plot = plot.plot(nrows=2)
            my_plot.colormap(fft.fftshift(coefs), [0])
            my_plot.colormap(fft.fftshift(mask), [1])
            my_plot.save("transfer_func.png")
            my_plot.close()
        
        fimage *= mask
        return fft.ifftshift(fft.ifft2(fimage), axes=(-2, -1)).real


def critical_sampling(image, arcsec_per_px, diameter, wavelength, threshold=1e-3):
    nx = image.shape[0]
    coords, _, _ = utils.get_coords(nx, arcsec_per_px, diameter, wavelength)
    aperture_func = lambda xs: utils.aperture_circ(xs, coef=15, radius =1.)
    defocus_func = lambda xs: 0.
    pa = phase_aberration([])
    ctf = coh_trans_func(aperture_func, pa, defocus_func)
    psf_ = psf(ctf, (nx+1)//2, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)

    return psf_.critical_sampling(image, threshold)
