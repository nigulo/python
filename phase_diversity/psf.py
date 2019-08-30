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
        if self.defocus_func is not None:
            self.defocus = self.defocus_func(xs)
        else:
            self.defocus = 0.
        
    def __call__(self):
        phase = self.phase_aberr()

        if __DEBUG__:
            my_plot = plot.plot(nrows=1, ncols=2)
            my_plot.colormap(self.pupil, [0])
            my_plot.colormap(self.pupil*phase, [1])
            
            my_plot.save("aperture_test.png")
            my_plot.close()

        return np.array([self.pupil*np.exp(1.j * phase), self.pupil*np.exp(1.j * (phase + self.defocus))])

def deconvolve_(Ds, Ss, gamma, do_fft = True, fft_shift = True, tip_tilt = None, a_est=None):
    assert(gamma == 1.0) # Because in likelihood we didn't involve gamma
    D = Ds[:,0]
    D_d = Ds[:,1]
    

    S = Ss[:,0]
    S_d = Ss[:,1]

    regularizer_eps = 1e-10
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

    if tip_tilt is not None and a_est is not None:
        image, image_F, Ps = tip_tilt.deconvolve(F_image, Ps, a_est)
    else:
        image = fft.ifft2(F_image).real
    #image = np.roll(np.roll(image, int(self.nx/2), axis=0), int(self.nx/2), axis=1)
    return image

class psf():

    '''
        diameter in centimeters
        wavelength in Angstroms
    '''
    def __init__(self, coh_trans_func, nx, arcsec_per_px, diameter, wavelength, corr_or_fft=True, tip_tilt=None):
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
        self.tip_tilt = tip_tilt
        
        
    def calc(self, alphas=None, normalize = True):
        if alphas is None:
            l = 1
        else:
            l = alphas.shape[0]
        self.incoh_vals = np.zeros((l, 2, self.nx1, self.nx1))
        self.otf_vals = np.zeros((l, 2, self.nx1, self.nx1), dtype='complex')
        
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
                vals = np.array([utils.upsample(vals[0]), utils.upsample(vals[1])])
                # In principle there shouldn't be negative values, but ...
                vals[vals < 0] = 0. # Set negative values to zero
                corr = fft.fftshift(fft.fft2(fft.ifftshift(vals, axes=(-2, -1))), axes=(-2, -1))
    
            if normalize:
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
            self.calc(alphas)
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
        dat_F= self.multiply(dat_F, alphas, a)
        
        return fft.ifft2(fft.ifftshift(dat_F, axes=(-2, -1))).real
        

    def deconvolve(self, Ds, alphas, gamma, do_fft = True, a_est=None):
        self.calc(alphas)
        return deconvolve_(Ds, self.otf_vals, gamma, do_fft = do_fft, tip_tilt=self.tip_tilt, a_est=a_est)

    '''
        Actually this is negative log likelihood
    '''
    def likelihood(self, theta, data):
        regularizer_eps = 0.#1e-10
        Ds = data[0]
        L = Ds.shape[0]
        D = Ds[:,0,:,:]
        D_d = Ds[:,1,:,:]
        
        gamma = data[1] # Not used
        jmax = self.coh_trans_func.phase_aberr.jmax
        alphas = theta[:L*jmax].reshape((L, -1))
        other = theta[L*jmax:]
        
        self.calc(alphas)
        
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
            Ps = np.ones((D.shape[0], 2, self.nx, self.nx), dtype='complex')
            self.tip_tilt.set_data(Ds, Ps)#, F)
            lik += self.tip_tilt.lik(other)

        return lik
        

    def likelihood_grad(self, theta, data):
        #print("likelihood_grad")
        regularizer_eps = 0.#1e-10

        Ds = data[0]
        L = Ds.shape[0]
        D = Ds[:,0,:,:]
        D_d = Ds[:,1,:,:]

        gamma = data[1] # Not used
        jmax = self.coh_trans_func.phase_aberr.jmax
        alphas = theta[:L*jmax].reshape((L, -1))
        other = theta[L*jmax:]
        
        self.calc(alphas)
        
        S = self.otf_vals[:,0,:,:]
        S_d = self.otf_vals[:,1,:,:]
        
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
                grads[l*jmax + i] = (a + a.conjugate()).real
        grads /= (self.nx*self.nx)
        
        
        #######################################################################
        # Tip-tilt estimation
        #######################################################################
        if self.tip_tilt is not None:
            Ps = np.ones((L, 2, self.nx, self.nx), dtype='complex')
            self.tip_tilt.set_data(Ds, Ps)#, F)
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


