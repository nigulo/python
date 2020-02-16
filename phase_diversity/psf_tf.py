import matplotlib as mpl
mpl.use('Agg')
import numpy as np

import tensorflow as tf

import utils
import zernike


class phase_aberration_tf():
    
    def __init__(self, alphas, start_index = 3):
        self.start_index= start_index
        if len(np.shape(alphas)) == 0:
            # alphas is an integer, representing jmax
            self.create_pols(alphas)
            self.jmax = alphas
        else:
            self.create_pols(len(alphas))
            self.set_alphas(tf.constant(alphas))
            self.jmax = len(alphas)
    
    def create_pols(self, num):
        self.pols = []
        for i in np.arange(self.start_index+1, self.start_index+num+1):
            n, m = zernike.get_nm(i)
            z = zernike.zernike(n, m)
            self.pols.append(z)

    def calc_terms(self, xs):
        terms = np.zeros(np.concatenate(([len(self.pols)], np.shape(xs)[:-1])))
        i = 0
        rhos_phis = utils.cart_to_polar(xs)
        for z in self.pols:
            terms[i] = z.get_value(rhos_phis)
            i += 1
        self.terms = tf.constant(terms, dtype='float32')

    def set_alphas(self, alphas):
        if len(self.pols) != self.jmax:
            self.create_pols(self.jmax)
        self.alphas = alphas
        #self.jmax = tf.shape(self.alphas).eval()[0]
    
            
    def __call__(self):
        #vals = np.zeros(tf.shape(self.terms)[1:])
        #for i in np.arange(0, len(self.terms)):
        #    vals += self.terms[i] * self.alphas[i]
        nx = self.terms.shape[1]
        alphas = tf.tile(tf.reshape(self.alphas, [self.jmax, 1, 1]), multiples=[1, nx, nx])
        #alphas1 = tf.complex(alphas1, tf.zeros((self.jmax, nx, nx)))
        vals = tf.math.reduce_sum(tf.math.multiply(self.terms, alphas), 0)
        #vals = tf.math.reduce_sum(tf.math.multiply(self.terms, tf.reshape(self.alphas, [self.jmax, 1, 1])), 0)
        return vals
    

'''
Coherent transfer function, also called as generalized pupil function
'''
class coh_trans_func_tf():

    def __init__(self, pupil_func, phase_aberr, defocus_func = None):
        self.pupil_func = pupil_func
        self.phase_aberr = phase_aberr
        self.defocus_func = defocus_func
        
    def calc(self, xs):
        self.phase_aberr.calc_terms(xs)
        pupil = self.pupil_func(xs)
        if self.defocus_func is not None:
            defocus = self.defocus_func(xs)
        else:
            assert(False)
        self.defocus = tf.complex(tf.constant(defocus, dtype='float32'), tf.zeros((defocus.shape[0], defocus.shape[1]), dtype='float32'))
        
        self.i = tf.constant(1.j, dtype='complex64')
        self.pupil = tf.constant(pupil, dtype='float32')
        self.pupil = tf.complex(self.pupil, tf.zeros((pupil.shape[0], pupil.shape[1]), dtype='float32'))
        
    def __call__(self):
        self.phase = self.phase_aberr()
        self.phase = tf.complex(self.phase, tf.zeros((self.phase.shape[0], self.phase.shape[1]), dtype='float32'))

        focus_val = tf.math.multiply(self.pupil, tf.math.exp(tf.math.scalar_mul(self.i, self.phase)))
        defocus_val = tf.math.multiply(self.pupil, tf.math.exp(tf.math.scalar_mul(self.i, (tf.math.add(self.phase, self.defocus)))))

        #return tf.concat(tf.reshape(focus_val, [1, focus_val.shape[0], focus_val.shape[1]]), tf.reshape(defocus_val, [1, defocus_val.shape[0], defocus_val.shape[1]]), 0)
        return tf.stack([focus_val, defocus_val])

    def get_defocus_val(self, focus_val):
        return tf.math.multiply(focus_val, tf.math.exp(tf.math.scalar_mul(self.i, self.defocus)))

class psf_tf():

    '''
        diameter in centimeters
        wavelength in Angstroms
    '''
    def __init__(self, coh_trans_func, nx, arcsec_per_px, diameter, wavelength):
        self.nx= nx
        coords, rc, x_limit = utils.get_coords(nx, arcsec_per_px, diameter, wavelength)
        self.coords = coords
        x_min = np.min(self.coords, axis=(0,1))
        x_max = np.max(self.coords, axis=(0,1))
        print("psf_coords", x_min, x_max, np.shape(self.coords))
        np.testing.assert_array_almost_equal(x_min, -x_max)
        self.incoh_vals = None
        self.otf_vals = None
        self.corr = None # only for testing purposes
        self.coh_trans_func = coh_trans_func
        self.coh_trans_func.calc(self.coords)
        

        
        
    def calc(self, alphas=None):
        #self.incoh_vals = tf.zeros((2, self.nx1, self.nx1))
        #self.otf_vals = tf.zeros((2, self.nx1, self.nx1), dtype='complex')
        
        if alphas is not None:
            self.coh_trans_func.phase_aberr.set_alphas(alphas)
        coh_vals = self.coh_trans_func()
    
        vals = tf.signal.ifft2d(coh_vals)
        vals = tf.math.real(tf.multiply(vals, tf.math.conj(vals)))
        vals = tf.signal.ifftshift(vals, axes=(1, 2))
        
        vals = tf.transpose(vals, (1, 2, 0))
        #vals = np.array([utils.upsample(vals[0]), utils.upsample(vals[1])])
        # Maybe have to add channels axis first
        vals = tf.image.resize(vals, size=(tf.shape(vals)[0]*2, tf.shape(vals)[1]*2))
        vals = tf.transpose(vals, (2, 0, 1))
        # In principle there shouldn't be negative values, but ...
        #vals[vals < 0] = 0. # Set negative values to zero
        vals = tf.cast(vals, dtype='complex64')
        corr = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(vals, axes=(1, 2))), axes=(1, 2))

        #if normalize:
        #    norm = np.sum(vals, axis = (1, 2)).repeat(vals.shape[1]*vals.shape[2]).reshape((vals.shape[0], vals.shape[1], vals.shape[2]))
        #    vals /= norm
        self.incoh_vals = vals
        self.otf_vals = corr
        return self.incoh_vals


    '''
    dat_F.shape = [l, 2, nx, nx]
    alphas.shape = [l, jmax]
    '''
    def multiply(self, dat_F, alphas):

        if self.otf_vals is None:
            self.calc(alphas=alphas)
        return tf.math.multiply(dat_F, self.otf_vals)


    def aberrate(self, x):
        nx = self.nx
        jmax = self.coh_trans_func.phase_aberr.jmax
        x = tf.reshape(x, [jmax + nx*nx])
        alphas = tf.slice(x, [0], [jmax])
        obj = tf.reshape(tf.slice(x, [jmax], [nx*nx]), [nx, nx])
        
        fobj = tf.signal.fft2d(tf.complex(obj, tf.zeros((nx, nx))))
        fobj = tf.signal.fftshift(fobj)
    
    
        DF = self.psf.multiply(fobj, alphas)
        DF = tf.signal.ifftshift(DF, axes = (1, 2))
        D = tf.math.real(tf.signal.ifft2d(DF))
        #D = tf.signal.fftshift(D, axes = (1, 2)) # Is it needed?
        D = tf.transpose(D, (1, 2, 0))
        D = tf.reshape(D, [1, D.shape[0], D.shape[1], D.shape[2]])
                    
        return D
