import matplotlib as mpl
mpl.use('Agg')
import numpy as np

import tensorflow as tf

import utils
import zernike
import sys


def _centered(arr, newshape):
    # Return the center newshape portion of the array.
    currshape = tf.shape(arr)[-2:]
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    return arr[..., startind[0]:endind[0], startind[1]:endind[1]]

def fftconv(in1, in2, mode="full", reorder_channels_before=True, reorder_channels_after=True):
    # Reorder channels to come second (needed for fft)
    if reorder_channels_before:
        if len(tf.shape(in1)) == 4:
            perm = [0, 3, 1, 2]
        else:
            perm = [2, 0, 1]

        in1 = tf.transpose(in1, perm=perm)
        in2 = tf.transpose(in2, perm=perm)

    # Extract shapes
    s1 = tf.convert_to_tensor(tf.shape(in1)[-2:])
    s2 = tf.convert_to_tensor(tf.shape(in2)[-2:])
    shape = s1 + s2 - 1

    # Compute convolution in fourier space
    sp1 = tf.signal.fft2d(in1, shape)
    sp2 = tf.signal.fft2d(in2, shape)
    ret = tf.signal.ifft2d(sp1 * sp2, shape)

    # Crop according to mode
    if mode == "full":
        cropped = ret
    elif mode == "same":
        cropped = _centered(ret, s1)
    elif mode == "valid":
        cropped = _centered(ret, s1 - s2 + 1)
    else:
        raise ValueError("Acceptable mode flags are 'valid',"
                         " 'same', or 'full'.")

    result = cropped
    # Reorder channels to last
    if reorder_channels_after:
        if len(tf.shape(in1)) == 4:
            perm = [0, 2, 3, 1]
        else:
            perm = [1, 2, 0]

        result = tf.transpose(result, perm=perm)
    return result

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
    def __init__(self, coh_trans_func, nx, arcsec_per_px, diameter, wavelength, corr_or_fft=False, num_frames=1):
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
        self.corr_or_fft = corr_or_fft
        self.num_frames = num_frames
        


    '''
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
    '''

    def calc(self, alphas=None, normalize=True):
        #self.incoh_vals = tf.Variable(tf.zeros([self.num_frames*2, self.nx, self.nx], dtype="complex64"))
        #self.otf_vals = tf.Variable(tf.zeros([self.num_frames*2, self.nx, self.nx], dtype="complex64"))
        
        
        def fn(alphas):
            if alphas is not None:
                self.coh_trans_func.phase_aberr.set_alphas(alphas)
            coh_vals = self.coh_trans_func()
        
            if self.corr_or_fft:
                corr = fftconv(coh_vals, tf.math.conj(coh_vals[:, ::-1, ::-1]), mode='full', reorder_channels_before=False, reorder_channels_after=False)/(self.nx*self.nx)
                vals = tf.math.real(tf.signal.fftshift(tf.signal.ifft2d(tf.signal.ifftshift(corr, axes=(1, 2))), axes=(1, 2)))
                #vals = tf.transpose(vals, (2, 0, 1))
            else:
                
                vals = tf.signal.ifft2d(coh_vals)
                vals = tf.math.real(tf.multiply(vals, tf.math.conj(vals)))
                vals = tf.signal.ifftshift(vals, axes=(1, 2))
                
                ###################################################################
                #vals = tf.transpose(vals, (1, 2, 0))
                ##vals = np.array([utils.upsample(vals[0]), utils.upsample(vals[1])])
                ## Maybe have to add channels axis first
                #vals = tf.image.resize(vals, size=(tf.shape(vals)[0]*2, tf.shape(vals)[1]*2))
                #vals = tf.transpose(vals, (2, 0, 1))
                ###################################################################
                # In principle there shouldn't be negative values, but ...
                #vals[vals < 0] = 0. # Set negative values to zero
                vals = tf.cast(vals, dtype='complex64')
                corr = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(vals, axes=(1, 2))), axes=(1, 2))

            if normalize:
                norm = tf.tile(tf.math.reduce_sum(vals, axis = (1, 2), keepdims=True), [1, tf.shape(vals)[1], tf.shape(vals)[1]])
                vals = tf.divide(vals, norm)
            #self.incoh_vals[2*i].assign(vals[0, :, :])
            #self.incoh_vals[2*i+1].assign(vals[1, :, :])
            #self.otf_vals[2*i].assign(corr[0, :, :])
            #self.otf_vals[2*i+1].assign(corr[1, :, :])
            
            return corr
            
        otf_vals = tf.map_fn(fn, alphas, dtype='complex64')
        self.otf_vals = tf.reshape(otf_vals, [self.num_frames*2, self.nx, self.nx])
            
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
        x = tf.reshape(x, [jmax*self.num_frames + nx*nx])
        alphas = tf.reshape(tf.slice(x, [0], [jmax*self.num_frames]), [self.num_frames, jmax])
        obj = tf.reshape(tf.slice(x, [jmax*self.num_frames], [nx*nx]), [1, nx, nx])
        obj = tf.tile(obj, [self.num_frames*2, 1, 1])
        
        fobj = tf.signal.fft2d(tf.complex(obj, tf.zeros((self.num_frames*2, nx, nx))))
        fobj = tf.signal.fftshift(fobj, axes = (1, 2))
    
        DF = self.multiply(fobj, alphas)
        DF = tf.signal.ifftshift(DF, axes = (1, 2))
        D = tf.math.real(tf.signal.ifft2d(DF))
        #D = tf.signal.fftshift(D, axes = (1, 2)) # Is it needed?
        D = tf.transpose(D, (1, 2, 0))
        D = tf.reshape(D, [1, D.shape[0], D.shape[1], D.shape[2]])

        #D = tf.transpose(tf.reshape(obj, [1, self.num_frames*2, self.nx, self.nx]), [0, 2, 3, 1])
        return D


    '''
        self.calc(alphas=alphas)
        Ps = self.otf_vals
        if not fft_shift_before:
            Ps = fft.ifftshift(Ps, axes=(-2, -1))
        if normalize:
            Ds = utils.normalize_(Ds, Ps)
    
        D = Ds[:, 0, :, :]
        D_d = Ds[:, 1, :, :]
        
        P = Ps[:, 0, :, :]
        P_d = Ps[:, 1, :, :]
    
        P_conj = P.conjugate()
        P_d_conj = P_d.conjugate()
    
        F_image = np.sum(D * P_conj + gamma * D_d * P_d_conj + regularizer_eps, axis=0)
        den = np.sum(P*P_conj + gamma * P_d * P_d_conj + regularizer_eps, axis=0)
        F_image /= den
    
        if fft_shift_before:
            F_image = fft.ifftshift(F_image, axes=(-2, -1))
    
        image = fft.ifft2(F_image).real
        if not fft_shift_before:
            image = fft.ifftshift(image, axes=(-2, -1))
    '''

    def deconvolve(self, x, do_fft = True):
        nx = self.nx
        jmax = self.coh_trans_func.phase_aberr.jmax
        x = tf.reshape(x, [jmax*self.num_frames + nx*nx*self.num_frames*2])
        alphas = tf.reshape(tf.slice(x, [0], [jmax*self.num_frames]), [self.num_frames, jmax])
        #Ds = tf.reshape(tf.slice(x, [jmax*self.num_frames], [nx*nx*self.num_frames*2]), [self.num_frames*2, nx, nx])
        Ds = tf.transpose(tf.reshape(tf.slice(x, [jmax*self.num_frames], [nx*nx*self.num_frames*2]), [nx, nx, self.num_frames*2]), [2, 0, 1])

        Ds = tf.complex(Ds, tf.zeros((self.num_frames*2, nx, nx)))
        Ds_F = tf.signal.fft2d(tf.signal.ifftshift(Ds, axes = (1, 2)))

        self.calc(alphas=alphas)
        Ps = self.otf_vals
        
        if do_fft:
            Ps = tf.signal.ifftshift(Ps, axes=(1, 2))

        Ps_conj = tf.math.conj(Ps)
    
        num = tf.math.reduce_sum(tf.multiply(Ds_F, Ps_conj), axis=[0])
        den = tf.math.reduce_sum(tf.multiply(Ps, Ps_conj), axis=[0])
        F_image = tf.divide(num, den)
    
        if do_fft:
            image = tf.math.real(tf.signal.ifft2d(F_image))
            image = tf.signal.ifftshift(image)
        else:
            image = F_image
        
        return image, Ps
        
    def mfbd_loss(self, x):
        nx = self.nx
        jmax = self.coh_trans_func.phase_aberr.jmax
        x = tf.reshape(x, [jmax*self.num_frames + nx*nx*self.num_frames*2])
        alphas = tf.reshape(tf.slice(x, [0], [jmax*self.num_frames]), [self.num_frames, jmax])
        #Ds = tf.reshape(tf.slice(x, [jmax*self.num_frames], [nx*nx*self.num_frames*2]), [self.num_frames*2, nx, nx])
        Ds = tf.transpose(tf.reshape(tf.slice(x, [jmax*self.num_frames], [nx*nx*self.num_frames*2]), [nx, nx, self.num_frames*2]), [2, 0, 1])

        Ds = tf.complex(Ds, tf.zeros((self.num_frames*2, nx, nx)))
        Ds_F = tf.signal.fft2d(tf.signal.ifftshift(Ds, axes = (1, 2)))
        Ds_F_conj = tf.math.conj(Ds_F)

        self.calc(alphas=alphas, normalize = False)
        Ps = self.otf_vals
        Ps = tf.signal.ifftshift(Ps, axes=(1, 2))

        Ps_conj = tf.math.conj(Ps)
    
        num = tf.math.reduce_sum(tf.multiply(Ds_F_conj, Ps), axis=[0])
        num = tf.multiply(num, tf.math.conj(num))
        
        den = tf.math.reduce_sum(tf.multiply(Ps, Ps_conj), axis=[0])

        loss = tf.math.reduce_sum(tf.multiply(Ds_F, Ds_F_conj), axis=[0]) - num/den

        return tf.math.real(loss)
    
    def deconvolve_aberrate(self, x):
        object_F, Ps = self.deconvolve(x, do_fft=False)
        #object_F = tf.signal.fftshift(object_F)
        object_F = tf.tile(tf.reshape(object_F, [1, self.nx, self.nx]), [self.num_frames*2, 1, 1])
        DF = tf.math.multiply(object_F, Ps)
        #DF = tf.signal.ifftshift(DF, axes = (1, 2))
        D = tf.math.real(tf.signal.ifft2d(DF))
        #D = tf.signal.fftshift(D, axes = (1, 2)) # Is it needed?
        D = tf.transpose(D, (1, 2, 0))
        D = tf.reshape(D, [1, D.shape[0], D.shape[1], D.shape[2]])
        return D        