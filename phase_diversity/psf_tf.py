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
    
            
    def __call__(self, alphas=None):
        #vals = np.zeros(tf.shape(self.terms)[1:])
        #for i in np.arange(0, len(self.terms)):
        #    vals += self.terms[i] * self.alphas[i]
        nx = self.terms.shape[1]
        if alphas is None:
            alphas = tf.tile(tf.reshape(self.alphas, [self.jmax, 1, 1]), multiples=[1, nx, nx])
        else:
            alphas = tf.tile(tf.reshape(alphas, [self.jmax, 1, 1]), multiples=[1, nx, nx])
        #alphas1 = tf.complex(alphas1, tf.zeros((self.jmax, nx, nx)))
        vals = tf.math.reduce_sum(tf.math.multiply(self.terms, alphas), 0)
        #vals = tf.math.reduce_sum(tf.math.multiply(self.terms, tf.reshape(self.alphas, [self.jmax, 1, 1])), 0)
        return vals
    
    def set_terms(self, terms):
        self.terms = tf.constant(terms, dtype='float32')
        

'''
Coherent transfer function, also called as generalized pupil function
'''
class coh_trans_func_tf():

    def __init__(self, pupil_func = None, phase_aberr = None, defocus_func = None):
        self.pupil_func = pupil_func
        self.phase_aberr = phase_aberr
        self.defocus_func = defocus_func
    
        self.i = tf.constant(1.j, dtype='complex64')

    def set_phase_aberr(self, phase_aberr):
        self.phase_aberr = phase_aberr
    
    def set_pupil(self, pupil):
        self.nx = pupil.shape[0]
        self.pupil = tf.constant(pupil, dtype='float32')
        self.pupil = tf.complex(self.pupil, tf.zeros((pupil.shape[0], pupil.shape[1]), dtype='float32'))
        
    #def set_defocus(self, defocus):
    #    self.defocus = tf.complex(tf.constant(defocus, dtype='float32'), tf.zeros((defocus.shape[0], defocus.shape[1]), dtype='float32'))

    def set_diversity(self, diversity):
        #self.diversity = tf.complex(tf.constant(diversity, dtype='float32'), tf.zeros_like(diversity, dtype='float32'))
        # diversity is already a tensor
        self.diversity = tf.complex(diversity, tf.zeros_like(diversity, dtype='float32'))

    def calc(self, xs):
        if self.phase_aberr is not None:
            self.phase_aberr.calc_terms(xs)
        if self.pupil_func is not None:
            pupil = self.pupil_func(xs)
        if self.defocus_func is not None:
            defocus = self.defocus_func(xs)

            diversity = np.zeros((2, defocus.shape[0], defocus.shape[1]))
            diversity[1] = defocus
            
            self.diversity = tf.complex(tf.constant(diversity, dtype='float32'), tf.zeros_like(diversity, dtype='float32'))
        else:
            assert(False)
        
        
        self.pupil = tf.constant(pupil, dtype='float32')
        self.pupil = tf.complex(self.pupil, tf.zeros((pupil.shape[0], pupil.shape[1]), dtype='float32'))
        
    def __call__(self, alphas=None, diversity=None):
        self.phase = self.phase_aberr(alphas)
        self.phase = tf.complex(self.phase, tf.zeros((self.phase.shape[0], self.phase.shape[1]), dtype='float32'))

        if diversity is None:
            diversity = self.diversity
        else:
            diversity = tf.complex(diversity, tf.zeros_like(diversity, dtype='float32'))
        focus_val = tf.math.multiply(self.pupil, tf.math.exp(tf.math.scalar_mul(self.i, tf.math.add(self.phase, diversity[0]))))
        defocus_val = tf.math.multiply(self.pupil, tf.math.exp(tf.math.scalar_mul(self.i, tf.math.add(self.phase, diversity[1]))))

        #return tf.concat(tf.reshape(focus_val, [1, focus_val.shape[0], focus_val.shape[1]]), tf.reshape(defocus_val, [1, defocus_val.shape[0], defocus_val.shape[1]]), 0)
        return tf.stack([focus_val, defocus_val])

    #def get_defocus_val(self, focus_val):
    #    return tf.math.multiply(focus_val, tf.math.exp(tf.math.scalar_mul(self.i, self.defocus)))

class psf_tf():

    '''
        diameter in centimeters
        wavelength in Angstroms
    '''
    def __init__(self, coh_trans_func, nx=None, arcsec_per_px=None, diameter=None, wavelength=None, corr_or_fft=False, 
                 num_frames=1, batch_size=1, set_diversity=False, mode=1, sum_over_batch=True, fltr=None):
        self.coh_trans_func = coh_trans_func
        if nx is None:
            # Everything is precalculated
            self.nx = coh_trans_func.nx
        else:
            self.nx = nx
            coords, rc, x_limit = utils.get_coords(nx, arcsec_per_px, diameter, wavelength)
            self.coords = coords
            x_min = np.min(self.coords, axis=(0,1))
            x_max = np.max(self.coords, axis=(0,1))
            print("psf_coords", x_min, x_max, np.shape(self.coords))
            np.testing.assert_array_almost_equal(x_min, -x_max)
            self.coh_trans_func.calc(self.coords)

        self.incoh_vals = None
        self.otf_vals = None
        self.corr = None # only for testing purposes

        self.corr_or_fft = corr_or_fft
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.set_diversity = set_diversity
        self.mode = mode
        self.sum_over_batch = sum_over_batch
        
        if fltr is not None:
            self.fltr = tf.constant(fltr, dtype='complex64')
        else:
            self.fltr = None
            
        self.jmax_used = None
        
        

    def set_num_frames(self, num_frames):
        self.num_frames = num_frames

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_jmax_used(self, jmax_used):
        self.jmax_used = jmax_used

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

    def calc(self, data=None, normalize=True):
        #self.incoh_vals = tf.Variable(tf.zeros([self.num_frames*2, self.nx, self.nx], dtype="complex64"))
        #self.otf_vals = tf.Variable(tf.zeros([self.num_frames*2, self.nx, self.nx], dtype="complex64"))
        
        #@tf.contrib.eager.defun
        def fn1(alphas, diversity):
            jmax = self.coh_trans_func.phase_aberr.jmax
    
            alphas = tf.slice(alphas, [0], [jmax])
            if self.jmax_used is not None and self.jmax_used < jmax:
                mask1 = tf.ones(self.jmax_used)
                mask2 = tf.zeros(jmax - self.jmax_used)
                mask = tf.concat([mask1, mask2], axis=0)
                alphas = alphas * mask

            #self.coh_trans_func.phase_aberr.set_alphas(alphas)

            coh_vals = self.coh_trans_func(alphas, diversity)
        
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

        #@tf.contrib.eager.defun
        def fn2(alphas_diversity):
            nx = self.nx
            jmax = self.coh_trans_func.phase_aberr.jmax
        
            alphas = tf.reshape(tf.slice(alphas_diversity, [0], [self.num_frames*jmax]), [self.num_frames, jmax])
            if self.set_diversity:
                diversity = tf.reshape(tf.slice(alphas_diversity, [self.num_frames*jmax], [2*nx*nx]), [2, nx, nx])
                #self.coh_trans_func.set_diversity(diversity)
            else:
                diversity = None
                
            return tf.map_fn(lambda alphas: fn1(alphas, diversity), alphas, dtype='complex64')
            
            
        otf_vals = tf.map_fn(fn2, data, dtype='complex64')
        self.otf_vals = tf.reshape(otf_vals, [self.batch_size, self.num_frames*2, self.nx, self.nx])
            
        return self.incoh_vals


    '''
    dat_F.shape = [l, 2, nx, nx]
    alphas.shape = [l, jmax]
    '''
    def multiply(self, dat_F, alphas_diversity):
        if alphas_diversity is not None or self.otf_vals is None:
            self.calc(data=alphas_diversity)
        return tf.math.multiply(dat_F, self.otf_vals)


    def aberrate(self, x):
        nx = self.nx
        jmax = self.coh_trans_func.phase_aberr.jmax
        x = tf.reshape(x, [self.batch_size*(self.batch_size*jmax*self.num_frames + nx*nx)])
        alphas = tf.reshape(tf.slice(x, [0], [self.batch_size*jmax*self.num_frames]), [self.batch_size, self.num_frames*jmax])
        obj = tf.reshape(tf.slice(x, [self.batch_size*jmax*self.num_frames], [self.batch_size*nx*nx]), [self.batch_size, 1, nx, nx])
        obj = tf.tile(obj, [1, self.num_frames*2, 1, 1])
        
        fobj = tf.signal.fft2d(tf.complex(obj, tf.zeros((self.batch_size, self.num_frames*2, nx, nx))))
        fobj = tf.signal.fftshift(fobj, axes = (2, 3))
    
        DF = self.multiply(fobj, alphas)
        DF = tf.signal.ifftshift(DF, axes = (2, 3))
        D = tf.math.real(tf.signal.ifft2d(DF))
        #D = tf.signal.fftshift(D, axes = (1, 2)) # Is it needed?
        D = tf.transpose(D, (0, 2, 3, 1))
        #D = tf.reshape(D, [1, D.shape[0], D.shape[1], D.shape[2]])

        #D = tf.transpose(tf.reshape(obj, [1, self.num_frames*2, self.nx, self.nx]), [0, 2, 3, 1])
        return D

    # For numpy inputs
    def Ds_reconstr(self, DP_real, DP_imag, PP, alphas_diversity):
        reconstr = self.reconstr(DP_real, DP_imag, PP, do_fft = False)
        reconstr = tf.reshape(reconstr, [self.batch_size, 1, self.nx, self.nx])
        reconstr = tf.tile(reconstr, [1, 2*self.num_frames, 1, 1])
        DF = self.multiply(reconstr, alphas_diversity)
        DF = tf.signal.ifftshift(DF, axes = (2, 3))
        D = tf.math.real(tf.signal.ifft2d(DF))
        D = tf.signal.fftshift(D, axes = (2, 3))
        D = tf.transpose(D, (0, 2, 3, 1))
        return D

    def Ds_reconstr2(self, reconstr, alphas_diversity):
        DF = self.multiply(reconstr, alphas_diversity)
        DF = tf.signal.ifftshift(DF, axes = (2, 3))
        D = tf.math.real(tf.signal.ifft2d(DF))
        D = tf.signal.fftshift(D, axes = (2, 3))
        D = tf.transpose(D, (0, 2, 3, 1))
        return D

    # For numpy inputs
    def reconstr(self, DP_real, DP_imag, PP, do_fft = True):
        one_obj = False
        if len(DP_real.shape) == 2:
            one_obj = True
            DP_real = np.reshape(DP_real, [1, self.nx, self.nx])
            DP_imag = np.reshape(DP_imag, [1, self.nx, self.nx])
            PP = np.reshape(PP, [1, self.nx, self.nx])
        DP = tf.complex(tf.constant(DP_real, dtype='float32'), tf.constant(DP_imag, dtype='float32'))
        #DP = tf.reshape(DP, [1, self.nx, self.nx])
        #PP = tf.complex(tf.reshape(PP, [1, self.nx, self.nx]), tf.zeros((1, self.nx, self.nx)))
        PP = tf.complex(tf.constant(PP, dtype='float32'), tf.zeros((1, self.nx, self.nx), dtype='float32'))
        
        obj = self.reconstr_(tf.math.conj(DP), PP, do_fft)
        if one_obj:
            return obj[0]
        else:
            return obj

    def reconstr_(self, DP, PP, do_fft = True):
        F_image = tf.divide(DP, PP)
        
        if self.fltr is not None:
            F_image = F_image * tf.constant(self.fltr)
    
        if do_fft:
            image = tf.math.real(tf.signal.ifft2d(F_image))
        else:
            image = F_image
        image = tf.signal.ifftshift(image, axes=(1, 2))
        return image

    def deconvolve(self, x, do_fft = True):
        nx = self.nx
        jmax = self.coh_trans_func.phase_aberr.jmax
        
        size1 = self.batch_size*jmax*self.num_frames
        size1a = self.num_frames*jmax
        size2 = self.batch_size*nx*nx*self.num_frames*2
        if self.set_diversity:
            size1 += self.batch_size*nx*nx*2
            size1a += 2*nx*nx

        x = tf.reshape(x, [size1 + size2])
        
        
        #x = tf.reshape(x, [self.batch_size*(jmax*self.num_frames + nx*nx*self.num_frames*2)])
        #alphas = tf.reshape(tf.slice(x, [0], [self.batch_size*jmax*self.num_frames]), [self.batch_size, self.num_frames, jmax])
        ##Ds = tf.reshape(tf.slice(x, [jmax*self.num_frames], [nx*nx*self.num_frames*2]), [self.num_frames*2, nx, nx])
        #Ds = tf.transpose(tf.reshape(tf.slice(x, [self.batch_size*jmax*self.num_frames], [self.batch_size*nx*nx*self.num_frames*2]), [self.batch_size, nx, nx, self.num_frames*2]), [0, 3, 1, 2])

        alphas_diversity = tf.reshape(tf.slice(x, [0], [size1]), [self.batch_size, size1a])
        #Ds = tf.reshape(tf.slice(x, [jmax*self.num_frames], [nx*nx*self.num_frames*2]), [self.num_frames*2, nx, nx])
        Ds = tf.transpose(tf.reshape(tf.slice(x, [size1], [size2]), [self.batch_size, nx, nx, self.num_frames*2]), [0, 3, 1, 2])


        Ds = tf.complex(Ds, tf.zeros((self.batch_size, self.num_frames*2, nx, nx)))
        Ds_F = tf.signal.fft2d(tf.signal.ifftshift(Ds, axes = (2, 3)))

        self.calc(data=alphas_diversity)
        Ps1 = self.otf_vals
        
        Ps = tf.signal.ifftshift(Ps1, axes=(2, 3))
        Ps_conj = tf.math.conj(Ps)
    
        num = tf.math.reduce_sum(tf.multiply(Ds_F, Ps_conj), axis=[1])
        den = tf.math.reduce_sum(tf.multiply(Ps, Ps_conj), axis=[1])
        
        image = self.reconstr_(num, den, do_fft)
        #F_image = tf.divide(num, den)
        #if self.fltr is not None:
        #    F_image = F_image * tf.constant(self.fltr)
        #if do_fft:
        #    image = tf.math.real(tf.signal.ifft2d(F_image))
        #else:
        #    image = F_image
        #image = tf.signal.ifftshift(image, axes=(1, 2))
        return image, Ps1
        
    def mfbd_loss(self, x):
        nx = self.nx
        jmax = self.coh_trans_func.phase_aberr.jmax
        mode = self.mode
        size1 = self.batch_size*jmax*self.num_frames
        size1a = self.num_frames*jmax
        size2 = self.batch_size*nx*nx*self.num_frames*2
        size3 = 0
        if self.set_diversity:
            size1 += self.batch_size*nx*nx*2
            size1a += 2*nx*nx
        if mode >= 2:
            size3 = self.batch_size*nx*nx*4

        x = tf.reshape(x, [size1 + size2 + size3])

        #alphas = tf.reshape(tf.slice(x, [0], [size]), [self.batch_size, self.num_frames, jmax])

        alphas_diversity = tf.reshape(tf.slice(x, [0], [size1]), [self.batch_size, size1a])
        #Ds = tf.reshape(tf.slice(x, [jmax*self.num_frames], [nx*nx*self.num_frames*2]), [self.num_frames*2, nx, nx])
        Ds = tf.transpose(tf.reshape(tf.slice(x, [size1], [size2]), [self.batch_size, nx, nx, self.num_frames*2]), [0, 3, 1, 2])
        if mode >= 2:
            DD_DP_PP = tf.reshape(tf.slice(x, [size1 + size2], [size3]), [self.batch_size, 4, nx, nx])
        
        Ds = tf.complex(Ds, tf.zeros((self.batch_size, self.num_frames*2, nx, nx)))
        Ds_F = tf.signal.fft2d(tf.signal.ifftshift(Ds, axes = (2, 3)))
        Ds_F_conj = tf.math.conj(Ds_F)

        self.calc(data=alphas_diversity, normalize = False)
        Ps = self.otf_vals
        Ps = tf.signal.ifftshift(Ps, axes=(2, 3))

        Ps_conj = tf.math.conj(Ps)
    
        num = tf.math.reduce_sum(tf.multiply(Ds_F_conj, Ps), axis=[1])
        
        if mode >= 2:
            #if self.sum_over_batch:
            #    num = tf.reshape(num, [1, nx, nx])
            #else:
            num = tf.reshape(num, [self.batch_size, 1, nx, nx])
            DP_real = tf.math.real(num)
            DP_imag = tf.math.imag(num)

            num1 = tf.complex(tf.slice(DD_DP_PP, [0, 1, 0, 0], [self.batch_size, 1, nx, nx]), 
                                         tf.slice(DD_DP_PP, [0, 2, 0, 0], [self.batch_size, 1, nx, nx]))
            num = tf.math.add(num, num1)
        if self.sum_over_batch:
            num = tf.math.reduce_sum(num, axis=[0])
        num = tf.multiply(num, tf.math.conj(num))
        num = tf.math.real(num)
        
        den = tf.math.reduce_sum(tf.multiply(Ps, Ps_conj), axis=[1])
        den = tf.math.real(den)
        if mode >= 2:
            PP1 = tf.slice(DD_DP_PP, [0, 3, 0, 0], [self.batch_size, 1, nx, nx])
            #if self.sum_over_batch:
            #    PP = tf.reshape(den, [1, nx, nx])
            #    PP1 = tf.math.reduce_sum(PP1, axis=[0])
            #else:
            PP = tf.reshape(den, [self.batch_size, 1, nx, nx])            
            den = tf.math.add(PP, PP1)

        if self.sum_over_batch:
            den = tf.math.reduce_sum(den, axis=[0])

        eps = tf.constant(1e-10)
        
        DD = tf.math.real(tf.math.reduce_sum(tf.multiply(Ds_F, Ds_F_conj), axis=[1]))
        if mode == 1:
            if self.sum_over_batch:
                DD = tf.math.reduce_sum(DD, axis=[0])
            return DD - tf.math.add(num, eps)/tf.math.add(den, eps)
            #return DD - tf.math.add(num, eps)/tf.math.add(den, eps)
        elif mode >= 2:
            DD1 = tf.slice(DD_DP_PP, [0, 0, 0, 0], [self.batch_size, 1, nx, nx])
            #if self.sum_over_batch:
            #    DD = tf.reshape(DD, [1, nx, nx])
            #    DD1 = tf.math.reduce_sum(DD1, axis=[0])
            #else:
            DD = tf.reshape(DD, [self.batch_size, 1, nx, nx])
            DD1 = tf.math.add(DD, DD1) 
            if self.sum_over_batch:
                DD1 = tf.math.reduce_sum(DD1, axis=[0])
            loss = DD1 - tf.math.add(num, eps)/tf.math.add(den, eps)
            
            if self.sum_over_batch:
                loss = tf.tile(tf.reshape(loss, [1, 1, nx, nx]), [self.batch_size, 1, 1, 1])
            #    return tf.tile(tf.reshape(tf.concat([loss, DD, DP_real, DP_imag, PP], axis=0), [1, 5, nx, nx]), [self.batch_size, 1, 1, 1])
            #    return tf.tile(tf.reshape(tf.concat([loss, DD, DP_real, DP_imag, PP], axis=0), [1, 5, nx, nx]), [self.batch_size, 1, 1, 1])
            #else:
            return tf.concat([loss, DD, DP_real, DP_imag, PP], axis=1)

    
    def deconvolve_aberrate(self, x):
        fobj, Ps = self.deconvolve(x, do_fft=False)
        DF = tf.math.multiply(fobj, Ps)
        DF = tf.signal.ifftshift(DF, axes = (2, 3))
        D = tf.math.real(tf.signal.ifft2d(DF))
        D = tf.signal.fftshift(D, axes = (2, 3)) # Is it needed?
        D = tf.transpose(D, (0, 2, 3, 1))
        #D = tf.reshape(D, [1, D.shape[0], D.shape[1], D.shape[2]])
        return D        
