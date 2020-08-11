import matplotlib as mpl
mpl.use('Agg')
import numpy as np

import torch

import utils
import zernike

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))
import floodfill

__DEBUG__ = True
if __DEBUG__:
    import plot


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

class phase_aberration_torch():
    
    def __init__(self, alphas, start_index = 3, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.start_index= start_index
        self.device = device
        if len(np.shape(alphas)) == 0:
            # alphas is an integer, representing jmax
            self.create_pols(alphas)
            self.jmax = alphas
        else:
            self.create_pols(len(alphas))
            self.set_alphas(torch.from_numpy(alphas).to(self.device, dtype=torch.float32))
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
        self.terms = torch.from_numpy(terms).to(self.device, dtype=torch.float32)

    def set_alphas(self, alphas):
        if len(self.pols) != self.jmax:
            self.create_pols(self.jmax)
        self.alphas = alphas
        #self.jmax = tf.shape(self.alphas).eval()[0]
    
            
    def __call__(self, alphas=None):
        #vals = np.zeros(tf.shape(self.terms)[1:])
        #for i in np.arange(0, len(self.terms)):
        #    vals += self.terms[i] * self.alphas[i]
        nx = self.terms.size()[1]
        if alphas is None:
            alphas = self.alphas
        shape1 = list(alphas.size()) + [1, 1]
        shape2 = [1]*len(alphas.size()) + [nx, nx]
        alphas = alphas.view(shape1).repeat(shape2)
        #alphas1 = tf.complex(alphas1, tf.zeros((self.jmax, nx, nx)))
        vals = torch.sum(alphas * self.terms, dim=0)
        #vals = tf.math.reduce_sum(tf.math.multiply(self.terms, tf.reshape(self.alphas, [self.jmax, 1, 1])), 0)
        return vals
    
    def set_terms(self, terms):
        self.terms = torch.from_numpy(terms).to(self.device, dtype=torch.float32)
        

'''
Coherent transfer function, also called as generalized pupil function
'''
class coh_trans_func_torch():

    def __init__(self, pupil_func = None, phase_aberr = None, defocus_func = None, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.pupil_func = pupil_func
        self.phase_aberr = phase_aberr
        self.defocus_func = defocus_func
        self.device = device
    
        self.i = torch.tensor(1.j)

    def set_phase_aberr(self, phase_aberr):
        self.phase_aberr = phase_aberr
    
    def set_pupil(self, pupil):
        self.nx = pupil.shape[0]
        self.pupil = torch.from_numpy(pupil).to(self.device, dtype=torch.complex64)
        
    #def set_defocus(self, defocus):
    #    self.defocus = tf.complex(tf.constant(defocus, dtype='float32'), tf.zeros((defocus.shape[0], defocus.shape[1]), dtype='float32'))

    def set_diversity(self, diversity):
        #self.diversity = tf.complex(diversity, tf.zeros_like(diversity, dtype='float32'))
        self.diversity = diversity

    def calc(self, xs):
        if self.phase_aberr is not None:
            self.phase_aberr.calc_terms(xs)
        if self.pupil_func is not None:
            pupil = self.pupil_func(xs)
        if self.defocus_func is not None:
            defocus = self.defocus_func(xs)

            diversity = np.zeros((2, defocus.shape[0], defocus.shape[1]))
            diversity[1] = defocus
            
            self.diversity = torch.from_numpy(diversity).to(self.device, dtype=torch.float32)
            #self.diversity = tf.complex(tf.constant(diversity, dtype='float32'), tf.zeros_like(diversity, dtype='float32'))
        else:
            assert(False)
        
        
        self.pupil = torch.from_numpy(pupil).to(self.device, dtype=torch.complex64)
        
    def __call__(self, alphas=None, diversity=None):
        self.phase = self.phase_aberr(alphas)
        #self.phase = tf.complex(self.phase, tf.zeros((self.phase.shape[0], self.phase.shape[1]), dtype='float32'))

        if diversity is None:
            diversity = self.diversity
        #else:
        #    diversity = tf.complex(diversity, tf.zeros_like(diversity, dtype='float32'))
        
        phase = self.phase + diversity[0]
        focus_val = self.pupil * (torch.cos(-phase) + torch.sin(-phase)*1.j)
        phase = self.phase + diversity[1]
        defocus_val = self.pupil * (torch.cos(-phase) + torch.sin(-phase)*1.j)
        
        #focus_val = tf.math.multiply(self.pupil, tf.math.exp(tf.math.scalar_mul(self.i, tf.math.add(self.phase, diversity[0]))))
        #defocus_val = tf.math.multiply(self.pupil, tf.math.exp(tf.math.scalar_mul(self.i, tf.math.add(self.phase, diversity[1]))))

        #return tf.concat(tf.reshape(focus_val, [1, focus_val.shape[0], focus_val.shape[1]]), tf.reshape(defocus_val, [1, defocus_val.shape[0], defocus_val.shape[1]]), 0)
        return torch.cat([focus_val.unsqueeze(0), defocus_val.unsqueese(0)])

    #def get_defocus_val(self, focus_val):
    #    return tf.math.multiply(focus_val, tf.math.exp(tf.math.scalar_mul(self.i, self.defocus)))

'''
def smart_fltr(F_image, threshold=1e-6):
    
    def fn(F_image):
        F_image = np.fft.fftshift(F_image.numpy())
        modulus = (F_image*F_image.conj()).real
        max_modulus = np.max(modulus)
        mask = np.zeros_like(modulus)
        mask[modulus > max_modulus*threshold] = 1
        ff = floodfill.floodfill(mask)
        ff.fill(mask.shape[0]//2, mask.shape[1]//2)
        ff = floodfill.floodfill(ff.labels)#, compFunc = lambda x, y: x>=y)
        ff.fill(0, 0)
        ff = floodfill.floodfill(ff.labels, compFunc = lambda x, y: x<=y)
        ff.fill(mask.shape[0]//2, mask.shape[1]//2)
        F_image[ff.labels == 0] = 0
        if __DEBUG__:
            my_plot = plot.plot()
            my_plot.colormap(np.log((F_image*F_image.conj()).real+1))
            my_plot.save("filtered.png")
            my_plot.close()
            my_plot = plot.plot()
            my_plot.colormap(mask)
            my_plot.save("mask.png")
            my_plot.close()
            my_plot = plot.plot()
            my_plot.colormap(ff.labels)
            my_plot.save("floodfill.png")
            my_plot.close()
        return tf.constant(np.fft.fftshift(F_image), dtype='complex64')
    return tf.map_fn(lambda F_image: fn(F_image), F_image, dtype='complex64')
'''

class psf_torch():

    '''
        diameter in centimeters
        wavelength in Angstroms
    '''
    def __init__(self, coh_trans_func, nx=None, arcsec_per_px=None, diameter=None, wavelength=None, corr_or_fft=False, 
                 num_frames=1, batch_size=1, set_diversity=False, mode=1, sum_over_batch=True, fltr=None, tt_weight=1.0, 
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):#, zero_avg_tiptilt=True):
        self.device = device
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
            self.fltr = torch.from_numpy(fltr).to(self.device, dtype=torch.complex64)
        else:
            self.fltr = None
            
        self.jmax_used = None
        self.tt_weight = torch.from_numpy(tt_weight).to(self.device, dtype=torch.float32)
        #self.zero_avg_tiptilt = zero_avg_tiptilt
        

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

    def calc(self, alphas, diversity, normalize=True, calc_psf=False):
        #self.incoh_vals = tf.Variable(tf.zeros([self.num_frames*2, self.nx, self.nx], dtype="complex64"))
        #self.otf_vals = tf.Variable(tf.zeros([self.num_frames*2, self.nx, self.nx], dtype="complex64"))
        
        jmax = self.coh_trans_func.phase_aberr.jmax

        if self.jmax_used is not None and self.jmax_used < jmax:
            mask1 = torch.ones(self.jmax_used)
            mask2 = torch.zeros(jmax - self.jmax_used)
            mask = torch.cat([mask1, mask2], axis=0)
            alphas = alphas * mask

        #self.coh_trans_func.phase_aberr.set_alphas(alphas)

        coh_vals = self.coh_trans_func(alphas, diversity)
        wf = self.coh_trans_func.phase
        #wf = tf.complex(self.coh_trans_func.phase, tf.zeros((tf.shape(self.coh_trans_func.phase)[0], tf.shape(self.coh_trans_func.phase)[1]), dtype='float32'))
    
        if self.corr_or_fft:
            corr = fftconv(coh_vals, torch.conj(coh_vals[:, ::-1, ::-1]), mode='full', reorder_channels_before=False, reorder_channels_after=False)/(self.nx*self.nx)
            #vals = torch.real(tf.signal.fftshift(torch.ifft2d(torch.ifftshift(corr, axes=(1, 2))), axes=(1, 2)))
            vals = torch.real(torch.ifft(corr, 2))
            #vals = tf.transpose(vals, (2, 0, 1))
        else:
            
            vals = torch.ifft(coh_vals, 2)
            vals1 = torch.real(vals * torch.conj(vals)) + 0.j

            #corr = tf.signal.fftshift(tf.signal.fft2d(vals1), axes=(1, 2))
            corr = torch.fft(vals1, 2)
            
            #if normalize:
            #    corr /= tf.reduce_sum(self.coh_trans_func.pupil)

        #if calc_psf:
        #    # This block is currently not used (needed for more direct psf calculation)
        #    vals = tf.signal.ifftshift(vals1, axes=(1, 2))
        #    if normalize:
        #        norm = tf.tile(tf.math.reduce_sum(vals, axis = (1, 2), keepdims=True), [1, tf.shape(vals)[1], tf.shape(vals)[1]])
        #        vals = tf.divide(vals, norm)
        
        return corr, torch.view(wf, [1, wf.size()[0], wf.size()[1]])

            


    '''
    dat_F.shape = [l, 2, nx, nx]
    alphas.shape = [l, jmax]
    '''
    def multiply(self, dat_F, alphas, diversity):
        otf_vals, _ = self.calc(alphas, diversity)
        return dat_F * otf_vals


    def aberrate(self, obj, alphas, diversity):
        #nx = self.nx
        #jmax = self.coh_trans_func.phase_aberr.jmax

        obj = obj.repeat(1, self.num_frames*2, 1, 1) + 0.j
        
        fobj = torch.fft(obj, 2)
        #fobj = tf.signal.fftshift(fobj, axes = (2, 3))
    
        DF = self.multiply(fobj, alphas, diversity)
        #DF = tf.signal.ifftshift(DF, axes = (2, 3))
        D = torch.real(torch.ifft(DF, 2))
        #D = tf.signal.fftshift(D, axes = (1, 2)) # Is it needed?
        
        #D = tf.transpose(D, (0, 2, 3, 1))
        return D

    # For numpy inputs
    def Ds_reconstr(self, DP_real, DP_imag, PP, alphas, diversity):
        reconstr = self.reconstr(DP_real, DP_imag, PP, do_fft = False)
        reconstr = reconstr.view(self.batch_size, 1, self.nx, self.nx)
        reconstr = reconstr.repeat(1, 2*self.num_frames, 1, 1)
        DF = self.multiply(reconstr, alphas, diversity)
        #DF = tf.signal.ifftshift(DF, axes = (2, 3))
        D = torch.real(torch.ifft(DF, 2))
        #D = tf.signal.fftshift(D, axes = (2, 3))
        #D = tf.transpose(D, (0, 2, 3, 1))
        return D

    def Ds_reconstr2(self, reconstr, alphas, diversity):
        DF = self.multiply(reconstr, alphas, diversity)
        #DF = tf.signal.ifftshift(DF, axes = (2, 3))
        D = torch.real(torch.ifft(DF, 2))
        #D = tf.signal.fftshift(D, axes = (2, 3))
        #D = tf.transpose(D, (0, 2, 3, 1))
        return D

    # For numpy inputs
    def reconstr(self, DP_real, DP_imag, PP, do_fft = True):
        one_obj = False
        if len(DP_real.shape) == 2:
            one_obj = True
            DP_real = np.reshape(DP_real, [1, self.nx, self.nx])
            DP_imag = np.reshape(DP_imag, [1, self.nx, self.nx])
            PP = np.reshape(PP, [1, self.nx, self.nx])
        DP = torch.from_numpy(DP_real + 1.j*DP_imag).to(self.device, dtype=torch.complex64)
        #DP = tf.reshape(DP, [1, self.nx, self.nx])
        #PP = tf.complex(tf.reshape(PP, [1, self.nx, self.nx]), tf.zeros((1, self.nx, self.nx)))
        PP = torch.from_numpy(PP+0.j).to(self.device, dtype=torch.complex64)
        
        obj, loss = self.reconstr_(torch.conj(DP), PP, do_fft)
        if one_obj:
            return obj[0]
        else:
            return obj

    def reconstr_(self, DP, PP, do_fft = True):
        eps = torch.tensor(1e-10 + 0.j).to(self.device, dtype=torch.complex64)
        F_image = (DP + eps)/(PP + eps)
        
        
        loss = -torch.sum(torch.real(F_image * torch.conj(DP))) # Without DD part
        
        if self.fltr is not None:
            #F_image = smart_fltr(F_image)
            F_image = F_image * self.fltr
    
        if do_fft:
            image = torch.real(torch.ifft(F_image, 2))
        else:
            image = F_image
        #image = tf.signal.ifftshift(image, axes=(1, 2))
        return image, loss

    def deconvolve(self, alphas, diversity, Ds, do_fft = True):
        #nx = self.nx
        #jmax = self.coh_trans_func.phase_aberr.jmax
        
        #size1 = self.batch_size*jmax*self.num_frames
        #size1a = self.num_frames*jmax
        #size2 = self.batch_size*nx*nx*self.num_frames*2
        #if self.set_diversity:
        #    size1 += self.batch_size*nx*nx*2
        #    size1a += 2*nx*nx

        #x = tf.reshape(x, [size1 + size2])
        
        #alphas_diversity = tf.reshape(tf.slice(x, [0], [size1]), [self.batch_size, size1a])
        #Ds = tf.transpose(tf.reshape(tf.slice(x, [size1], [size2]), [self.batch_size, nx, nx, self.num_frames*2]), [0, 3, 1, 2])


        #Ds = tf.complex(Ds, tf.zeros((self.batch_size, self.num_frames*2, nx, nx)))
        
        #Ds_F = tf.signal.fft2d(tf.signal.ifftshift(Ds, axes = (2, 3)))
        Ds_F = torch.fft(Ds + 0.j, 2)

        Ps1, _ = self.calc(alphas, diversity)
        
        Ps = Ps1#tf.signal.ifftshift(Ps1, axes=(2, 3))
        Ps_conj = torch.conj(Ps)
    
        num = torch.sum(Ds_F*Ps_conj, axis=1)
        den = torch.sum(Ps*Ps_conj, axis=1)
        
        image, loss = self.reconstr_(num, den, do_fft)
        #F_image = tf.divide(num, den)
        #if self.fltr is not None:
        #    F_image = F_image * tf.constant(self.fltr)
        #if do_fft:
        #    image = tf.math.real(tf.signal.ifft2d(F_image))
        #else:
        #    image = F_image
        #image = tf.signal.ifftshift(image, axes=(1, 2))
        return image, Ps1, self.wf, loss
        
    def mfbd_loss(self, alphas, diversity, Ds, tt, DD_DP_PP):
        nx = self.nx
        mode = self.mode
        #alphas = tf.reshape(tf.slice(x, [0], [size]), [self.batch_size, self.num_frames, jmax])

        if self.sum_over_batch:
            tt_sum = tf.reduce_sum(tt, axis=[0, 1])
            tt_sum = tt_sum * tt_sum
            tt_sum = tf.reduce_sum(tt_sum)
            tt_sum = tf.tile(tf.reshape(tt_sum, [1, 1]), [nx, nx])
        else:
            tt_sum = tf.reduce_sum(tt, axis=[1])
            tt_sum = tt_sum * tt_sum
            tt_sum = tf.reduce_sum(tt_sum, axis=[1])
            tt_sum = tf.tile(tf.reshape(tt_sum, [self.batch_size, 1, 1]), [1, nx, nx])
            
        
        #Ds_F = tf.signal.fft2d(tf.signal.ifftshift(Ds, axes = (2, 3)))
        Ds_F = torch.fft(Ds + 0.j, 2)
            
        Ds_F_conj = torch.conj(Ds_F)

        self.calc(alphas, diversity, normalize = False)
        Ps = self.otf_vals
        #Ps = tf.signal.ifftshift(Ps, axes=(2, 3))

        Ps_conj = torch.conj(Ps)
    
        num = torch.sum(Ds_F_conj*Ps, axis=1)
        
        if mode >= 2:
            #if self.sum_over_batch:
            #    num = tf.reshape(num, [1, nx, nx])
            #else:
            num = num.view(self.batch_size, 1, nx, nx)
            DP_real = torch.real(num)
            DP_imag = torch.imag(num)

            num1 = DD_DP_PP[:, 1] + DD_DP_PP[:, 2]*1.j
            #num1 = tf.complex(tf.slice(DD_DP_PP, [0, 1, 0, 0], [self.batch_size, 1, nx, nx]), 
            #                             tf.slice(DD_DP_PP, [0, 2, 0, 0], [self.batch_size, 1, nx, nx]))
            num = num + num1
        if self.sum_over_batch:
            num = torch.sum(num, axis=0)
        num = num*torch.conj(num)
        num = torch.real(num)
        
        den = torch.sum(Ps*Ps_conj, axis=1)
        den = torch.real(den)
        if mode >= 2:
            #PP1 = tf.slice(DD_DP_PP, [0, 3, 0, 0], [self.batch_size, 1, nx, nx])
            PP1 = DD_DP_PP[:, 3]
            #if self.sum_over_batch:
            #    PP = tf.reshape(den, [1, nx, nx])
            #    PP1 = tf.math.reduce_sum(PP1, axis=[0])
            #else:
            PP = den.view(self.batch_size, 1, nx, nx)
            den = PP + PP1

        if self.sum_over_batch:
            den = torch.sum(den, axis=0)

        eps = torch.tensor(1e-10)
        
        DD = torch.real(torch.sum(Ds_F*Ds_F_conj, axis=1))
        if mode == 1:
            if self.sum_over_batch:
                DD = torch.sum(DD, axis=0)
            return DD - (num + eps)/(den + eps) + self.tt_weight * tt_sum
            #return DD - tf.math.add(num, eps)/tf.math.add(den, eps)
        elif mode >= 2:
            #DD1 = tf.slice(DD_DP_PP, [0, 0, 0, 0], [self.batch_size, 1, nx, nx])
            DD1 = DD_DP_PP[:, 0]
            
            #if self.sum_over_batch:
            #    DD = tf.reshape(DD, [1, nx, nx])
            #    DD1 = tf.math.reduce_sum(DD1, axis=[0])
            #else:
            DD = DD.view(self.batch_size, 1, nx, nx)
            DD1 = DD + DD1 
            if self.sum_over_batch:
                DD1 = torch.sum(DD1, axis=0)
            loss = DD1 - (num + eps)/(den + eps)
            
            #if self.sum_over_batch:
            #    loss = tf.tile(tf.reshape(loss, [1, 1, nx, nx]), [self.batch_size, 1, 1, 1])
            return loss, DD, DP_real, DP_imag, PP

    
