import matplotlib as mpl
mpl.use('Agg')
import numpy as np

import torch

import utils
import zernike

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))

__DEBUG__ = True
if __DEBUG__:
    import plot


from typing import Optional

###############################################################################
# Methods for convenience until gradient propagation
# supports complex numbers

emulate_complex = True
def create_complex(re, im):
    re = torch.unsqueeze(re, -1)
    im = torch.unsqueeze(im, -1)
    return torch.cat([re, im], -1)

def to_complex(x, re=True):
    x = torch.unsqueeze(x, -1)
    zeros = torch.zeros_like(x)
    if re:
        return torch.cat([x, zeros], -1)
    else:
        return torch.cat([zeros, x], -1)
        

def real(x):
    return x[..., 0]

def imag(x):
    return x[..., 1]

def conj(x):
    re = x[..., 0]
    im = x[..., 1]
    re = torch.unsqueeze(re, -1)
    im = torch.unsqueeze(im, -1)
    return torch.cat([re, im*-1], -1)

def mul(x, y):
    x_re = x[..., 0]
    x_im = x[..., 1]
    y_re = y[..., 0]
    y_im = y[..., 1]
    x_re = torch.unsqueeze(x_re, -1)
    x_im = torch.unsqueeze(x_im, -1)
    y_re = torch.unsqueeze(y_re, -1)
    y_im = torch.unsqueeze(y_im, -1)
    return torch.cat([x_re*y_re-x_im*y_im, x_re*y_im+x_im*y_re], -1)

def div(x, y):
    x_re = x[..., 0]
    x_im = x[..., 1]
    y_re = y[..., 0]
    y_im = y[..., 1]
    x_re = torch.unsqueeze(x_re, -1)
    x_im = torch.unsqueeze(x_im, -1)
    y_re = torch.unsqueeze(y_re, -1)
    y_im = torch.unsqueeze(y_im, -1)
    den = (y_re**2+y_im**2)
    return torch.cat([(x_re*y_re+x_im*y_im)/den, (x_im*y_re-x_re*y_im)/den], -1)
    

def fft(x):
    return torch.fft(x, 2)

def ifft(x):
    return torch.ifft(x, 2)
    
###############################################################################
'''

def real(x):
    return torch.real(x)

def imag(x):
    return torch.imag(x)

def conj(x):
    return torch.conj(x)

def fft(x):
    x_real = torch.real(x)
    x_imag = torch.imag(x)
    x_real = torch.unsqueeze(x_real, -1)# + 0.j
    x_imag = torch.unsqueeze(x_imag, -1)# + 0.j
    x = torch.cat([x_real, x_imag], -1)
        
    fx = torch.fft(x, 2)
    
    return fx[..., 0] + fx[..., 1]*1.j

def ifft(x):
    x_real = torch.real(x)
    x_imag = torch.imag(x)
    x_real = torch.unsqueeze(x_real, -1)# + 0.j
    x_imag = torch.unsqueeze(x_imag, -1)# + 0.j
    x = torch.cat([x_real, x_imag], -1)
        
    fx = torch.ifft(x, 2)
    
    return fx[..., 0] + fx[..., 1]*1.j

def fftshift(x):
    return torch.roll(x, [x.size()[-1]//2, x.size()[-2]//2], [-1, -2])

def ifftshift(x):
    return torch.roll(x, [-x.size()[-1]//2, -x.size()[-2]//2], [-1, -2])
'''
###############################################################################
'''
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
'''

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
        dim = len(alphas.size()) - 1
        #if len(alphas.size()) == 2:
        #    # We have multiple atmospheric frames
        #    dim = 1
        #else:
        #    # We have single atmospheric frame
        #    dim = 0
        shape1 = list(alphas.size()) + [1, 1]
        shape2 = [1]*len(alphas.size()) + [nx, nx]
        alphas = alphas.view(shape1).repeat(shape2)
        #alphas1 = tf.complex(alphas1, tf.zeros((self.jmax, nx, nx)))
        vals = torch.sum(alphas * self.terms, dim=dim)
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
    
        #self.i = torch.tensor(1.j)

    def set_phase_aberr(self, phase_aberr):
        self.phase_aberr = phase_aberr
    
    def set_pupil(self, pupil):
        self.nx = pupil.shape[0]
        #self.pupil = torch.from_numpy(pupil).to(self.device, dtype=torch.complex64)
        self.pupil = to_complex(torch.from_numpy(pupil)).to(self.device, dtype=torch.float32)
        
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
        
        
        #self.pupil = torch.from_numpy(pupil).to(self.device, dtype=torch.complex64)
        self.pupil = to_complex(torch.from_numpy(pupil)).to(self.device, dtype=torch.float32)
        
    def __call__(self, alphas=None, diversity=None):
        self.phase = self.phase_aberr(alphas)
        #self.phase = tf.complex(self.phase, tf.zeros((self.phase.shape[0], self.phase.shape[1]), dtype='float32'))

        if diversity is None:
            diversity = self.diversity
            
        if len(diversity.size()) == 4:
            assert(len(self.phase.size()) == 4)
            div_focus = diversity[:, 0]
            div_defocus = diversity[:, 1]
            
            div_focus = div_focus.unsqueeze(1)
            div_defocus = div_defocus.unsqueeze(1)

            div_focus = div_focus.repeat(1, self.phase.size()[1], 1, 1)
            div_defocus = div_defocus.repeat(1, self.phase.size()[1], 1, 1)
            
        else:
            div_focus = diversity[0]
            div_defocus = diversity[1]
        #else:
        #    diversity = tf.complex(diversity, tf.zeros_like(diversity, dtype='float32'))
        #print("phase, diversity", self.phase.size(), diversity.size())
        phase = self.phase + div_focus
        #focus_val = self.pupil * (torch.cos(-phase) + torch.sin(-phase)*1.j)
        focus_val = mul(self.pupil, create_complex(torch.cos(-phase), torch.sin(-phase)))
        phase = self.phase + div_defocus
        #defocus_val = self.pupil * (torch.cos(-phase) + torch.sin(-phase)*1.j)
        defocus_val = mul(self.pupil, create_complex(torch.cos(-phase), torch.sin(-phase)))
        
        size = list(focus_val.size())
        index = len(size) - 2
        if emulate_complex:
            index -= 1
        size[index - 1] *= 2
        return torch.cat([focus_val.unsqueeze(index), defocus_val.unsqueeze(index)], index).view(size)

    #def get_defocus_val(self, focus_val):
    #    return tf.math.multiply(focus_val, tf.math.exp(tf.math.scalar_mul(self.i, self.defocus)))


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
            #self.fltr = torch.from_numpy(fltr).to(self.device, dtype=torch.complex64)
            self.fltr = to_complex(torch.from_numpy(fltr)).to(self.device, dtype=torch.float32)
        else:
            self.fltr = None
            
        self.jmax_used = None
        self.tt_weight = torch.tensor(tt_weight).to(self.device, dtype=torch.float32)
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
            mask = torch.cat([mask1, mask2], axis=0).to(self.device, dtype=torch.float32)
            alphas = alphas * mask

        #self.coh_trans_func.phase_aberr.set_alphas(alphas)

        coh_vals = self.coh_trans_func(alphas, diversity)
        wf = self.coh_trans_func.phase
        #wf = tf.complex(self.coh_trans_func.phase, tf.zeros((tf.shape(self.coh_trans_func.phase)[0], tf.shape(self.coh_trans_func.phase)[1]), dtype='float32'))
    
        if self.corr_or_fft:
            print("Not implemented")
            #corr = fftconv(coh_vals, torch.conj(coh_vals[:, ::-1, ::-1]), mode='full', reorder_channels_before=False, reorder_channels_after=False)/(self.nx*self.nx)
            #vals = torch.real(fftshift(ifft(ifftshift(corr))))
        else:
            
            vals = ifft(coh_vals)
            vals1 = to_complex(real(mul(vals, conj(vals))))

            #corr = tf.signal.fftshift(tf.signal.fft2d(vals1), axes=(1, 2))
            #corr = fftshift(fft(vals1))
            corr = fft(vals1)
            
            #if normalize:
            #    corr /= tf.reduce_sum(self.coh_trans_func.pupil)

        #if calc_psf:
        #    # This block is currently not used (needed for more direct psf calculation)
        #    vals = tf.signal.ifftshift(vals1, axes=(1, 2))
        #    if normalize:
        #        norm = tf.tile(tf.math.reduce_sum(vals, axis = (1, 2), keepdims=True), [1, tf.shape(vals)[1], tf.shape(vals)[1]])
        #        vals = tf.divide(vals, norm)
        #print("corr", corr.size())
        
        return corr, wf.squeeze()#wf.view(1, wf.size()[0], wf.size()[1])

            


    '''
    dat_F.shape = [num_frames, 2, nx, nx]
    alphas.shape = [num_frames, jmax]
    '''
    def multiply(self, dat_F, alphas, diversity):
        otf_vals, _ = self.calc(alphas, diversity)
        print("dat_F, otf_vals", dat_F.size(), otf_vals.size())
        return mul(dat_F, otf_vals)


    def aberrate(self, obj, alphas, diversity=None):
        #nx = self.nx
        #jmax = self.coh_trans_func.phase_aberr.jmax

        obj = to_complex(obj.unsqueeze(0).repeat(self.num_frames*2, 1, 1))
        print("obj, alphas", obj.size(), alphas.size())
        
        fobj = fft(obj)
        #fobj = fftshift(fobj)
    
        DF = self.multiply(fobj, alphas, diversity)
        #DF = ifftshift(DF)
        D = real(ifft(DF))
        #D = ifftshift(D)
        
        #D = tf.transpose(D, (0, 2, 3, 1))
        return D/(D.size()[1]*D.size()[2])

    # For numpy inputs
    def Ds_reconstr(self, DP_real, DP_imag, PP, alphas, diversity):
        reconstr = self.reconstr(DP_real, DP_imag, PP, do_fft = False)
        reconstr = reconstr.view(self.batch_size, 1, self.nx, self.nx)
        reconstr = reconstr.repeat(1, 2*self.num_frames, 1, 1)
        DF = self.multiply(reconstr, alphas, diversity)
        #DF = tf.signal.ifftshift(DF, axes = (2, 3))
        D = real(ifft(DF))
        #D = tf.signal.fftshift(D, axes = (2, 3))
        #D = tf.transpose(D, (0, 2, 3, 1))
        return D

    def Ds_reconstr2(self, reconstr, alphas, diversity):
        DF = self.multiply(reconstr, alphas, diversity)
        #DF = tf.signal.ifftshift(DF, axes = (2, 3))
        D = real(ifft(DF))
        #D = tf.signal.fftshift(D, axes = (2, 3))
        #D = tf.transpose(D, (0, 2, 3, 1))
        return D

    # For numpy inputs
    def reconstr(self, DP_real, DP_imag, PP, do_fft = True):
        one_obj = False
        print("DP_real, DP_imag, PP", DP_real.size(), DP_imag.size(), PP.size())
        if len(DP_real.shape) == 2:
            one_obj = True
            DP_real = np.reshape(DP_real, [1, self.nx, self.nx])
            DP_imag = np.reshape(DP_imag, [1, self.nx, self.nx])
            PP = np.reshape(PP, [1, self.nx, self.nx])
        #DP = torch.from_numpy(DP_real + 1.j*DP_imag).to(self.device, dtype=torch.complex64)
        DP = create_complex(torch.from_numpy(DP_real), torch.from_numpy(DP_imag)).to(self.device, dtype=torch.float32)
        #PP = torch.from_numpy(PP).to(self.device, dtype=torch.complex64)
        PP = to_complex(torch.from_numpy(PP)).to(self.device, dtype=torch.float32)
        
        obj, loss = self.reconstr_(conj(DP), PP, do_fft)
        
        if one_obj:
            obj = obj[0]

        return obj

    def reconstr_(self, DP, PP, do_fft = True):
        #eps = torch.tensor(1e-10).to(self.device, dtype=torch.complex64)
        eps = to_complex(torch.tensor(1e-10)).to(self.device, dtype=torch.float32)
        #F_image = (DP + eps)/(PP + eps)
        print("DP, PP", DP.size(), PP.size())
        F_image = div(DP + eps, PP + eps)
        
        
        loss = -torch.sum(real(mul(F_image, conj(DP)))) # Without DD part
        
        if self.fltr is not None:
            #F_image = smart_fltr(F_image)
            F_image = mul(F_image, self.fltr)
    
        if do_fft:
            image = real(ifft(F_image))
        else:
            image = F_image
        #image = tf.signal.ifftshift(image, axes=(1, 2))
        return image, loss

    '''
        Ds: [num_frames, 2, nx, nx]
        diversity: [2, nx, nx]
    '''
    def deconvolve(self, Ds, alphas, diversity = None, do_fft = True):
        if len(Ds.size()) == 3:
            Ds = torch.unsqueeze(Ds, 0)
            assert(len(alphas.size()) == 2)
            alphas = torch.unsqueeze(alphas, 0)
        else:
            assert(len(alphas.size()) >= 2)
            if len(alphas.size()) == 2:
                alphas = alphas.unsqueeze(1)
        Ds_F = fft(to_complex(Ds))

        #print("alphas, diversity", alphas.size(), diversity.size())
        Ps, wf = self.calc(alphas, diversity)
        print("Ds_F, Ps", Ds_F.size(), Ps.size())
        
        Ps_conj = conj(Ps)
    
        num = torch.sum(mul(Ds_F, Ps_conj), axis=[0, 1])
        den = torch.sum(mul(Ps, Ps_conj), axis=[0, 1])
        
        image, loss = self.reconstr_(num, den, do_fft)
        print("image", image.size())
        return image, Ps, wf, loss
        
    '''
        Ds: [batch_size, num_frames, 2, nx, nx], where first dimension can be omitted
        alphas: [batch_size, num_frames, jmax], where first dimension can be omitted
        diversity: [2, nx, nx]
    '''
    def mfbd_loss(self, Ds, alphas, diversity, DD_DP_PP=None):
        nx = self.nx
        mode = self.mode
        #alphas = tf.reshape(tf.slice(x, [0], [size]), [self.batch_size, self.num_frames, jmax])

        if len(Ds.size()) == 3:
            Ds = torch.unsqueeze(Ds, 0)
            assert(len(alphas.size()) == 2)
            alphas = torch.unsqueeze(alphas, 0)
        else:
            assert(len(alphas.size()) >= 2)
            if len(alphas.size()) == 2:
                alphas = alphas.unsqueeze(1)
        batch_size = Ds.size()[0]
        assert(self.batch_size == batch_size) # TODO: get rid of class member

        tt = alphas[:, :, 2]
        if tt is not None:
            if self.sum_over_batch:
                tt_sum = torch.sum(tt, axis=[0, 1])
                tt_sum = tt_sum * tt_sum
                tt_sum = torch.sum(tt_sum)
                tt_sum = tt_sum.view(1, 1).repeat(nx, nx)
            else:
                tt_sum = torch.sum(tt, axis=1)
                tt_sum = tt_sum * tt_sum
                tt_sum = torch.sum(tt_sum, axis=1)
                tt_sum = tt_sum.view(batch_size, 1, 1).repeat(1, nx, nx)
        else:
            tt_sum = torch.tensor(0., dtype=torch.float32)
            
        #Ds_F = tf.signal.fft2d(tf.signal.ifftshift(Ds, axes = (2, 3)))
        Ds_F = fft(to_complex(Ds))
            
        Ds_F_conj = conj(Ds_F)
        
        Ps, wf = self.calc(alphas, diversity, normalize = False)
        #Ps = tf.signal.ifftshift(Ps, axes=(2, 3))

        Ps_conj = conj(Ps)
        
        num = torch.sum(mul(Ds_F_conj, Ps), axis=1)
        
        if mode >= 2:
            #if self.sum_over_batch:
            #    num = tf.reshape(num, [1, nx, nx])
            #else:
            num = num.view(batch_size, 1, nx, nx)
            DP_real = real(num)
            DP_imag = imag(num)

            num1 = create_complex(DD_DP_PP[:, 1], DD_DP_PP[:, 2])
            #num1 = tf.complex(tf.slice(DD_DP_PP, [0, 1, 0, 0], [self.batch_size, 1, nx, nx]), 
            #                             tf.slice(DD_DP_PP, [0, 2, 0, 0], [self.batch_size, 1, nx, nx]))
            num = num + num1
        if self.sum_over_batch:
            num = torch.sum(num, axis=0)
        DP_conj = conj(num)
        num = mul(num, conj(num))
        num = real(num)
        
        den = torch.sum(mul(Ps, Ps_conj), axis=1)
        den = real(den)
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

        eps = torch.tensor(1e-10, dtype=torch.float32)
        
        DD = real(torch.sum(mul(Ds_F, Ds_F_conj), axis=1))
        if mode == 1:
            if self.sum_over_batch:
                DD = torch.sum(DD, axis=0)
            return DD - (num + eps)/(den + eps) + self.tt_weight * tt_sum, num, den, DP_conj, Ps, wf
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
            return loss, num, den, DP_conj, DD, DP_real, DP_imag, PP, Ps, wf

    