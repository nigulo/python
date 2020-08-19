import sys
sys.path.append('..')
sys.path.append('../../utils')
import numpy as np
import psf_torch as psf_torch
import psf as psf
#import numpy.fft as fft
#import scipy.signal as signal
import unittest
import utils
import plot
import torch
import misc
import matplotlib.pyplot as plt
import numpy.fft as fft

jmax = 50
diameter = 100.0
wavelength = 5250.0

num_frames = 3


def get_params(nx):

    #arcsec_per_px = .03*(wavelength*1e-10)/(diameter*1e-2)*180/np.pi*3600
    arcsec_per_px = .25*(wavelength*1e-10)/(diameter*1e-2)*180/np.pi*3600
    print("arcsec_per_px=", arcsec_per_px)
    defocus = 2.*np.pi*1000
    #defocus = (0., 0.)
    return (arcsec_per_px, defocus)

def create_psf(nx):
    arcsec_per_px, defocus = get_params(nx)
    aperture_func = lambda xs: utils.aperture_circ(xs, coef=15, radius =1.)
    defocus_func = lambda xs: defocus*np.sum(xs*xs, axis=2)

    pa = psf_torch.phase_aberration_torch(jmax, start_index=0)
    ctf = psf_torch.coh_trans_func_torch(aperture_func, pa, defocus_func)
    psf_torch_ = psf_torch.psf_torch(ctf, nx, arcsec_per_px=arcsec_per_px, diameter=diameter, wavelength=wavelength, num_frames=num_frames)
    
    pa = psf.phase_aberration(jmax, start_index=0)
    ctf = psf.coh_trans_func(aperture_func, pa, defocus_func)
    psf_ = psf.psf(ctf, nx, arcsec_per_px, diameter, wavelength)
    return psf_torch_, psf_


class test_complex(unittest.TestCase):
    
    def test(self):
        a = torch.tensor([1. + .5j, 2.-1.j, 3.+2.j])
        a_re = torch.real(a)
        a_im = torch.imag(a)

        b = torch.tensor([-2.5+0.j, 1.+4.j, -0.3-2.2j])
        b_re = torch.real(b)
        b_im = torch.imag(b)
        
        a1 = psf_torch.create_complex(a_re, a_im)
        b1 = psf_torch.create_complex(b_re, b_im)

        np.testing.assert_almost_equal(psf_torch.real(a1).numpy(), torch.real(a).numpy(), 15)
        np.testing.assert_almost_equal(psf_torch.imag(a1).numpy(), torch.imag(a).numpy(), 15)

        c = torch.tensor([1., 2., 3.])
        c1 = psf_torch.to_complex(c)
        c2 = psf_torch.to_complex(c, False)
        np.testing.assert_almost_equal(psf_torch.real(c1).numpy(), c.numpy(), 15)
        np.testing.assert_almost_equal(psf_torch.imag(c1).numpy(), np.zeros_like(c), 15)
        np.testing.assert_almost_equal(psf_torch.real(c2).numpy(), np.zeros_like(c), 15)
        np.testing.assert_almost_equal(psf_torch.imag(c2).numpy(), c.numpy(), 15)


        expected = torch.conj(a)
        result = psf_torch.conj(a1)
        
        np.testing.assert_almost_equal(psf_torch.real(result).numpy(), torch.real(expected).numpy(), 15)
        np.testing.assert_almost_equal(psf_torch.imag(result).numpy(), torch.imag(expected).numpy(), 15)
        
        expected = a*b
        result = psf_torch.mul(a1, b1)
        
        np.testing.assert_almost_equal(psf_torch.real(result).numpy(), torch.real(expected).numpy(), 15)
        np.testing.assert_almost_equal(psf_torch.imag(result).numpy(), torch.imag(expected).numpy(), 15)
        
        expected = a/b
        result = psf_torch.div(a1, b1)
        
        np.testing.assert_almost_equal(psf_torch.real(result).numpy(), torch.real(expected).numpy(), 15)
        np.testing.assert_almost_equal(psf_torch.imag(result).numpy(), torch.imag(expected).numpy(), 15)
        
        

class test_psf_torch(unittest.TestCase):
    
    def test(self):
        nx = 100
        psf_torch_, psf_ = create_psf(nx)

        image = plt.imread("psf_tf_test_input.png")

        image -= np.mean(image)
        image /= np.std(image)
        image = image.astype("float32")
        
        image1 = utils.upsample(image)

        alphas = np.random.normal(size=(jmax*num_frames))*500.
        alphas = alphas.astype("float32")
        alphas = np.reshape(alphas, (num_frames, jmax))
        alphas1 = np.array(np.reshape(alphas, (num_frames, jmax)))
        D_expected = psf_.convolve(image1, alphas = alphas1)
        
        alphas_torch = torch.from_numpy(alphas)
        #self.objs = np.reshape(self.objs, (len(self.objs), -1))
        
        image_torch = torch.from_numpy(image)
        D = psf_torch_.aberrate(image_torch, alphas_torch)
        D_np = D.numpy()

        #D_expected_ds = np.zeros((nx, nx, 2*num_frames))
        for l in np.arange(num_frames):

            D_expected_0 = misc.sample_image(D_expected[l, 0], 0.5)
            D_expected_1 = misc.sample_image(D_expected[l, 1], 0.5)
            #D_expected_ds[:, :, 2*l] = D_expected_0
            #D_expected_ds[:, :, 2*l+1] = D_expected_1
            
            my_plot = plot.plot(nrows=3, ncols=2)
            print("D", D.shape)
            my_plot.colormap(D_np[l, 0, :, :], [0, 0], show_colorbar=True, colorbar_prec=.6)
            my_plot.colormap(D_np[l, 1, :, :], [0, 1])
            my_plot.colormap(D_expected_0, [1, 0])
            my_plot.colormap(D_expected_1, [1, 1])
    
            my_plot.colormap(np.abs(D_expected_0 - D_np[l, 0, :, :]), [2, 0], colorbar_prec=.6)
            my_plot.colormap(np.abs(D_expected_1 - D_np[l, 1, :, :]), [2, 1])
                
            my_plot.save("test_psf_torch_aberrate" + str(l) + ".png")
            my_plot.close()

        #######################################################################

        dat_F = fft.fftshift(fft.fft2(image1), axes=(-2, -1))
        dat_F = psf_.multiply(dat_F, alphas = alphas)

        image_deconv_expected = psf_.deconvolve(dat_F, alphas = alphas1, gamma=1.0, do_fft=True, fft_shift_before=True, ret_all=False, a_est=None, normalize=False)

        print("D", D.size())
        image_deconv, _, _, _ = psf_torch_.deconvolve(D, alphas_torch)

        my_plot = plot.plot(nrows=2, ncols=2)
        my_plot.colormap(image, [0, 0], show_colorbar=True, colorbar_prec=.3)
        my_plot.colormap(image_deconv_expected, [0, 1])
        my_plot.colormap(image, [1, 0])
        my_plot.colormap(image_deconv.numpy()[0], [1, 1])

            
        my_plot.save("test_psf_torch_deconvolve.png")
        my_plot.close()

        #diversity = np.tile(psf_.coh_trans_func.get_diversity(), [num_frames, 1, 1])
        diversity = psf_.coh_trans_func.get_diversity()
        diversity = diversity.astype("float32")
        print("diversity", diversity.shape)
        diversity_torch = torch.from_numpy(diversity)


        psf_torch_.set_diversity = True
        mfbd_loss, num, den, num_conj = psf_torch_.mfbd_loss(D, alphas_torch, diversity_torch)
        mfbd_loss = mfbd_loss.numpy()

        my_plot = plot.plot(nrows=1, ncols=1)
        my_plot.colormap(mfbd_loss, [0], show_colorbar=True, colorbar_prec=.3)
            
        my_plot.save("test_psf_torch_mfbd_loss.png")
        my_plot.close()
        print("mfbd_loss", np.sum(mfbd_loss))

        np.testing.assert_almost_equal(D, D_expected, 15)


if __name__ == '__main__':
    unittest.main()