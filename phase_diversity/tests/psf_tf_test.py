import sys
sys.path.append('..')
sys.path.append('../../utils')
import numpy as np
import psf_tf as psf_tf
import psf as psf
#import numpy.fft as fft
#import scipy.signal as signal
import unittest
import utils
import plot
import tensorflow as tf
import misc
import matplotlib.pyplot as plt

jmax = 50
diameter = 100.0
wavelength = 5250.0

image = plt.imread("psf_tf_test_input.png")


def get_params(nx):

    #arcsec_per_px = .03*(wavelength*1e-10)/(diameter*1e-2)*180/np.pi*3600
    arcsec_per_px = .25*(wavelength*1e-10)/(diameter*1e-2)*180/np.pi*3600
    print("arcsec_per_px=", arcsec_per_px)
    defocus = 2.*np.pi*100
    #defocus = (0., 0.)
    return (arcsec_per_px, defocus)

def create_psf(nx):
    arcsec_per_px, defocus = get_params(nx)
    aperture_func = lambda xs: utils.aperture_circ(xs, coef=15, radius =1.)
    defocus_func = lambda xs: defocus*np.sum(xs*xs, axis=2)

    pa = psf_tf.phase_aberration_tf(jmax, start_index=0)
    ctf = psf_tf.coh_trans_func_tf(aperture_func, pa, defocus_func)
    psf_tf_ = psf_tf.psf_tf(ctf, nx, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)
    
    
    pa = psf.phase_aberration(jmax, start_index=0)
    ctf = psf.coh_trans_func(aperture_func, pa, defocus_func)
    psf_ = psf.psf(ctf, nx, arcsec_per_px, diameter, wavelength)
    return psf_tf_, psf_

class test_aberrate(unittest.TestCase):
    
    def test(self):
        nx = 100
        psf_tf_, psf_ = create_psf(nx)
        image1 = utils.upsample(image)

        alphas = np.random.normal(size=(jmax))*500.
        D_expected = psf_.convolve(image1, alphas = np.array([alphas]))[0]
        D_expected_0 = misc.sample_image(D_expected[0], 0.5)
        D_expected_1 = misc.sample_image(D_expected[1], 0.5)
        
        alphas_tf = tf.constant(alphas, dtype='float32')
        #self.objs = np.reshape(self.objs, (len(self.objs), -1))
        
        image_tf = tf.constant(image.flatten(), dtype='float32')
        D = psf_tf_.aberrate(tf.concat((alphas_tf, image_tf), 0))

        
        my_plot = plot.plot(nrows=3, ncols=2)
        print(D.shape)
        my_plot.colormap(D[0,:,:,0], [0, 0], show_colorbar=True, colorbar_prec=.3)
        my_plot.colormap(D[0,:,:,1], [0, 1])
        my_plot.colormap(D_expected_0, [1, 0])
        my_plot.colormap(D_expected_1, [1,1])

        my_plot.colormap(D_expected_0 - D[0,:,:,0], [2, 0], colorbar_prec=.3)
        my_plot.colormap(D_expected_1 - D[0,:,:,1], [2, 1])
            
        my_plot.save("test_aberrate.png")
        my_plot.close()

        np.testing.assert_almost_equal(D, D_expected, 15)


if __name__ == '__main__':
    unittest.main()