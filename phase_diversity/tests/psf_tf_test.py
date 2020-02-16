import sys
sys.path.append('..')
import numpy as np
import psf_tf as psf_tf
import numpy.fft as fft
import scipy.signal as signal
import unittest
import utils
sys.path.append('../../utils')
import plot
import misc


jmax = 50
diameter = 100.0
wavelength = 5250.0

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
    return psf_tf(ctf, nx//2, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)

class test_aberrate(unittest.TestCase):
    
    def test(self):
        psf_tf_ = create_psf(100)
        psf_tf_.aberrate()

        np.testing.assert_almost_equal(result, expected, 15)
        
