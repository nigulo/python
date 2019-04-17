import sys
sys.path.append('..')
sys.path.append('../../utils')
import numpy as np
import psf_basis
import psf
import unittest
import plot
import utils
import misc

image = np.array([[0.41960785, 0.38039216, 0.36862746, 0.38039216, 0.40784314, 0.40392157,
  0.38431373, 0.4509804,  0.45882353, 0.5137255 ],
 [0.4117647,  0.38039216, 0.39607844, 0.39215687, 0.34117648, 0.3529412,
  0.35686275, 0.37254903, 0.36862746, 0.38039216],
 [0.36862746, 0.34901962, 0.30980393, 0.3254902,  0.31764707, 0.3372549,
  0.3019608,  0.3254902,  0.33333334, 0.34901962],
 [0.3254902,  0.34509805, 0.3647059,  0.37254903, 0.41568628, 0.36078432,
  0.33333334, 0.32156864, 0.28235295, 0.30980393],
 [0.4392157,  0.4509804,  0.5019608,  0.4627451,  0.4745098,  0.43529412,
  0.36078432, 0.3254902,  0.2901961,  0.2627451 ],
 [0.54901963, 0.5058824,  0.56078434, 0.56078434, 0.5803922,  0.49803922,
  0.3882353,  0.34117648, 0.28627452, 0.30588236],
 [0.6039216,  0.61960787, 0.64705884, 0.61960787, 0.627451,   0.5568628,
  0.42745098, 0.3647059,  0.32941177, 0.32156864],
 [0.6666667,  0.7294118,  0.69803923, 0.7176471,  0.62352943, 0.5803922,
  0.45882353, 0.37254903, 0.3372549,  0.3372549 ],
 [0.6745098,  0.654902,   0.7019608,  0.6862745,  0.6431373,  0.5529412,
  0.42352942, 0.40392157, 0.37254903, 0.39215687],
 [0.6509804,  0.6901961,  0.6509804,  0.6392157,  0.58431375, 0.5294118,
  0.45490196, 0.39607844, 0.36862746, 0.37254903]])

class test_defocus(unittest.TestCase):

    def test(self):

        jmax = 10
        arcsec_per_px = 0.057
        #arcsec_per_px = 0.1
        diameter = 20.0
        wavelength = 5250.0
        defocus = 3.#*np.pi

        for downsample_factor in [0, 1, 2]:
            image1 = image
            if downsample_factor > 0:
                image1 = image[::downsample_factor, ::downsample_factor]
            
            nx_orig = np.shape(image1)[0]
    
            image2 = utils.upsample(image1)
            nx = np.shape(image2)[0]
    
            aperture_func = lambda xs: utils.aperture_circ(xs, diameter, 15.0)
        
    
            ###################################################################
            # Defocus via fourth Zernike term
            pa0 = psf.phase_aberration([defocus])
            ctf0 = psf.coh_trans_func(aperture_func, pa0, lambda xs: 0.)
            psf0 = psf.psf(ctf0, nx_orig, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)
            #pa0.set_alphas([1.])
            D0, D0_d = psf0.convolve(image2)
            
            ###################################################################
            # Defocus defined by explicit function
            pa1 = psf.phase_aberration([])
            defocus_func = lambda xs: defocus*2*np.sum(xs*xs, axis=2)
            ctf1 = psf.coh_trans_func(aperture_func, pa1, defocus_func)
            psf1 = psf.psf(ctf1, nx_orig, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)
            D1, D1_d = psf1.convolve(image2)

            #my_plot = plot.plot_map(nrows=1, ncols=2)
            #my_plot.plot(D0, [0])
            #my_plot.plot(D1_d, [1])
            #my_plot.save("defous_test" + str(downsample_factor) + ".png")
            #my_plot.close()

            np.testing.assert_almost_equal(D0, D0_d)
            np.testing.assert_almost_equal(D0_d, D1_d)

            ###################################################################
            # Defocus in PSF basis

            D1 = misc.normalize(D1)
            D1_d = misc.normalize(D1_d)

            psf_b = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength, defocus = defocus*1.5)
            psf_b.create_basis()
            betas = np.zeros(jmax, dtype='complex')
            D2, D2_d = psf_b.convolve(image2, betas)

            D2 = misc.normalize(D2)
            D2_d = misc.normalize(D2_d)
        
            my_plot = plot.plot_map(nrows=2, ncols=2)
            my_plot.plot(D1, [0, 0])
            my_plot.plot(D2, [0, 1])
            my_plot.plot(D1_d, [1, 0])
            my_plot.plot(D2_d, [1, 1])
            my_plot.save("defous_test" + str(downsample_factor) + ".png")
            my_plot.close()
        

            print(D2/D1)            
            np.testing.assert_almost_equal(D1, D2, 15)
            np.testing.assert_almost_equal(D1_d, D2_d, 15)

        
if __name__ == '__main__':
    unittest.main()
