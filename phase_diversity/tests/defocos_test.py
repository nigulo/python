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

image = np.array([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0. ],
                  [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0. ],
                  [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0. ],
                  [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0. ],
                  [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0. ],
                  [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. ],
                  [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0. ],
                  [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0. ],
                  [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0. ],
                  [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0. ],
                  [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0. ]
                  ])

class test_defocus(unittest.TestCase):

    def test(self):

        jmax = 10
        arcsec_per_px = 0.057
        #arcsec_per_px = 0.1
        diameter = 20.0
        wavelength = 5250.0
        defocus = 2.#*np.pi

        for downsample_factor in [0, 1, 2]:
            image1 = image
            if downsample_factor > 0:
                image1 = image[::downsample_factor, ::downsample_factor]
            
            nx_orig = np.shape(image1)[0]
    
            image2 = utils.upsample(image1)
            nx = np.shape(image2)[0]
    
            aperture_func = lambda xs: utils.aperture_circ(xs, r=1.0527, coef=15.)
            #aperture_func = lambda xs: utils.aperture_circ(xs, r=.1, coef=100.)
        
    
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

            psf_b = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = arcsec_per_px*2, diameter = diameter, wavelength = wavelength, defocus = defocus*2.21075039)
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
        

            np.testing.assert_almost_equal(D1, D2, 15)
            np.testing.assert_almost_equal(D1_d, D2_d, 15)

        
if __name__ == '__main__':
    unittest.main()
