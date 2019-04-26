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
        defocus = 2.

        counter = 0
        #for arcsec_per_px in[0.057, 0.02579]:
        for defocus in[1., 2., 3.]:

            image1 = image
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

            #my_plot = plot.plot(nrows=1, ncols=2)
            #my_plot.colormap(D0, [0])
            #my_plot.colormap(D1_d, [1])
            #my_plot.save("defous_test" + str(downsample_factor) + ".png")
            #my_plot.close()

            np.testing.assert_almost_equal(D0, D0_d)
            np.testing.assert_almost_equal(D0_d, D1_d)

            ###################################################################
            # Defocus in PSF basis

            D1 = misc.normalize(D1)
            D1_d = misc.normalize(D1_d)

            psf_b = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = arcsec_per_px*2, diameter = diameter, wavelength = wavelength, defocus = defocus)#*2.21075039)
            psf_b.create_basis()
            betas = np.zeros(jmax, dtype='complex')
            D2, D2_d = psf_b.convolve(image2, betas)
            D2 = np.roll(np.roll(D2, 1, axis=0), 1, axis=1)
            D2_d = np.roll(np.roll(D2_d, 1, axis=0), 1, axis=1)

            D2 = misc.normalize(D2)
            D2_d = misc.normalize(D2_d)
        
            my_plot = plot.plot(nrows=2, ncols=2)
            my_plot.colormap(D1, [0, 0])
            my_plot.colormap(D2, [0, 1])
            my_plot.colormap(D1_d, [1, 0])
            my_plot.colormap(D2_d, [1, 1])
            my_plot.save("defous_test" + str(counter) + ".png")
            my_plot.close()
        
            np.testing.assert_almost_equal(D1, D2, 2)
            np.testing.assert_almost_equal(D1_d, D2_d, 1)
            
            counter += 1

        
if __name__ == '__main__':
    unittest.main()
