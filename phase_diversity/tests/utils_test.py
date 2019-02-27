import sys
sys.path.append('..')
import numpy as np
import psf
import unittest

class test_aperture_circ(unittest.TestCase):

    def test(self):
        
        coords = np.array([[0.0, 0.0],
                          [0.0, 0.5],
                          [0.0, 0.9],
                          [0.0, 1.0],
                          [0.0, 1.1],
                          [0.5, 0.5],
                          [0.9/np.sqrt(2), 0.9/np.sqrt(2)],
                          [1.0/np.sqrt(2), 1.0/np.sqrt(2)],
                          [1.1/np.sqrt(2), 1.1/np.sqrt(2)]])
        
        expected = np.array([1.0,
                             0.9997965239912775,
                             0.7602499389065231,
                             0.5,
                             0.23975006109347657,
                             0.980823770312415,
                             0.7602499389065231,
                             0.5,
                             0.23975006109347657])
    
        for i in np.arange(0, len(expected)):
            np.testing.assert_almost_equal(psf.aperture_circ(coords[i]), expected[i], 10)

        for x in np.linspace(-1, 1, 100):
            for y in np.linspace(-1, 1, 100):
                np.testing.assert_almost_equal(psf.aperture_circ(np.array([x, y]), r=1.0, coef=1e5), psf.aperture_circ(np.array([x, y]), r=1.0, coef=0.0), 10)
                
                
        # Test the whole array at once
        np.testing.assert_almost_equal(psf.aperture_circ(coords), expected, 10)
        
        xs = np.linspace(-1.0, 1.0, 100)
        ys = np.linspace(-1.0, 1.0, 100)
        us = np.transpose([np.tile(xs, 100), np.repeat(ys, 100)])
        np.testing.assert_almost_equal(psf.aperture_circ(us, r=1.0, coef=1e5), psf.aperture_circ(us, r=1.0, coef=0.0), 10)

        
if __name__ == '__main__':
    unittest.main()
