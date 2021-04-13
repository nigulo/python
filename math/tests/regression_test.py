import sys
sys.path.append('..')
sys.path.append('../..')
import numpy as np
import unittest
import regression as r

class test_fit_polynom(unittest.TestCase):

    def test(self):
        expected_coefs = [.4, -.3, .2, -.05]
        xs = np.linspace(0, 4, 20)
        powers = np.arange(len(expected_coefs))
        powers = np.reshape(np.repeat(powers, len(xs)), (len(powers), len(xs)))
        ws = np.reshape(np.repeat(expected_coefs, len(xs)), (len(powers), len(xs)))
        ys = np.sum(ws*xs**powers, axis=0)
        
        print(ys)
    
    
        coefs = r.fit_polynom(xs, ys, degree=len(expected_coefs) - 1)        
        np.testing.assert_array_almost_equal(coefs, expected_coefs)
        
        
if __name__ == '__main__':
    unittest.main()