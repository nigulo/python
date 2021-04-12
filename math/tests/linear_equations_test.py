import sys
sys.path.append('..')
sys.path.append('../..')
import numpy as np
import unittest
import linear_equations as le

class test_solve_gauss(unittest.TestCase):

    def test(self):
        '''
        System of equations:
        2x +  y +  3z = 1
        2x + 6y +  8z = 3
        6x + 8y + 18z = 5
        Solution:
        (x, y, z) = (3/10, 2/5, 0)
        '''
    m = np.array ([
        [2, 1, 3, 1],
        [2, 6, 8, 3],
        [6, 8, 18, 5]
        ], dtype=float)

    expected_result = np.array([3/10, 2/5, 0])
    result = le.solve_gauss(m)
    np.testing.assert_array_almost_equal(result, expected_result)
        
        
if __name__ == '__main__':
    unittest.main()