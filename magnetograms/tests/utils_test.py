import sys
sys.path.append('..')
import numpy as np
import unittest
import utils


class test_get_closest(unittest.TestCase):

    def test(self):
        xs = np.array([0.0, 1.0, 2.0, 3.0])
        ys = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        #(xs_c, ys_c), (indices_x, indices_y) = utils.get_closest(xs, ys, 0.0, 0.0)
        (xs_c, ys_c), (indices_x, indices_y) = utils.get_closest([xs, ys], np.array([0.0, 0.0]))
        np.testing.assert_equal(xs_c, np.array([0.,  1.]))
        np.testing.assert_equal(ys_c, np.array([0.,  -0.5]))
        np.testing.assert_equal(indices_x, np.array([0,  1]))
        np.testing.assert_equal(indices_y, np.array([2,  1]))

        #(xs_c, ys_c), (indices_x, indices_y) = utils.get_closest(xs, ys, 3.0, 1.0)
        (xs_c, ys_c), (indices_x, indices_y) = utils.get_closest([xs, ys], np.array([3.0, 1.0]))
        np.testing.assert_equal(xs_c, np.array([3.,  2.]))
        np.testing.assert_equal(ys_c, np.array([1.,  0.5]))
        np.testing.assert_equal(indices_x, np.array([3,  2]))
        np.testing.assert_equal(indices_y, np.array([4,  3]))

        #(xs_c, ys_c), (indices_x, indices_y) = utils.get_closest(xs, ys, 1.6, -0.4)
        (xs_c, ys_c), (indices_x, indices_y) = utils.get_closest([xs, ys], np.array([1.6, -0.4]))
        np.testing.assert_equal(xs_c, np.array([2.,  1.]))
        np.testing.assert_equal(ys_c, np.array([-0.5,  0.0]))
        np.testing.assert_equal(indices_x, np.array([2,  1]))
        np.testing.assert_equal(indices_y, np.array([1,  2]))
        

class test_bilinear_interp(unittest.TestCase):

    def test(self):
        xs = np.array([0.0, 1.0])
        ys = np.array([-0.5, 0.0])
        coefs = utils.bilinear_interp([xs, ys], np.array([0.0, 0.0]))
        np.testing.assert_equal(coefs, np.array([0., 0., 1., 0.]))

        xs = np.array([2.0, 3.0])
        ys = np.array([0.5, 1.0])
        coefs = utils.bilinear_interp([xs, ys], np.array([3.0, 1.0]))
        np.testing.assert_equal(coefs, np.array([0., 0., 0., 1.]))

        xs = np.array([1.0, 2.0])
        ys = np.array([-0.5, 0.0])
        coefs = utils.bilinear_interp([xs, ys], np.array([1.6, -0.4]))
        np.testing.assert_array_almost_equal(coefs, np.array([0.32, 0.48, 0.08, 0.12]))
        
        xs = np.array([1.0, 2.0, 3.0])
        ys = np.array([-0.5, 0.0, 0.5])
        zs = np.array([0., .2])
        coefs = utils.bilinear_interp([xs, ys, zs], np.array([1.6, -0.4, 0.1]))
        np.testing.assert_array_almost_equal(coefs, np.array(
                [2.540160e-03, 2.286144e-02, 4.665600e-04, 6.350400e-04, 5.715360e-03,
                 1.166400e-04, 3.136000e-05, 2.822400e-04, 5.760000e-06, 2.540160e-03,
                 2.286144e-02, 4.665600e-04, 6.350400e-04, 5.715360e-03, 1.166400e-04,
                 3.136000e-05, 2.822400e-04, 5.760000e-06]))
        

class test_calc_W(unittest.TestCase):
    
    def test(self):
        u_mesh = [np.array([[0., 1.], [0., 1.]]), np.array([[-0.25, -0.25], [0.25, 0.25]])]
        us = np.array([[0., -0.25], [1., -0.25], [0., 0.25], [1., 0.25]])

        xys = np.array([[0., -0.25], [1., -0.25], 
                        [0., 0.25], [1., 0.25]])
        W = utils.calc_W(u_mesh, us, xys)
        np.testing.assert_array_almost_equal(W, np.array(
            [[1.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
             [0.,   1.,   0.,   0.,   0.,   0.,   0.,   0.],
             [0.,   0.,   1.,   0.,   0.,   0.,   0.,   0.],
             [0.,   0.,   0.,   1.,   0.,   0.,   0.,   0.],
             [0.,   0.,   0.,   0.,   1.,   0.,   0.,   0.],
             [0.,   0.,   0.,   0.,   0.,   1.,   0.,   0.],
             [0.,   0.,   0.,   0.,   0.,   0.,   1.,   0.],
             [0.,   0.,   0.,   0.,   0.,   0.,   0.,   1.]]))

        xys = np.array([[0., -0.25], [0.5, -0.25], [1., -0.25], 
                        [0., 0.], [0.5, 0.], [1., 0.], 
                        [0., 0.25], [0.5, 0.25], [1., 0.25]])
        W = utils.calc_W(u_mesh, us, xys)
        np.testing.assert_array_almost_equal(W, np.array(
            [[1.,   0.,   0.,   0.,   0.,   0.,   0.,   0.  ],
             [0.,   1.,   0.,   0.,   0.,   0.,   0.,   0.  ],
             [0.5,  0.,   0.5,  0.,   0.,   0.,   0.,   0.  ],
             [0.,   0.5,  0.,   0.5,  0.,   0.,   0.,   0.  ],
             [0.,   0.,   1.,   0.,   0.,   0.,   0.,   0.  ],
             [0.,   0.,   0.,   1.,   0.,   0.,   0.,   0.  ],
             [0.5,  0.,   0.,   0.,   0.5,  0.,   0.,   0.  ],
             [0.,   0.5,  0.,   0.,   0.,   0.5,  0.,   0.  ],
             [0.25, 0.,   0.25, 0.,   0.25, 0.,   0.25, 0.  ],
             [0.,   0.25, 0.,   0.25, 0.,   0.25, 0.,   0.25],
             [0.,   0.,   0.5,  0.,   0.,   0.,   0.5,  0.  ],
             [0.,   0.,   0.,   0.5,  0.,   0.,   0.,   0.5 ],
             [0.,   0.,   0.,   0.,   1.,   0.,   0.,   0.  ],
             [0.,   0.,   0.,   0.,   0.,   1.,   0.,   0.  ],
             [0.,   0.,   0.,   0.,   0.5,  0.,   0.5,  0.  ],
             [0.,   0.,   0.,   0.,   0.,   0.5,  0.,   0.5 ],
             [0.,   0.,   0.,   0.,   0.,   0.,   1.,   0.  ],
             [0.,   0.,   0.,   0.,   0.,   0.,   0.,   1.  ]]))
        
if __name__ == '__main__':
    unittest.main()        

