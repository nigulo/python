import sys
sys.path.append('../')
import matplotlib as mpl

mpl.use('Agg')
mpl.rcParams['figure.figsize'] = (20, 30)
import pickle
import numpy as np
#import pylab as plt
#import pandas as pd
import GPR_div_free as GPR_div_free
import scipy.misc
import numpy.random as random
import scipy.interpolate as interp
import scipy.sparse.linalg as sparse
from matplotlib import cm
import unittest
import utils


#import os
#import os.path
#from scipy.stats import gaussian_kde
#from sklearn.cluster import KMeans
import numpy.linalg as la
import matplotlib.pyplot as plt

num_iters = 50
num_chains = 1
inference = False

eps = 0.001
learning_rate = 1.0
max_num_tries = 20


if len(sys.argv) > 3:
    num_iters = int(sys.argv[3])
if len(sys.argv) > 4:
    num_chains = int(sys.argv[4])

n_jobs = num_chains

model = pickle.load(open('model.pkl', 'rb'))

n1 = 10
n2 = 10
n = n1*n2
x1_range = 1.0
x2_range = 1.0
x1 = np.linspace(0, x1_range, n1)
x2 = np.linspace(0, x2_range, n2)
x_mesh = np.meshgrid(x1, x2)
x = np.dstack(x_mesh).reshape(-1, 2)
x_flat = np.reshape(x, (2*n, -1))

m1 = 7
m2 = 7
m = m1 * m2
u1 = np.linspace(0, x1_range, m1)
u2 = np.linspace(0, x2_range, m2)
u_mesh = np.meshgrid(u1, u2)
u = np.dstack(u_mesh).reshape(-1, 2)

print "u_mesh=", u_mesh
print "u=", u


class test_get_closest(unittest.TestCase):

    def test(self):
        xs = np.array([0.0, 1.0, 2.0, 3.0])
        ys = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        (xs_c, ys_c), (indices_x, indices_y) = utils.get_closest(xs, ys, 0.0, 0.0)
        np.testing.assert_equal(xs_c, np.array([0.,  1.]))
        np.testing.assert_equal(ys_c, np.array([0.,  -0.5]))
        np.testing.assert_equal(indices_x, np.array([0,  1]))
        np.testing.assert_equal(indices_y, np.array([2,  1]))

        (xs_c, ys_c), (indices_x, indices_y) = utils.get_closest(xs, ys, 3.0, 1.0)
        np.testing.assert_equal(xs_c, np.array([3.,  2.]))
        np.testing.assert_equal(ys_c, np.array([1.,  0.5]))
        np.testing.assert_equal(indices_x, np.array([3,  2]))
        np.testing.assert_equal(indices_y, np.array([4,  3]))

        (xs_c, ys_c), (indices_x, indices_y) = utils.get_closest(xs, ys, 1.6, -0.4)
        np.testing.assert_equal(xs_c, np.array([2.,  1.]))
        np.testing.assert_equal(ys_c, np.array([-0.5,  0.0]))
        np.testing.assert_equal(indices_x, np.array([2,  1]))
        np.testing.assert_equal(indices_y, np.array([1,  2]))

class test_bilinear_interp(unittest.TestCase):

    def test(self):
        xs = np.array([0.0, 1.0])
        ys = np.array([-0.5, 0.0])
        coefs = utils.bilinear_interp(xs, ys, 0.0, 0.0)
        np.testing.assert_equal(coefs, np.array([0., 0., 1., 0.]))

        xs = np.array([2.0, 3.0])
        ys = np.array([0.5, 1.0])
        coefs = utils.bilinear_interp(xs, ys, 3.0, 1.0)
        np.testing.assert_equal(coefs, np.array([0., 0., 0., 1.]))

        xs = np.array([1.0, 2.0])
        ys = np.array([-0.5, 0.0])
        coefs = utils.bilinear_interp(xs, ys, 1.6, -0.4)
        np.testing.assert_array_almost_equal(coefs, np.array([0.32, 0.48, 0.08, 0.12]))
        
if __name__ == '__main__':
    unittest.main()        

