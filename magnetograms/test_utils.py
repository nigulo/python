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
        np.testing.assert_equal(xs_c , np.array([ 0.,  1.]))
        np.testing.assert_equal(ys_c , np.array([ 0.,  -0.5]))
        np.testing.assert_equal(indices_x, np.array([0,  1]))
        np.testing.assert_equal(indices_y, np.array([2,  1]))

        (xs_c, ys_c), (indices_x, indices_y) = utils.get_closest(xs, ys, 3.0, 1.0)
        np.testing.assert_equal(xs_c , np.array([ 3.,  2.]))
        np.testing.assert_equal(ys_c , np.array([ 1.,  0.5]))
        np.testing.assert_equal(indices_x, np.array([3,  2]))
        np.testing.assert_equal(indices_y, np.array([4,  3]))

        (xs_c, ys_c), (indices_x, indices_y) = utils.get_closest(xs, ys, 1.6, -0.4)
        np.testing.assert_equal(xs_c , np.array([ 2.,  1.]))
        np.testing.assert_equal(ys_c , np.array([ -0.5,  0.0]))
        np.testing.assert_equal(indices_x, np.array([2,  1]))
        np.testing.assert_equal(indices_y, np.array([1,  2]))
        
if __name__ == '__main__':
    unittest.main()        

print x_mesh

sig_var_train = 0.2
length_scale_train = 0.1
noise_var_train = 0.000001
mean_train = 0.0

gp_train = GPR_div_free.GPR_div_free(sig_var_train, length_scale_train, noise_var_train)
K = gp_train.calc_cov(x, x, True)
#U = gp_train.calc_cov(u, u, data_or_test=True)
#W = calc_W(u_mesh, u, x)#np.zeros((len(x1)*len(x2)*2, len(u1)*len(u2)*2))
#K = np.dot(W.T, np.dot(U, W))

print "SIIN"
for i in np.arange(0, n1):
    for j in np.arange(0, n2):
        assert(K[i, j]==K[j, i])

L = la.cholesky(K)
s = np.random.normal(0.0, 1.0, 2*n)

y = np.repeat(mean_train, 2*n) + np.dot(L, s)

y = np.reshape(y, (n, 2))
y_orig = np.array(y)
print y_orig

fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(nrows=2, ncols=2, sharex=True)
fig.set_size_inches(12, 12)

extent=[x1.min(), x1.max(), x2.min(), x2.max()]

def reverse_colourmap(cmap, name = 'my_cmap_r'):
     return mpl.colors.LinearSegmentedColormap(name, cm.revcmap(cmap._segmentdata))

my_cmap = reverse_colourmap(plt.get_cmap('gnuplot'))

im11 = ax11.imshow(np.reshape(K[0::2,0::2], (len(x1)*len(x2), len(x1)*len(x2))),extent=extent,cmap=my_cmap,origin='lower', vmin=np.min(K), vmax=np.max(K))
im12 = ax12.imshow(np.reshape(K[0::2,1::2], (len(x1)*len(x2), len(x1)*len(x2))),extent=extent,cmap=my_cmap,origin='lower', vmin=np.min(K), vmax=np.max(K))
im21 = ax21.imshow(np.reshape(K[1::2,0::2], (len(x1)*len(x2), len(x1)*len(x2))),extent=extent,cmap=my_cmap,origin='lower', vmin=np.min(K), vmax=np.max(K))
im22 = ax22.imshow(np.reshape(K[1::2,1::2], (len(x1)*len(x2), len(x1)*len(x2))),extent=extent,cmap=my_cmap,origin='lower', vmin=np.min(K), vmax=np.max(K))

fig.savefig("cov_true.png")

fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(nrows=2, ncols=2, sharex=True)
fig.set_size_inches(12, 12)

U = gp_train.calc_cov(u, u, data_or_test=True)
W = calc_W(u_mesh, u, x)#np.zeros((len(x1)*len(x2)*2, len(u1)*len(u2)*2))

for i in np.arange(0, np.shape(W)[0]):
    W_row = W[i, 0::2]
    for j in np.arange(0, np.shape(W_row)[0]):
        if W_row[j] != 0:
            print j
    print "SUM:", i, np.sum(W_row)

#K1 = np.dot(W, np.dot(U, W.T))

#im11 = ax11.imshow(np.reshape(K1[0::2,0::2], (len(x1)*len(x2), len(x1)*len(x2))),extent=extent,cmap=my_cmap,origin='lower', vmin=np.min(K1), vmax=np.max(K1))
#im12 = ax12.imshow(np.reshape(K1[0::2,1::2], (len(x1)*len(x2), len(x1)*len(x2))),extent=extent,cmap=my_cmap,origin='lower', vmin=np.min(K1), vmax=np.max(K1))
#im21 = ax21.imshow(np.reshape(K1[1::2,0::2], (len(x1)*len(x2), len(x1)*len(x2))),extent=extent,cmap=my_cmap,origin='lower', vmin=np.min(K1), vmax=np.max(K1))
#im22 = ax22.imshow(np.reshape(K1[1::2,1::2], (len(x1)*len(x2), len(x1)*len(x2))),extent=extent,cmap=my_cmap,origin='lower', vmin=np.min(K1), vmax=np.max(K1))

#extent=[u1.min(), u1.max(), u2.min(), u2.max()]

#im11 = ax11.imshow(np.reshape(U[0,0::2], (len(u1), len(u2))),extent=extent,cmap=my_cmap,origin='lower', vmin=np.min(U), vmax=np.max(U))
#im12 = ax12.imshow(np.reshape(U[0,1::2], (len(u1), len(u2))),extent=extent,cmap=my_cmap,origin='lower', vmin=np.min(U), vmax=np.max(U))
#im21 = ax21.imshow(np.reshape(U[1,0::2], (len(u1), len(u2))),extent=extent,cmap=my_cmap,origin='lower', vmin=np.min(U), vmax=np.max(U))
#im22 = ax22.imshow(np.reshape(U[1,1::2], (len(u1), len(u2))),extent=extent,cmap=my_cmap,origin='lower', vmin=np.min(U), vmax=np.max(U))

extent=[u1.min(), u1.max(), u2.min(), u2.max()]

#print "SHAPE:", np.shape(W[0,0::2]), len(x1), len(u2)

im11 = ax11.imshow(np.reshape(W[0::2,0::2], (len(u1)*len(u2), len(x1)*len(x2))),extent=extent,cmap=my_cmap,origin='lower', vmin=np.min(W), vmax=np.max(W))
im12 = ax12.imshow(np.reshape(W[0::2,1::2], (len(u1)*len(u2), len(x1)*len(x2))),extent=extent,cmap=my_cmap,origin='lower', vmin=np.min(W), vmax=np.max(W))
im21 = ax21.imshow(np.reshape(W[1::2,0::2], (len(u1)*len(u2), len(x1)*len(x2))),extent=extent,cmap=my_cmap,origin='lower', vmin=np.min(W), vmax=np.max(W))
im22 = ax22.imshow(np.reshape(W[1::2,1::2], (len(u1)*len(u2), len(x1)*len(x2))),extent=extent,cmap=my_cmap,origin='lower', vmin=np.min(W), vmax=np.max(W))

fig.savefig("cov_approx.png")




