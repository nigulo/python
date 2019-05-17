import os
os.environ["OMP_NUM_THREADS"] = "16" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "16" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "16" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "16" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "16" # export NUMEXPR_NUM_THREADS=6

import sys
sys.path.append('../utils')
import matplotlib as mpl

mpl.use('Agg')
mpl.rcParams['figure.figsize'] = (20, 30)
#import pickle
import numpy as np
#import pylab as plt
#import pandas as pd
import cov_sq_exp as cov_sq_exp
import scipy.misc
import numpy.random as random
import scipy.sparse.linalg as sparse
import scipy.stats as stats
import scipy.special as special
import utils
import plot
import pymc3 as pm
#import os
#import os.path
#from scipy.stats import gaussian_kde
#from sklearn.cluster import KMeans
import numpy.linalg as la
import matplotlib.pyplot as plt
import sampling
import kiss_gp
from scipy.integrate import simps

import time
import os.path
from astropy.io import fits
from scipy.io import readsav
import scipy.signal as signal
import pickle
import infer_dbz

state_file = None#state.pkl"
if len(sys.argv) > 1:
    state_file = sys.argv[1]


if state_file is None:
    state_file = 'data3d.pkl'
if os.path.isfile(state_file):
    y = pickle.load(open(state_file, 'rb'))
else:
    raise Exception("State file not found")
    
'''
bx = np.zeros((y.shape[0], y.shape[1]))
by = np.zeros((y.shape[0], y.shape[1]))
bz = np.zeros((y.shape[0], y.shape[1]))

for i in np.arange(0, y.shape[0]):
    for j in np.arange(0, y.shape[1]):
        z_index = np.random.choice(np.arange(2, 4), size=1)
        bx[i, j] = y[i, j, z_index, 0]
        by[i, j] = y[i, j, z_index, 1]
        bz[i, j] = y[i, j, z_index, 2]
'''
        
bx = np.array(y[:, :, 3, 0])
by = np.array(y[:, :, 3, 1])
bz = np.array(y[:, :, 3, 2])

#bx += np.random.normal(loc=0., scale = np.std(bx)*.5, size = bx.shape)
#by += np.random.normal(loc=0., scale = np.std(by)*.5, size = by.shape)
#bz += np.random.normal(loc=0., scale = np.std(bz)*.5, size = bz.shape)

n1 = bz.shape[0]
n2 = bz.shape[1]
n = n1*n2

x1_range = 1.0
x2_range = 1.0

x1 = np.linspace(0, x1_range, n1)
x2 = np.linspace(0, x2_range, n2)
x_mesh = np.meshgrid(x2, x1)
x = np.dstack(x_mesh).reshape(-1, 2)
x_flat = np.reshape(x, (2*n, -1))
x = x.reshape(-1, 2)

###############################################################################
# Add correlated noise to bz

sig_var_train = np.var(bz)*.2
length_scale_train = .2
noise_var_train = 0.0001
mean_train = 0.

gp_train = cov_sq_exp.cov_sq_exp(sig_var_train, length_scale_train, noise_var_train, dim_out=1)
K = gp_train.calc_cov(x, x, True)

for i in np.arange(0, n1):
    for j in np.arange(0, n2):
        assert(K[i, j]==K[j, i])

L = la.cholesky(K)
s = np.random.normal(0.0, 1.0, n)

bz_noise = np.repeat(mean_train, n) + np.dot(L, s)
bz_noise = np.reshape(bz_noise, (n1, n2))
bz += bz_noise
###############################################################################
        
b = np.sqrt(bx**2 + by**2 + bz**2)
phi = np.arctan2(by, bx)
theta = np.arccos((bz+1e-10)/(b+1e-10))


print(n1, n2)


myplot = plot.plot(nrows=1, ncols=1)
myplot.set_color_map('bwr')
myplot.colormap(bz)
myplot.vectors(x_mesh[0], x_mesh[1], bx, by, [], units='width', color = 'k')
myplot.save("simple_test2_input.png")
myplot.close()

idbz = infer_dbz.infer_dbz(bx, by, bz)
bz1 = idbz.calc()


myplot = plot.plot(nrows=1, ncols=1)
myplot.set_color_map('bwr')
#myplot.colormap((bz[:-1, :-1] - bz1.reshape((n1-1, n2-1))))
myplot.colormap((bz1.reshape((n1-1, n2-1))))
myplot.vectors(x_mesh[0], x_mesh[1], bx, by, [], units='width', color = 'k')
myplot.save("simple_test2_output.png")
myplot.close()

for i in np.arange(0, y.shape[2]):
    test_plot = plot.plot(nrows=1, ncols=1)
    test_plot.set_color_map('bwr')
    
    test_plot.colormap(y[:, :, i, 2])
    test_plot.vectors(x_mesh[0], x_mesh[1], y[:, :, i, 0], y[:, :, i, 1], [], units='width', color = 'k')
    test_plot.save("simple_test2_layer" + str(i) +".png")
    test_plot.close()

