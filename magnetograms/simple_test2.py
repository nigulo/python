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
import cov_div_free as cov_div_free
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
    

bx = y[:, :, 4, 0]
by = y[:, :, 4, 1]
bz = y[:, :, 4, 2]
b = np.sqrt(bx**2 + by**2 + bz**2)
phi = np.arctan2(by, bx)
theta = np.arccos((bz+1e-10)/(b+1e-10))


n1 = b.shape[0]
n2 = b.shape[1]
x1_range = 1.0
x2_range = 1.0
n = n1*n2
print(n1, n2)

x1 = np.linspace(0, x1_range, n1)
x2 = np.linspace(0, x2_range, n2)
x_mesh = np.meshgrid(x2, x1)
x = np.dstack(x_mesh).reshape(-1, 2)
x_flat = np.reshape(x, (2*n, -1))

m1 = 10
m2 = 10

m = m1 * m2
u1 = np.linspace(0, x1_range, m1)
u2 = np.linspace(0, x2_range, m2)
u_mesh = np.meshgrid(u1, u2)
u = np.dstack(u_mesh).reshape(-1, 2)
        
myplot = plot.plot(nrows=1, ncols=1)
myplot.set_color_map('bwr')
myplot.colormap(bz)
myplot.vectors(x_mesh[0], x_mesh[1], bx, by, [], units='width', color = 'k')
myplot.save("simple_test2_input.png")
myplot.close()

idbz = infer_dbz.infer_dbz(bx, by, bz)
dbz = idbz.calc()


myplot = plot.plot(nrows=1, ncols=1)
myplot.set_color_map('bwr')
myplot.colormap(dbz)
myplot.vectors(x_mesh[0], x_mesh[1], bx, by, [], units='width', color = 'k')
myplot.save("simple_test2_output.png")
myplot.close()


