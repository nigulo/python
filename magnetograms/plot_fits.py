import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['figure.figsize'] = (20, 30)
import sys
sys.path.append('../utils')
import numpy as np
from astropy.io import fits
import plot


hdul = fits.open('pi-ambiguity-test/resME0_spot.fits')
dat = hdul[0].data[:,::4,::4]
b = dat[0]
theta = dat[1]
phi = dat[2]

n1 = b.shape[0]
n2 = b.shape[1]

inference = True
sample_or_optimize = False
num_samples = 1
num_chains = 4
inference_after_iter = 20

bz = b*np.cos(theta)
bz = bz
bxy = b*np.sin(theta)
bx = bxy*np.cos(phi)
by = bxy*np.sin(phi)

my_plot = plot.plot_map(nrows=1, ncols=3)
my_plot.set_color_map('bwr')

my_plot.plot(np.reshape(bx, (n1, n2)), [0])
my_plot.plot(np.reshape(by, (n1, n2)), [1])
my_plot.plot(np.reshape(np.arctan2(by, bx), (n1, n2)), [2])

my_plot.save("results.png")

my_plot.close()
