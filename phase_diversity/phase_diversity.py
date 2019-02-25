import os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6

import sys
sys.path.append('../utils')

import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import scipy.special as special
import scipy.signal as signal
import numpy.fft as fft
import matplotlib.pyplot as plt
from matplotlib import cm
import sampling
import pymc3 as pm

import matplotlib.pyplot as plt
import psf_basis

def reverse_colourmap(cmap, name = 'my_cmap_r'):
     return mpl.colors.LinearSegmentedColormap(name, cm.revcmap(cmap._segmentdata))
my_cmap = reverse_colourmap(plt.get_cmap('binary'))#plt.get_cmap('winter')


num_samples = 100
num_chains = 4


image = plt.imread('granulation.png')

image = image[0:10,0:10]

nx = np.shape(image)[0]
ny = np.shape(image)[1]

assert(nx == ny)

fimage = fft.fft2(image)
#vals = fft.ifft2(vals)
#vals = fft.ifft2(vals).real
#fimage = np.roll(np.roll(fimage, int(nx/2), axis=0), int(ny/2), axis=1)
    
print(np.shape(image))

jmax = 5
arcsec_per_px = 0.055
diameter = 50.0
wavelength = 5250.0
F_D = 1.0
gamma = 1.0


def sample(D, D_d, gamma, psf):

    betas_real = [None] * jmax
    betas_imag = [None] * jmax
    s = sampling.Sampling()
    with s.get_model():
        for i in np.arange(0, psf.jmax):
            betas_real[i] = pm.Normal('beta_r' + str(i), sd=1.0)
            betas_imag[i] = pm.Normal('beta_i' + str(i), sd=1.0)

        a1 = pm.Normal('a1' + str(i), sd=1.0)
        a2 = pm.Normal('a2' + str(i), sd=1.0)
        a3 = pm.Normal('a3' + str(i), sd=1.0)
        a4 = pm.Normal('a4' + str(i), sd=1.0)
        a5 = pm.Normal('a5' + str(i), sd=1.0)
    print(type(betas_real[0]))
    print(type(a1))
    print(np.concatenate((betas_real, betas_imag)))
    print([betas_real[0], betas_real[1]])
    #trace = s.sample(psf.likelihood, np.concatenate((betas_real, betas_imag)), [D, D_d, gamma], num_samples, num_chains, psf.likelihood_grad)
    trace = s.sample(psf.likelihood, [betas_real[0], betas_real[1], betas_real[2], betas_real[3], betas_real[4], betas_imag[0],
        betas_imag[1], betas_imag[2], betas_imag[3], betas_imag[4]], [D, D_d, gamma], num_samples, num_chains, psf.likelihood_grad)

    betas = np.zeros(psf.jmax, dtype='complex')    
    for i in np.arange(0, psf.jmax):
        betas[i] = np.mean(trace['beta_r' + str(i)]) + 1.j*np.mean(trace['beta_i' + str(i)])
    return betas


psf = psf_basis.psf_basis(jmax = jmax, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength, nx = nx, F_D = F_D)
psf.create_basis()

for trial in np.arange(0, 10):
    betas = np.random.normal(size=jmax) + np.random.normal(size=jmax)*1.j
    psf.set_betas(betas)
    
    D, D_d = psf.multiply(fimage, betas)
    
    
    betas = sample(D, D_d, gamma, psf)
    
    fimage_est = psf.get_image(D, D_d, betas, do_fft = False)

    for (fft_image, label) in [(D, ""), (D_d, "_d"), (fimage_est, "_est")]:

        measurement = fft.ifft2(fft_image).real
        measurement = np.roll(np.roll(measurement, int(nx/2), axis=0), int(ny/2), axis=1)
        print(np.shape(measurement))
        print(measurement)
        
        extent=[0., 1., 0., 1.]
        plot_aspect=(extent[1]-extent[0])/(extent[3]-extent[2])#*2/3 
        
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(6, 3)
        ax1.imshow(image[::-1],extent=extent,cmap=my_cmap,origin='lower')
        ax1.set_aspect(aspect=plot_aspect)
        
        ax2.imshow(measurement[::-1],extent=extent,cmap=my_cmap,origin='lower')
        ax2.set_aspect(aspect=plot_aspect)
        
        fig.savefig("measurement" + str(trial) + label + ".png")
        plt.close(fig)
