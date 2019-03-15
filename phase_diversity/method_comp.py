import os
os.environ["OMP_NUM_THREADS"] = "3" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "3" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "3" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "3" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "3" # export NUMEXPR_NUM_THREADS=6

import sys
sys.path.append('../utils')

import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import scipy.special as special
import scipy.signal as signal
import numpy.fft as fft
import matplotlib.pyplot as plt

from scipy.signal import correlate2d as correlate
from multiprocessing import Pool
from functools import partial

import utils
import psf
import psf_basis
import kolmogorov

import plot

num_frames = 10

image = plt.imread('granulation.png')

image = image[0:200,0:200]

nx = np.shape(image)[0]
ny = np.shape(image)[1]

assert(nx == ny)

fimage = fft.fft2(image)
#vals = fft.ifft2(vals)
#vals = fft.ifft2(vals).real
#fimage = np.roll(np.roll(fimage, int(nx/2), axis=0), int(ny/2), axis=1)
    
jmax = 5
arcsec_per_px = 0.055
diameter = 50.0
wavelength = 5250.0
F_D = 1.0
gamma = 1.0


def main():
    
    fried = np.linspace(.1, 2., 1) # Fried parameter (in meters).
    num_realizations = 10    # Number of realizations per fried parameter. 
    
    size = 200		# Total size of the phase-screen from which the final realizations are cropped. 
			# Needs to be kept high enough to include enough power in tip/tilt modes. Currently
			# set at 4 times the diameter of the wavefront. 

    sampling = 1.      # Sampling rate of the data on which the Point Spread Functions are used. A sampling of unity 
                        # denotes critical sampling. 

    
    aperture_func = lambda u: utils.aperture_circ(u, 0.2, 15.0)

    wavefront = kolmogorov.kolmogorov(fried, num_realizations, size, sampling)
    nx = np.shape(wavefront)[2]
    ny = np.shape(wavefront)[3]
    
    x1 = np.linspace(-1., 1., nx)
    x2 = np.linspace(-1., 1., ny)
    pupil_coords = np.dstack(np.meshgrid(x1, x2))
    pupil = aperture_func(pupil_coords)

    my_plot = plot.plot_map(nrows=1, ncols=1)
    my_plot.plot(pupil)
    my_plot.save("pupil.png")
    my_plot.close()

    
    #pupil[int(nx/2),:] = 0
    #pupil[:,int(ny/2)] = 0
    
    #apodize(wavefront, pupil)
    
    
    
    for i in np.arange(0, len(fried)):
        for j in np.arange(0, num_realizations):
            my_plot = plot.plot_map(nrows=1, ncols=1)
            my_plot.plot(wavefront[i,j,:,:])
            my_plot.save("kolmogorov" + str(i) + "_" + str(j) + ".png")
            my_plot.close()

            
            ctf = psf.coh_trans_func(aperture_func, psf.wavefront(wavefront[i,j,:,:]), lambda u: 0.0)
            psf_vals = psf.psf(ctf, nx, ny).calc()

            my_plot = plot.plot_map(nrows=1, ncols=1)
            my_plot.plot(psf_vals)

            my_plot.save("psf" + str(i) + "_" + str(j) + ".png")
            my_plot.close()
            
    
if __name__ == "__main__":
    main()
