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

sys.path.append('../utils')
import plot

def init_amplitude(fried, size, pupsize):

    if size % 2 == 0:
        raise ValueError('Dimension needs to be odd')

    x = np.linspace(-(size-1)/2, (size-1)/2, size)
    y = np.linspace(-(size-1)/2, (size-1)/2, size)

    xx, yy    = np.meshgrid(x, y)
    frequency = np.sqrt(xx**2. + yy**2.)

    np.seterr(divide='ignore')
    power = 0.023*np.power(size/(pupsize*fried), 5./3.)*np.power(frequency, -11./3.)  # Kolmogorov-Obukhov power law. 
    power[int(size/2), int(size/2)] = power[int(size/2)+1, int(size/2)+1]

    return np.sqrt(power)

def init_phase_odd_conj(iterations, N):

    if N % 2 == 0:
        raise ValueError('Dimension required to be odd')

    h = int(N/2)
    gen   = np.random.RandomState()
    phase = gen.normal(loc=0.0, scale=1.0, size=(iterations, N,N))
    top   = phase[:,:h,:]
    bot   = np.rot90(top, 2, axes=(1,2))
    line  = phase[:,h,:h]


    phase[:,h+1:,:]   = -bot
    phase[:,h,h+1:] = -np.fliplr(line)
    phase[:,h,h]   = 0
    return phase

def init_phase_odd_symm(iterations, N):

    if N % 2 == 0:
        raise ValueError('Dimension required to be odd')

    h = int(N/2)
    gen   = np.random.RandomState()
    phase = gen.normal(loc=0.0, scale=1.0, size=(iterations, N,N))
    top   = phase[:,:h,:]
    bot   = np.rot90(top, 2, axes=(1,2))
    line  = phase[:,h,:h]

    phase[:,h+1:,:]   = bot
    phase[:,h,h+1:] = np.fliplr(line)
    phase[:,h,h]   = 0
    return phase

def init_phase_gen(iterations, size):

    gen = np.random.RandomState()
    phase = (gen.normal(loc=0.0, scale=1.0, size=(iterations, size, size)))
    return phase

def return_wave(fried, size, wfsize, pupsize, iterations=400):

    if size % 2 == 0:
         old_size = size
         size = size+1
    else:
         old_size = size

    print('Simulating for fried parameter...%0.4f' %(fried))
    amp, phase = init_amplitude(fried, size, pupsize), init_phase_odd_symm(iterations, size) + 1j*init_phase_odd_conj(iterations, size)
    amp = amp.reshape(1, size, size).repeat(iterations, 0)
    print(amp.shape, phase.shape)
    fourier = amp*phase

    w  = np.fft.ifftn(np.fft.ifftshift(fourier, axes=(1,2)), axes=(1,2)).real[:,(size-old_size):,(size-old_size):]
    w *= size**2.

    print(w.shape)

    start, end = int((size-wfsize)/2), int((size+wfsize)/2)
    return w[:,start:end,start:end]

'''
Simulate phase-screens obeying the Kolmogorov-Obukhov power law.
Description can be found in Nagaraju et. al (2012) http://adsabs.harvard.edu/abs/2012ApOpt..51.7953K
'''
def kolmogorov(): 

    fried = np.linspace(0.01, 0.1, 2) # Fried parameter (in meters).

    size = 252		# Total size of the phase-screen from which the final realizations are cropped. 
			# Needs to be kept high enough to include enough power in tip/tilt modes. Currently
			# set at 4 times the diameter of the wavefront. 

    sampling = 1.0      # Sampling rate of the data on which the Point Spread Functions are used. A sampling of unity 
                        # denotes critical sampling. 

    wfsize = size/4.   
    pupsize = wfsize/sampling # Final size of the phase-screens. They correspond to a 1m aperture telescope. 
    iterations = 400    # Number of realizations per fried parameter. 

    parallelize = Pool(32)
    return_wave_parallel = partial(return_wave, size=size, wfsize=wfsize, pupsize=pupsize, iterations=iterations)
    return_wavefront = np.asarray(parallelize.map(return_wave_parallel, fried))

    print('Returned array size: ', return_wavefront.shape)

    return return_wavefront



# Crop Kolmogorov phase-screens to the required size, and set their individual pistons to zero.
# NB! The input data is modified
def apodize(dat, pupil):

    ptile = np.tile(pupil, (dat.shape[0], dat.shape[1], 1,1))
    dat *= ptile

    for i in range(dat.shape[0]):
        print('Iteration: ',i)
        for j in range(dat.shape[1]):
            dat[i,j,:,:] -= (dat[i,j,:,:].sum()/pupil.sum())*pupil


def return_psf(wfs):
 
    psf = np.zeros((wfs.shape[0], 2*wfs.shape[1]-1, 2*wfs.shape[2]-1))
    for i in range(wfs.shape[0]):
        pupil = mask*np.exp(1j*wfs[i,:,:])
        auto = correlate(pupil, pupil.conj(), mode='full')
        psf[i,:,:] = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(auto))).real
        psf[i,:,:] /= psf[i,:,:].sum()
 
    return psf

def calc_psf(wavefronts):

    #wavefronts = fits.open(sys.argv[1])[0].data

    parallelize = Pool(96)	 
    result = parallelize.map(return_psf, wavefronts)
    psf = np.asarray(result)

    return psf


def main():
    wavefront = kolmogorov()
    nx = np.shape(wavefront)[2]
    ny = np.shape(wavefront)[3]
    
    x1 = np.linspace(0., 1., nx)
    x2 = np.linspace(0., 1., ny)
    pupil_coords = np.dstack(np.meshgrid(x1, x2))
    pupil = utils.aperture_circ(pupil_coords, r=1.0, coef=5.0)

    my_plot = plot.plot_map(nrows=1, ncols=1)
    my_plot.plot(pupil)
    my_plot.save("pupil.png")
    my_plot.close()

    
    #pupil[int(nx/2),:] = 0
    #pupil[:,int(ny/2)] = 0
    apodize(wavefront, pupil)
    for i in np.arange(0, 2):
        for j in np.arange(0, 10):
            my_plot = plot.plot_map(nrows=1, ncols=1)
            my_plot.plot(wavefront[i,j,:,:])
            my_plot.save("kolmogorov" + str(i) + "_" + str(j) + ".png")
            my_plot.close()
    
if __name__ == "__main__":
    main()