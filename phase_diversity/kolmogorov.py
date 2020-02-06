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
from multiprocessing import Pool
from functools import partial

import utils
import psf

import plot


pool_size = 3 # Thread pool size

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

    print('Simulating for fried parameter...%f' %(fried))
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
def kolmogorov(fried, num_realizations, size, sampling): 

    wfsize = size/4.   
    pupsize = wfsize/sampling # Final size of the phase-screens. They correspond to a 1m aperture telescope. 

    parallelize = Pool(pool_size)
    return_wave_parallel = partial(return_wave, size=size, wfsize=wfsize, pupsize=pupsize, iterations=num_realizations)
    return_wavefront = np.asarray(parallelize.map(return_wave_parallel, fried))
    parallelize.close()
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


def main():
    
    fried = np.linspace(.1, 2., 1) # Fried parameter (in meters).
    num_realizations = 10    # Number of realizations per fried parameter. 
    
    size = 200		# Total size of the phase-screen from which the final realizations are cropped. 
			# Needs to be kept high enough to include enough power in tip/tilt modes. Currently
			# set at 4 times the diameter of the wavefront. 

    sampling = 1.      # Sampling rate of the data on which the Point Spread Functions are used. A sampling of unity 
                        # denotes critical sampling. 

    
    aperture_func = lambda u: utils.aperture_circ(u, 0.2, 15.0)

    wavefront = kolmogorov(fried, num_realizations, size, sampling)
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
