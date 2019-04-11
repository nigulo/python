import os
os.environ["OMP_NUM_THREADS"] = "3" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "3" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "3" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "3" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "3" # export NUMEXPR_NUM_THREADS=6


import numpy as np
import scipy.misc
import scipy.special as special
import scipy.signal as signal
import numpy.fft as fft
import utils
import sys
sys.path.append('../utils')
import plot
import misc
import psf_basis_sampler

import pymc3 as pm
import pyhdust.triangle as triangle
import psf_basis
import psf
import kolmogorov
import scipy.optimize
import matplotlib.pyplot as plt

num_frames = 10

image = plt.imread('granulation.png')

image = image[0:20,0:20]

nx_orig = np.shape(image)[0]

image = utils.upsample(image)

nx = np.shape(image)[0]

assert(np.shape(image)[0] == np.shape(image)[1])

fimage = fft.fftshift(fft.fft2(image))
    
jmax = 5
arcsec_per_px = 0.055
diameter = 20.0
wavelength = 5250.0
f = 0#0.1
gamma = 1.0


aperture_func = lambda xs: utils.aperture_circ(xs, 1.0, 15.0)
#defocus_func = lambda xs: 2.*np.pi*np.sum(xs*xs, axis=2)#10.*(2*np.sum(xs*xs, axis=2) - 1.)
defocus_func = lambda xs: 0.#np.sum(xs*xs, axis=2)


psf_b = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength, defocus = f)
psf_b.create_basis()

my_plot = plot.plot_map(nrows=num_frames + 1, ncols=5)

image_est_mean = np.zeros((nx, nx))
D_mean = np.zeros((nx, nx))
D_d_mean = np.zeros((nx, nx))
        
image_norm = misc.normalize(image)

wavefront = kolmogorov.kolmogorov(fried = np.array([.5]), num_realizations=num_frames, size=4*nx_orig, sampling=1.)

sampler = psf_basis_sampler.psf_basis_sampler(psf_b, gamma, num_samples=1)
for trial in np.arange(0, num_frames):
    
    #pa = psf.phase_aberration(np.random.normal(size=5)*2)
    pa = psf.phase_aberration([])
    ctf = psf.coh_trans_func(aperture_func, pa, defocus_func)
    #ctf = psf.coh_trans_func(aperture_func, psf.wavefront(wavefront[0,trial,:,:]), defocus_func)
    psf_ = psf.psf(ctf, nx_orig, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)
    
    D, D_d = psf_.multiply(fimage)
    
    betas_est = sampler.sample(D, D_d, "samples" + str(trial) + ".png")    
    image_est = psf_b.deconvolve(D, D_d, betas_est, gamma, do_fft = True)
    image_est = psf_basis.maybe_invert(image_est, image)


    #image_est = psf_.deconvolve(D, D_d, gamma, do_fft = True)

    D = fft.ifft2(fft.ifftshift(D)).real
    D_d = fft.ifft2(fft.ifftshift(D_d)).real

    #image_min = np.min(image)
    #image_max = np.max(image)
    
    D_norm = misc.normalize(D)
    D_d_norm = misc.normalize(D_d)
    image_est_norm = misc.normalize(image_est)

    my_plot.plot(image, [trial, 0])
    my_plot.plot(D, [trial, 1])
    my_plot.plot(D_d, [trial, 2])
    my_plot.plot(image_est_norm, [trial, 3])
    my_plot.plot(np.abs(image_est_norm-image_norm), [trial, 4])
    
    #image_est = fft.ifft2(fimage_est).real
    #image_est = np.roll(np.roll(image_est, int(nx/2), axis=0), int(ny/2), axis=1)
    image_est_mean += image_est_norm

    D_mean += D_norm
    D_d_mean += D_d_norm

    my_plot.save("estimates.png")

image_est_mean /= num_frames
D_mean /= num_frames
D_d_mean /= num_frames

my_plot.plot(image_norm, [num_frames, 0])
my_plot.plot(D_mean, [num_frames, 1])
my_plot.plot(D_d_mean, [num_frames, 2])
my_plot.plot(image_est_mean, [num_frames, 3])
my_plot.plot(np.abs(image_est_mean-image_norm), [num_frames, 4])

my_plot.save("estimates.png")
my_plot.close()
