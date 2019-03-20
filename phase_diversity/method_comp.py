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
import psf_sampler
import psf_basis
import psf_basis_sampler
import kolmogorov
import misc

import plot

###############################################################################
# Parameters
fried = np.linspace(0.25, 2., 1) # Fried parameter (in meters).
num_realizations = 10    # Number of realizations per fried parameter. 
jmax = 5
arcsec_per_px = 0.055
diameter = 50.0
wavelength = 5250.0
F_D = 1.0
gamma = 1.0
###############################################################################



def main():
    image = plt.imread('granulation.png')
    image = image[0:30,0:30]
    
    nx = np.shape(image)[0]
    ny = np.shape(image)[1]
    
    assert(nx == ny)
    
    image1 = utils.upscale(image)
    fimage = fft.fftshift(fft.fft2(image))
    fimage1 = fft.fftshift(fft.fft2(image1))

    
    aperture_func = lambda u: utils.aperture_circ(u, 1.0, 15.0)
    defocus_func = lambda xs: 2.*np.pi*np.sum(xs*xs, axis=2)#100.*(2*np.sum(xs*xs, axis=2) - 1.)

    wavefront = kolmogorov.kolmogorov(fried, num_realizations, nx*4, sampling=1.)
    
    x1 = np.linspace(-1., 1., nx)
    x2 = np.linspace(-1., 1., ny)
    pupil_coords = np.dstack(np.meshgrid(x1, x2))
    pupil = aperture_func(pupil_coords)

    my_plot = plot.plot_map(nrows=1, ncols=1)
    my_plot.plot(pupil)
    my_plot.save("pupil.png")
    my_plot.close()

    
    ###########################################################################
    # Create objects for image reconstruction
    ctf = psf.coh_trans_func(aperture_func, psf.phase_aberration(jmax), defocus_func)
    psf_ = psf.psf(ctf, nx, ny)
    sampler = psf_sampler.psf_sampler(psf_, gamma, num_samples=2)


    psf_b = psf_basis.psf_basis(jmax = jmax, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength, nx = nx, F_D = F_D)
    psf_b.create_basis()
    sampler_b = psf_basis_sampler.psf_basis_sampler(psf_b, gamma, num_samples=5)



    ###########################################################################
    
    plot_res = plot.plot_map(nrows=num_realizations + 1, ncols=7)
    
    image_est_mean = np.zeros((nx*2-1, nx*2-1))
    image_est_b_mean = np.zeros((nx, nx))
    D_mean = np.zeros((nx*2-1, nx*2-1))
    D_d_mean = np.zeros((nx*2-1, nx*2-1))
            
    image_norm = misc.normalize(image)
    image_norm1 = misc.normalize(image1)

    
    for i in np.arange(0, len(fried)):
        for j in np.arange(0, num_realizations):
            print("Realization: " + str(j))
            my_plot = plot.plot_map(nrows=1, ncols=1)
            my_plot.plot(wavefront[i,j,:,:])
            my_plot.save("kolmogorov" + str(i) + "_" + str(j) + ".png")
            my_plot.close()

            #pa_true = psf.phase_aberration(np.random.normal(size=5)*2)
            #pa_true = psf.phase_aberration([])
            #ctf_true = psf.coh_trans_func(aperture_func, pa_true, defocus_func)

            ctf_true = psf.coh_trans_func(aperture_func, psf.wavefront(wavefront[i,j,:,:]), defocus_func)
            psf_true = psf.psf(ctf_true, nx, ny)
            psf_vals_true = psf_true.calc(defocus=False)
            psf_vals_d_true = psf_true.calc(defocus=True)

            plot_psf = plot.plot_map(nrows=1, ncols=2)
            plot_psf.plot(psf_vals_true, [0])
            plot_psf.plot(psf_vals_d_true, [1])

            plot_psf.save("psf" + str(i) + "_" + str(j) + ".png")
            plot_psf.close()
            
            ###################################################################
            # Create convolved image and do the estimation
            DF, DF_d = psf_true.multiply(fimage1)
            
            alphas_est = sampler.sample(DF, DF_d, "samples" + str(j) + ".png")
            image_est = psf_.deconvolve(DF, DF_d, alphas_est, gamma, do_fft = True)
            
            DF1 = utils.downscale(DF)
            DF1_d = utils.downscale(DF_d) 

            betas_est = sampler_b.sample(DF1, DF1_d, "samples_b" + str(j) + ".png")
            image_est_b = psf_b.deconvolve(DF1, DF1_d, betas_est, gamma, do_fft = True)
        
            D = fft.ifft2(fft.ifftshift(DF)).real
            D_d = fft.ifft2(fft.ifftshift(DF_d)).real
        
            #image_min = np.min(image)
            #image_max = np.max(image)
            
            D_norm = misc.normalize(D)
            D_d_norm = misc.normalize(D_d)
            image_est_norm = misc.normalize(image_est)
            image_est_b_norm = misc.normalize(image_est_b)
            
        
            #my_plot.plot(image_norm, [trial, 0])
            #my_plot.plot(D_norm, [trial, 1])
            #my_plot.plot(D_d_norm, [trial, 2])
            #my_plot.plot(image_est_norm, [trial, 3])
            #my_plot.plot(np.abs(image_est_norm-image_norm), [trial, 4])
        
            plot_res.plot(image, [j, 0])
            plot_res.plot(D, [j, 1])
            plot_res.plot(D_d, [j, 2])
            plot_res.plot(image_est, [j, 3])
            plot_res.plot(np.abs(image_est-image1), [j, 4])
            plot_res.plot(image_est_b, [j, 5])
            plot_res.plot(np.abs(image_est_b-image), [j, 6])
            
            #image_est = fft.ifft2(fimage_est).real
            #image_est = np.roll(np.roll(image_est, int(nx/2), axis=0), int(ny/2), axis=1)
            image_est_mean += image_est_norm
            image_est_b_mean += image_est_b_norm
        
            D_mean += D_norm
            D_d_mean += D_d_norm
        
            plot_res.save("estimates.png")
            
            
    image_est_mean /= num_realizations
    image_est_b_mean /= num_realizations
    D_mean /= num_realizations
    D_d_mean /= num_realizations
    
    plot_res.plot(image_norm, [num_realizations, 0])
    plot_res.plot(D_mean, [num_realizations, 1])
    plot_res.plot(D_d_mean, [num_realizations, 2])
    plot_res.plot(image_est_mean, [num_realizations, 3])
    plot_res.plot(np.abs(image_est_mean-image_norm1), [num_realizations, 4])
    plot_res.plot(image_est_b_mean, [num_realizations, 5])
    plot_res.plot(np.abs(image_est_b_mean-image_norm), [num_realizations, 6])
    
    plot_res.save("estimates.png")
    plot_res.close()

if __name__ == "__main__":
    main()
