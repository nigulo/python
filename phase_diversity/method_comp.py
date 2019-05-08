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
import pickle

import plot

state_file = None#state.pkl"
if len(sys.argv) > 1:
    state_file = sys.argv[1]

print(state_file)

def load(filename):
    if filename is not None:
        data_file = filename
        if os.path.isfile(data_file):
            return pickle.load(open(data_file, 'rb'))
    return None

def save(filename, state):
    if filename is None:
        filename = "state.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(state, f)

###############################################################################
# Parameters
fried = np.linspace(0.2, 2., 1) # Fried parameter (in meters).
num_realizations = 5    # Number of realizations per fried parameter. 
#jmax = 5
#arcsec_per_px = 0.055
#diameter = 50.0
#wavelength = 5250.0
#F_D = 1.0
#gamma = 1.0
###############################################################################

def get_params(nx):
    coef1 = 4.**(-np.log2(float(nx)/11))
    coef2 = 2.**(-np.log2(float(nx)/11))
    print("coef1, coef2", coef1, coef2)
    arcsec_per_px = coef1*0.1
    print("arcsec_per_px=", arcsec_per_px)
    defocus = 3.
    return (arcsec_per_px, defocus)

def calibrate(arcsec_per_px, nx):
    coef = np.log2(float(nx)/11)
    return 3.0*arcsec_per_px*2.**coef


def main():
    image = plt.imread('granulation1.png')
    image = image[0:100,0:100,0]
    
    nx_orig = np.shape(image)[0]
    image = utils.upsample(image)
    assert(np.shape(image)[0] == np.shape(image)[1])
    nx = np.shape(image)[0]
    
    state = load(state_file)


    if state == None:
        print("Creating new state")
        jmax = 3
        #arcsec_per_px = 0.057
        #arcsec_per_px = 0.011
        diameter = 20.0
        wavelength = 5250.0
        gamma = 1.0
        nx = np.shape(image)[0]
    
        arcsec_per_px, defocus = get_params(nx_orig)#wavelength/diameter*1e-8*180/np.pi*3600
        #arcsec_per_px1=wavelength/diameter*1e-8*180/np.pi*3600/4.58
    
        psf_b = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = calibrate(arcsec_per_px, nx_orig), diameter = diameter, wavelength = wavelength, defocus = defocus*2.2)
        psf_b.create_basis()
    
        save(state_file, [jmax, arcsec_per_px, diameter, wavelength, defocus, gamma, nx, psf_b.get_state()])
    else:
        print("Using saved state")
        jmax = state[0]
        arcsec_per_px = state[1]
        diameter = state[2]
        wavelength = state[3]
        defocus = state[4]
        gamma = state[5]
        nx = state[6]
        #arcsec_per_px1=wavelength/diameter*1e-8*180/np.pi*3600/4.58
        
        assert(nx == np.shape(image)[0])
        
        psf_b = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = calibrate(arcsec_per_px, nx_orig), diameter = diameter, wavelength = wavelength, defocus = defocus*2.2)
        psf_b.set_state(state[7])


    
    fimage = fft.fft2(image)
    fimage = fft.fftshift(fimage)

    
    aperture_func = lambda xs: utils.aperture_circ(xs, coef=15., radius =1.)
    defocus_func = lambda xs: defocus*2*np.sum(xs*xs, axis=2)

    wavefront = kolmogorov.kolmogorov(fried, num_realizations, nx_orig*4, sampling=1.)
    
    x1 = np.linspace(-1., 1., nx)
    pupil_coords = np.dstack(np.meshgrid(x1, x1))
    pupil = aperture_func(pupil_coords)

    my_plot = plot.plot(nrows=1, ncols=1)
    my_plot.colormap(pupil)
    my_plot.save("pupil.png")
    my_plot.close()

    
    ###########################################################################
    # Create objects for image reconstruction
    ctf = psf.coh_trans_func(aperture_func, psf.phase_aberration(jmax*2), defocus_func)
    psf_ = psf.psf(ctf, nx_orig, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)
    sampler = psf_sampler.psf_sampler(psf_, gamma, num_samples=1)

    sampler_b = psf_basis_sampler.psf_basis_sampler(psf_b, gamma, num_samples=1)



    ###########################################################################
    
    plot_res = plot.plot(nrows=num_realizations + 1, ncols=7)
    
    image_est_mean = np.zeros((nx, nx))
    image_est_b_mean = np.zeros((nx, nx))
    D_mean = np.zeros((nx, nx))
    D_d_mean = np.zeros((nx, nx))
            
    image_norm = misc.normalize(image)

    
    for i in np.arange(0, len(fried)):
        for j in np.arange(0, num_realizations):
            print("Realization: " + str(j))
            my_plot = plot.plot(nrows=1, ncols=1)
            my_plot.colormap(wavefront[i,j,:,:])
            my_plot.save("kolmogorov" + str(i) + "_" + str(j) + ".png")
            my_plot.close()

            #pa_true = psf.phase_aberration(np.random.normal(size=5)*2)
            #pa_true = psf.phase_aberration([])
            #ctf_true = psf.coh_trans_func(aperture_func, pa_true, defocus_func)

            #pa_true = psf.phase_aberration(np.random.normal(size=5)*.001)
            #pa_true = psf.phase_aberration([])
            #ctf_true = psf.coh_trans_func(aperture_func, pa_true, defocus_func)
            ctf_true = psf.coh_trans_func(aperture_func, psf.wavefront(wavefront[i,j,:,:]), defocus_func)
            psf_true = psf.psf(ctf_true, nx_orig, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)
            psf_vals_true = psf_true.calc(defocus=False)
            psf_vals_d_true = psf_true.calc(defocus=True)

            plot_psf = plot.plot(nrows=1, ncols=2)
            plot_psf.colormap(psf_vals_true, [0])
            plot_psf.colormap(psf_vals_d_true, [1])

            plot_psf.save("psf" + str(i) + "_" + str(j) + ".png")
            plot_psf.close()
            
            ###################################################################
            # Create convolved image and do the estimation
            DF, DF_d = psf_true.multiply(fimage)

            alphas_est = sampler.sample(DF, DF_d, "samples" + str(j) + ".png")
            image_est = psf_.deconvolve(DF, DF_d, alphas_est, gamma, do_fft = True)

            DF = fft.ifftshift(DF)
            DF_d = fft.ifftshift(DF_d)
            
            betas_est = sampler_b.sample(DF, DF_d, "samples_b" + str(j) + ".png")
            image_est_b = psf_b.deconvolve(DF, DF_d, betas_est, gamma, do_fft = True)
        
            D = fft.ifft2(DF).real
            D_d = fft.ifft2(DF_d).real
        
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
        
            plot_res.colormap(image, [j, 0])
            plot_res.colormap(D, [j, 1])
            plot_res.colormap(D_d, [j, 2])
            plot_res.colormap(image_est, [j, 3])
            plot_res.colormap(np.abs(image_est_norm-image_norm), [j, 4])
            plot_res.colormap(image_est_b, [j, 5])
            plot_res.colormap(np.abs(image_est_b_norm-image_norm), [j, 6])
            
            image_est_mean += image_est_norm
            image_est_b_mean += image_est_b_norm
        
            D_mean += D_norm
            D_d_mean += D_d_norm
        
            plot_res.save("method_comp.png")
            
            
    image_est_mean /= num_realizations
    image_est_b_mean /= num_realizations
    D_mean /= num_realizations
    D_d_mean /= num_realizations
    
    plot_res.colormap(image_norm, [num_realizations, 0])
    plot_res.colormap(D_mean, [num_realizations, 1])
    plot_res.colormap(D_d_mean, [num_realizations, 2])
    plot_res.colormap(image_est_mean, [num_realizations, 3])
    plot_res.colormap(np.abs(image_est_mean-image_norm), [num_realizations, 4])
    plot_res.colormap(image_est_b_mean, [num_realizations, 5])
    plot_res.colormap(np.abs(image_est_b_mean-image_norm), [num_realizations, 6])
    
    plot_res.save("method_comp.png")
    plot_res.close()

if __name__ == "__main__":
    main()
