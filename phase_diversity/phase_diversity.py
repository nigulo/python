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
import sampling

import pymc3 as pm
import pyhdust.triangle as triangle
import psf_basis
import psf
import kolmogorov
import scipy.optimize
import matplotlib.pyplot as plt

do_sampling = False
num_samples = 500
num_chains = 3

num_MAP_trials = 10


num_frames = 10

image = plt.imread('granulation.png')

image = image[0:20,0:20]

nx_orig = np.shape(image)[0]
ny_orig = np.shape(image)[1]

image = utils.resize(image)

nx = np.shape(image)[0]
ny = np.shape(image)[1]

assert(nx == ny)

fimage = fft.fft2(image)
#fimage = fft.fftshift(fimage)
    
jmax = 5
arcsec_per_px = 0.055
diameter = 50.0
wavelength = 5250.0
F_D = 1.0
gamma = 1.0


def sample(D, D_d, gamma, psf_b, plot_file = None):



    betas_est = np.zeros(psf_b.jmax, dtype='complex')    

    s = sampling.Sampling()
    if do_sampling:
        betas_real = [None] * jmax
        betas_imag = [None] * jmax
        with s.get_model():
            for i in np.arange(0, psf_b.jmax):
                betas_real[i] = pm.Normal('beta_r' + str(i), sd=1.0)
                betas_imag[i] = pm.Normal('beta_i' + str(i), sd=1.0)

        trace = s.sample(psf_b.likelihood, betas_real + betas_imag, [D, D_d, gamma], num_samples, num_chains, psf_b.likelihood_grad)
        samples = []
        var_names = []
        for i in np.arange(0, psf_b.jmax):
            var_name_r = 'beta_r' + str(i)
            var_name_i = 'beta_i' + str(i)
            samples.append(trace[var_name_r])
            samples.append(trace[var_name_i])
            var_names.append(r"$\Re \beta_" + str(i) + r"$")
            var_names.append(r"$\Im \beta_" + str(i) + r"$")
            betas_est[i] = np.mean(trace[var_name_r]) + 1.j*np.mean(trace[var_name_i])
        
        samples = np.asarray(samples).T
        if plot_file is not None:
            fig, ax = plt.subplots(nrows=psf_b.jmax*2, ncols=psf_b.jmax*2)
            fig.set_size_inches(6*psf_b.jmax, 6*psf_b.jmax)
            triangle.corner(samples[:,:], labels=var_names, fig=fig)
            fig.savefig(plot_file)

    else:
        
        def lik_fn(params):
            return psf_b.likelihood(params, [D, D_d, gamma])

        def grad_fn(params):
            return psf_b.likelihood_grad(params, [D, D_d, gamma])
        
        
        # Optional methods:
        # Nelder-Mead Simplex algorithm (method='Nelder-Mead')
        # Broyden-Fletcher-Goldfarb-Shanno algorithm (method='BFGS')
        # Powell’s method (method='powell')
        # Newton-Conjugate-Gradient algorithm (method='Newton-CG')
        # Trust-Region Newton-Conjugate-Gradient Algorithm (method='trust-ncg')
        # Trust-Region Truncated Generalized Lanczos / Conjugate Gradient Algorithm (method='trust-krylov')
        # Trust-Region Nearly Exact Algorithm (method='trust-exact')
        # Trust-Region Constrained Algorithm (method='trust-constr')
        # Sequential Least SQuares Programming (SLSQP) Algorithm (method='SLSQP')
        # Unconstrained minimization (method='brent')
        # Bounded minimization (method='bounded')
        #
        # Not all of them use the given gradient function.

        min_loglik = None
        min_res = None
        for trial_no in np.arange(0, num_MAP_trials):
            res = scipy.optimize.minimize(lik_fn, np.random.normal(size=psf_b.jmax*2), method='BFGS', jac=grad_fn, options={'disp': True})
            loglik = res['fun']
            #assert(loglik == lik_fn(res['x']))
            if min_loglik is None or loglik < min_loglik:
                min_loglik = loglik
                min_res = res
        for i in np.arange(0, psf_b.jmax):
            betas_est[i] = min_res['x'][i] + 1.j*min_res['x'][psf_b.jmax + i]
        
    print(betas_est)
    #betas_est = np.random.normal(size=psf.jmax) + np.random.normal(size=psf.jmax)*1.j
    
    return betas_est


aperture_func = lambda u: utils.aperture_circ(u, 1.0, 15.0)

psf_b = psf_basis.psf_basis(jmax = jmax, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength, nx = nx, F_D = F_D)
psf_b.create_basis()

my_plot = plot.plot_map(nrows=num_frames + 1, ncols=5)

image_est_mean = np.zeros((nx, nx))
D_mean = np.zeros((nx, nx))
D_d_mean = np.zeros((nx, nx))
        
image_norm = misc.normalize(image)

wavefront = kolmogorov.kolmogorov(fried = np.array([.5]), num_realizations=num_frames, size=4*nx_orig, sampling=1.)

for trial in np.arange(0, num_frames):
    
    
    #pa = psf.phase_aberration([(trial+1, 10.0)])
    pa = psf.phase_aberration([])
    ctf = psf.coh_trans_func(aperture_func, pa, lambda xs: 100.*(2*np.sum(xs*xs, axis=2) - 1.))
    #ctf = psf.coh_trans_func(aperture_func, psf.wavefront(wavefront[0,trial,:,:]), lambda xs: 100.*(2*np.sum(xs*xs, axis=2) - 1.))
    psf_ = psf.psf(ctf, nx_orig, ny_orig)
    
    D, D_d = psf_.multiply(fimage)
    
    betas_est = sample(D, D_d, gamma, psf_b, "samples" + str(trial) + ".png")
    
    image_est = psf_b.deconvolve(D, D_d, betas_est, gamma, do_fft = True)
    #image_est = psf_.deconvolve(D, D_d, gamma, do_fft = True)

    D = fft.fftshift(fft.ifft2(D).real)
    D_d = fft.fftshift(fft.ifft2(D_d).real)

    #image_min = np.min(image)
    #image_max = np.max(image)
    
    D_norm = misc.normalize(D)
    D_d_norm = misc.normalize(D_d)
    image_est_norm = misc.normalize(image_est)
    

    #my_plot.plot(image_norm, [trial, 0])
    #my_plot.plot(D_norm, [trial, 1])
    #my_plot.plot(D_d_norm, [trial, 2])
    #my_plot.plot(image_est_norm, [trial, 3])
    #my_plot.plot(np.abs(image_est_norm-image_norm), [trial, 4])

    my_plot.plot(image, [trial, 0])
    my_plot.plot(D, [trial, 1])
    my_plot.plot(D_d, [trial, 2])
    my_plot.plot(image_est, [trial, 3])
    my_plot.plot(np.abs(image_est-image), [trial, 4])
    
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
