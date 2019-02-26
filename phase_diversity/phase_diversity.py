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
import matplotlib.colorbar as cb
from matplotlib import cm
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.ticker import LogFormatterMathtext, FormatStrFormatter
import matplotlib.colors as colors
import matplotlib.ticker as ticker

import numpy as np
import scipy.special as special
import scipy.signal as signal
import numpy.fft as fft
import matplotlib.pyplot as plt
from matplotlib import cm
import sampling
import pymc3 as pm

import pyhdust.triangle as triangle
import matplotlib.pyplot as plt
import psf_basis
import scipy.optimize

def reverse_colourmap(cmap, name = 'my_cmap_r'):
     return mpl.colors.LinearSegmentedColormap(name, cm.revcmap(cmap._segmentdata))
my_cmap = reverse_colourmap(plt.get_cmap('binary'))#plt.get_cmap('winter')


do_sampling = False
num_samples = 500
num_chains = 3

num_MAP_trials = 5


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


def sample(D, D_d, gamma, psf, plot_file = None):



    betas_est = np.zeros(psf.jmax, dtype='complex')    

    s = sampling.Sampling()
    if do_sampling:
        betas_real = [None] * jmax
        betas_imag = [None] * jmax
        with s.get_model():
            for i in np.arange(0, psf.jmax):
                betas_real[i] = pm.Normal('beta_r' + str(i), sd=1.0)
                betas_imag[i] = pm.Normal('beta_i' + str(i), sd=1.0)

        trace = s.sample(psf.likelihood, betas_real + betas_imag, [D, D_d, gamma], num_samples, num_chains, psf.likelihood_grad)
        samples = []
        var_names = []
        for i in np.arange(0, psf.jmax):
            var_name_r = 'beta_r' + str(i)
            var_name_i = 'beta_i' + str(i)
            samples.append(trace[var_name_r])
            samples.append(trace[var_name_i])
            var_names.append(r"$\Re \beta_" + str(i) + r"$")
            var_names.append(r"$\Im \beta_" + str(i) + r"$")
            betas_est[i] = np.mean(trace[var_name_r]) + 1.j*np.mean(trace[var_name_i])
        
        samples = np.asarray(samples).T
        if plot_file is not None:
            fig, ax = plt.subplots(nrows=psf.jmax*2, ncols=psf.jmax*2)
            fig.set_size_inches(6*psf.jmax, 6*psf.jmax)
            triangle.corner(samples[:,:], labels=var_names, fig=fig)
            fig.savefig(plot_file)

    else:
        
        def lik_fn(params):
            return psf.likelihood(params, [D, D_d, gamma])

        def grad_fn(params):
            return psf.likelihood_grad(params, [D, D_d, gamma])
        
        
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
            res = scipy.optimize.minimize(lik_fn, np.random.normal(size=psf.jmax*2), method='BFGS', jac=grad_fn, options={'disp': True})
            loglik = res['fun']
            #assert(loglik == lik_fn(res['x']))
            if min_loglik is None or loglik < min_loglik:
                min_loglik = loglik
                min_res = res
        for i in np.arange(0, psf.jmax):
            betas_est[i] = min_res['x'][i] + 1.j*min_res['x'][psf.jmax + i]
        
    print(betas_est)
    #betas_est = np.random.normal(size=psf.jmax) + np.random.normal(size=psf.jmax)*1.j
    
    return betas_est


psf = psf_basis.psf_basis(jmax = jmax, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength, nx = nx, F_D = F_D)
psf.create_basis()

extent=[0., 1., 0., 1.]
plot_aspect=(extent[1]-extent[0])/(extent[3]-extent[2])#*2/3 

fig, axes = plt.subplots(nrows=num_frames + 1, ncols=5)
fig.set_size_inches(4.5*5, 3*(num_frames+1))

image_est_mean = np.zeros((nx, nx))
D_mean = np.zeros((nx, nx))
D_d_mean = np.zeros((nx, nx))

def plot(fig, ax, dat, colorbar = True):
    im = ax.imshow(dat[::-1],extent=extent,cmap=my_cmap,origin='lower')
    ax.set_aspect(aspect=plot_aspect)
    
    if colorbar:
        l_f = FormatStrFormatter('%1.1f')

        pos = ax.get_position().get_points()
        #x0 = pos[0, 0]
        y0 = pos[0, 1]
        x1 = pos[1, 0]
        y1 = pos[1, 1]
        
        cbar_ax = fig.add_axes([x1 + 0.01, y0, 0.02, y1-y0])
        fig.colorbar(im, cax=cbar_ax, format=l_f)#, label=r'Label')
        

for trial in np.arange(0, num_frames):
    betas = np.random.normal(size=jmax) + np.random.normal(size=jmax)*1.j
    
    D, D_d = psf.multiply(fimage, betas)
    
    betas_est = sample(D, D_d, gamma, psf, "samples" + str(trial) + ".png")
    
    image_est = psf.deconvolve(D, D_d, betas_est, gamma, do_fft = True)

    D = fft.ifft2(D).real
    D = np.roll(np.roll(D, int(nx/2), axis=0), int(ny/2), axis=1)

    D_d = fft.ifft2(D_d).real
    D_d = np.roll(np.roll(D_d, int(nx/2), axis=0), int(ny/2), axis=1)

    ax1, ax2, ax3, ax4, ax5 = axes[trial]

    #image_min = min([np.min(image), np.min(D), np.min(D_d), np.min(image_est)])
    #image_max = max([np.max(image), np.max(D), np.max(D_d), np.max(image_est)])

    plot(fig, ax1, image)
    plot(fig, ax2, D)
    plot(fig, ax3, D_d)
    plot(fig, ax4, image_est)
    plot(fig, ax5, (image_est-image))
    
    #image_est = fft.ifft2(fimage_est).real
    #image_est = np.roll(np.roll(image_est, int(nx/2), axis=0), int(ny/2), axis=1)
    image_est_mean += image_est

    D_mean += D
    D_d_mean += D_d

    fig.savefig("estimates.png")

image_est_mean /= num_frames
D_mean /= num_frames
D_d_mean /= num_frames

ax1, ax2, ax3, ax4, ax5 = axes[num_frames]

plot(fig, ax1, image)
plot(fig, ax2, D_mean)
plot(fig, ax3, D_d_mean)
plot(fig, ax4, image_est_mean)
plot(fig, ax5, (image_est_mean-image))

fig.savefig("estimates.png")
plt.close(fig)
