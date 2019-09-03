import os
import sys
sys.path.append('../utils')
sys.path.append('..')
import config

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

import time
import utils
import psf
import psf_sampler
import psf_basis
import psf_basis_sampler
import kolmogorov
import misc
import pickle
import tip_tilt
from astropy.io import fits

import plot

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
        pickle.dump(state, f, protocol=4)

###############################################################################
# Parameters
num_realizations = 5    # Number of realizations per fried parameter. 
max_frames = min(10, num_realizations)
fried_param=5.
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
    arcsec_per_px = coef1*0.2
    print("arcsec_per_px=", arcsec_per_px)
    defocus = 0.#3.
    return (arcsec_per_px, defocus)

def calibrate(arcsec_per_px, nx):
    coef = np.log2(float(nx)/11)
    return 3.0*arcsec_per_px*2.**coef


def main():
    
    state_file = None#state.pkl"
    wavefront_file = None#state.pkl"
    image_file = None
    
    for arg in sys.argv:
        if arg[:6] == "state=":
            state_file = arg[6:]
        elif arg[:10] == "wavefront=":
            wavefront_file = arg[10:]
        elif arg[:6] == "image=":
            image_file = arg[6:]
    
    #if len(sys.argv) > 1:
    #    state_file = sys.argv[1]
    
    #if len(sys.argv) > 2:
    #    wavefront_file = sys.argv[2]
    
    print(state_file)
    print(wavefront_file)
    print(image_file)

    if image_file is None:
        image_file = 'icont.fits'
        dir = "images"


    print(image_file)
    if image_file[-5:] == '.fits':
        hdul = fits.open(dir + "/" + image_file)
        image = hdul[0].data
        hdul.close()
    else:
        image = plt.imread(dir + "/" + image_file)[:, :, 0]
        #image = plt.imread(dir + "/" + file)
    image = misc.sample_image(image, .5)
    print("Image shape", image.shape)

    nx_orig = 50
    start_index_x = 0#np.random.randint(0, start_index_max)
    start_index_y = 0#np.random.randint(0, start_index_max)
    
    image = image[start_index_x:start_index_x + nx_orig,start_index_y:start_index_y + nx_orig]
    
    nx_orig = np.shape(image)[0]
    image = utils.upsample(image)
    assert(np.shape(image)[0] == np.shape(image)[1])
    nx = np.shape(image)[0]

    state = load(state_file)
    wavefront = load(wavefront_file)



    if state == None:
        print("Creating new state")
        jmax = 10
        #arcsec_per_px = 0.057
        #arcsec_per_px = 0.011
        diameter = 20.0
        wavelength = 5250.0
        gamma = 1.0
        nx = np.shape(image)[0]
    
        arcsec_per_px, defocus1 = get_params(nx_orig)#wavelength/diameter*1e-8*180/np.pi*3600
        (defocus_psf, defocus_psf_b) = defocus1
        #arcsec_per_px1=wavelength/diameter*1e-8*180/np.pi*3600/4.58
    
    
        if num_realizations == 1:
            tt = None
        else:
            coords, _, _ = utils.get_coords(nx, arcsec_per_px, diameter, wavelength)
            tt = tip_tilt.tip_tilt(coords, prior_prec=((np.max(coords[0])-np.min(coords[0]))/2)**2)

        psf_b = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength, defocus = defocus_psf_b, tip_tilt=tt)
        psf_b.create_basis()
    
        save(state_file, [jmax, arcsec_per_px, diameter, wavelength, defocus1, gamma, nx, psf_b.get_state()])
    else:
        print("Using saved state")
        jmax = state[0]
        arcsec_per_px = state[1]
        diameter = state[2]
        wavelength = state[3]
        defocus1 = state[4]
        (defocus_psf, defocus_psf_b) = defocus1
        gamma = state[5]
        nx = state[6]
        #arcsec_per_px1=wavelength/diameter*1e-8*180/np.pi*3600/4.58
        print("jmax, arcsec_per_px, diameter, wavelength, defocus, gamma, nx", jmax, arcsec_per_px, diameter, wavelength, defocus1, gamma, nx)
        
        assert(nx == np.shape(image)[0])

        
        if num_realizations == 1:
            tt = None
        else:
            coords, _, _ = utils.get_coords(nx, arcsec_per_px, diameter, wavelength)
            tt = tip_tilt.tip_tilt(coords, prior_prec=((np.max(coords[0])-np.min(coords[0]))/2)**2)
        
        psf_b = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength, defocus = defocus_psf_b, tip_tilt=tt)
        psf_b.set_state(state[7])


    for iii in np.arange(0, 2):
        my_test_plot = plot.plot()
        my_test_plot.colormap(image)
        my_test_plot.save("critical_sampling" + str(iii) + ".png")
        my_test_plot.close()
        image = psf_basis.critical_sampling(image, arcsec_per_px, diameter, wavelength)
    
    fimage = fft.fft2(image)
    fimage = fft.fftshift(fimage)

    
    aperture_func = lambda xs: utils.aperture_circ(xs, coef=15., radius =1.)
    defocus_func = lambda xs: defocus_psf*np.sum(xs*xs, axis=2)

    if wavefront is None:
        wavefront = kolmogorov.kolmogorov(fried = np.array([fried_param]), num_realizations=num_realizations, size=4*nx_orig, sampling=1.)
        save("wavefront.pkl", wavefront)

   
    x1 = np.linspace(-1., 1., nx)
    pupil_coords = np.dstack(np.meshgrid(x1, x1))
    pupil = aperture_func(pupil_coords)

    my_plot = plot.plot(nrows=1, ncols=1)
    my_plot.colormap(pupil)
    my_plot.save("pupil.png")
    my_plot.close()

    
    ###########################################################################
    # Create objects for image reconstruction
    
    ctf = psf.coh_trans_func(aperture_func, psf.phase_aberration(jmax), defocus_func)
    psf_ = psf.psf(ctf, nx_orig, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength, tip_tilt=None)
    sampler = psf_sampler.psf_sampler(psf_, gamma, num_samples=1)

    sampler_b = psf_basis_sampler.psf_basis_sampler(psf_b, gamma, num_samples=1)



    ###########################################################################
    
    Ds = np.zeros((num_realizations, 2, nx, nx), dtype='complex') # in Fourier space
    Ds1 = np.zeros((num_realizations, 2, nx, nx)) # in image space
    Ps = np.ones((num_realizations, 2, nx, nx), dtype='complex')
    
    D_mean = np.zeros((nx, nx))
    D_d_mean = np.zeros((nx, nx))
    
    for i in np.arange(0, num_realizations):
        print("Realization: " + str(i))
        my_plot1 = plot.plot(nrows=1, ncols=1)
        my_plot1.colormap(wavefront[0,i,:,:])
        my_plot1.save("kolmogorov" + str(i) + ".png")
        my_plot1.close()

        #pa_true = psf.phase_aberration(np.random.normal(size=5)*2)
        #pa_true = psf.phase_aberration([])
        #ctf_true = psf.coh_trans_func(aperture_func, pa_true, defocus_func)

        #pa_true = psf.phase_aberration(np.random.normal(size=5)*.001)
        pa_true = psf.phase_aberration([])
        ctf_true = psf.coh_trans_func(aperture_func, pa_true, defocus_func)
        #ctf_true = psf.coh_trans_func(aperture_func, psf.wavefront(wavefront[0,i,:,:]), defocus_func)
        psf_true = psf.psf(ctf_true, nx_orig, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)

        ###################################################################
        # Create convolved image and do the estimation
        DFs = psf_true.multiply(fimage)
        DF = DFs[0, 0]
        DF_d = DFs[0, 1]
        
        DF = fft.ifftshift(DF)
        DF_d = fft.ifftshift(DF_d)

        D = fft.ifft2(DF).real
        D_d = fft.ifft2(DF_d).real
        
        Ds[i, 0] = DF
        Ds[i, 1] = DF_d

        Ds1[i, 0] = D
        Ds1[i, 1] = D_d
    
    
        D_mean += D
        D_d_mean += D_d
        
        
        # TESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTEST
        # Remove after checked
        D1 = DF
        D1_d = DF_d
        print("psf_true.otf_vals", psf_true.otf_vals.shape)
        P1 = psf_true.otf_vals[0, 0, :, :]
        P1_d = psf_true.otf_vals[0, 1, :, :]
    
        P1_conj = P1.conjugate()
        P1_d_conj = P1_d.conjugate()
    
    
        #D1 = fft.ifftshift(D1)
        #D1_d = fft.ifftshift(D1_d)
        P1 = fft.ifftshift(P1)
        P1_d = fft.ifftshift(P1_d)
        
        F1_image = D1 * P1_conj + gamma * D1_d * P1_d_conj
        den1 = P1*P1_conj + gamma * P1_d * P1_d_conj
        F1_image /= den1
    
   
        if False:
            D1 = fft.ifftshift(D1)
            D1_d = fft.ifftshift(D1_d)
            F1_image = fft.ifftshift(F1_image)
            P1 = fft.ifft2(fft.ifftshift(P1)).real
            P1_d = fft.ifft2(fft.ifftshift(P1_d)).real
        else:
            P1 = fft.ifftshift(fft.ifft2(P1)).real
            P1_d = fft.ifftshift(fft.ifft2(P1_d)).real
        D1 = fft.ifft2(D1).real
        D1_d = fft.ifft2(D1_d).real
        my_plot = plot.plot(nrows=1, ncols=5)
        my_plot.colormap(D1, [0])
        my_plot.colormap(D1_d, [1])
        my_plot.colormap(fft.ifft2(F1_image).real, [2])
        my_plot.colormap(np.log(P1), [3])
        my_plot.colormap(np.log(P1_d), [4])
        my_plot.save("method_comp_deconvolve" + str(i) + ".png")
        # Remove after checked
        # TESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTEST
        

    start = time.time()

    res = sampler.sample(Ds, "samples.png")
    if False:#tt is not None:
        alphas_est, a_est = res
    else:
        alphas_est = res
        a_est = None
    #print("betas_est, a_est", betas_est, a_est)
    image_est, F, Ps = psf_.deconvolve(Ds, alphas_est, gamma, ret_all = True, a_est=a_est, normalize=True)
    #image_est, F, Ps = psf_b.deconvolve(Ds, alphas_est, gamma, ret_all = True, a_est=a_est, normalize=True)

    end = time.time()
    print("PSF reconstruction took: " + str(end - start))


    start = time.time()
    res_b = sampler_b.sample(Ds, "samples_b.png")
    if tt is not None:
        betas_est, a_est = res_b
    else:
        betas_est = res_b
        a_est = None
    #print("betas_est, a_est", betas_est, a_est)
    image_est_b, F_b, Ps_b = psf_b.deconvolve(Ds, betas_est, gamma, ret_all = True, a_est=a_est, normalize=True)

    end = time.time()
    print("PSF_basis reconstruction took: " + str(end - start))
    
    
    vmin = np.min(image)
    vmax = np.max(image)
    
    image_est = fft.ifftshift(image_est, axes=(-2, -1))
    image_est_b = fft.ifftshift(image_est_b, axes=(-2, -1))
    
    
    image_est_mean = np.zeros((nx, nx))
    image_est_b_mean = np.zeros((nx, nx))
    
    my_plot = plot.plot(nrows=max_frames + 1, ncols=7)
    my_plot.set_axis()
    
    
    for trial in np.arange(0, num_realizations):
        image_est_i = image_est[trial]
        #image_est_i = psf_basis.critical_sampling(image_est_i, arcsec_per_px, diameter, wavelength)

        image_est_b_i = image_est_b[trial]
        image_est_b_i = psf_basis.critical_sampling(image_est_b_i, arcsec_per_px, diameter, wavelength)

        image_est_mean += image_est_i
        image_est_b_mean += image_est_b_i
        if trial < max_frames:
            my_plot.colormap(image, [trial, 0], vmin=vmin, vmax=vmax)
            my_plot.colormap(Ds1[trial, 0], [trial, 1], vmin=vmin, vmax=vmax)
            my_plot.colormap(Ds1[trial, 1], [trial, 2], vmin=vmin, vmax=vmax)
            
            my_plot.colormap(image_est_i, [trial, 3], vmin=vmin, vmax=vmax)
            my_plot.colormap(image_est_b_i, [trial, 4], vmin=vmin, vmax=vmax)
            
            my_plot.colormap(np.abs(image_est_i-image), [trial, 5], vmin=vmin, vmax=vmax)
            my_plot.colormap(np.abs(image_est_b_i-image), [trial, 6], vmin=vmin, vmax=vmax)


    image_est_mean /= num_realizations
    image_est_b_mean /= num_realizations
    D_mean /= num_realizations
    D_d_mean /= num_realizations
            
    my_plot.colormap(image, [max_frames, 0], vmin=vmin, vmax=vmax)
    my_plot.colormap(D_mean, [max_frames, 1], vmin=vmin, vmax=vmax)
    my_plot.colormap(D_d_mean, [max_frames, 2], vmin=vmin, vmax=vmax)
    my_plot.colormap(image_est_mean, [max_frames, 3], vmin=vmin, vmax=vmax)
    my_plot.colormap(image_est_b_mean, [max_frames, 4], vmin=vmin, vmax=vmax)
    
    my_plot.save("estimates.png")
    my_plot.close()
    
    



if __name__ == "__main__":
    main()
