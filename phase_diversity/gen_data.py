import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import numpy.random as random
sys.setrecursionlimit(10000)
import tensorflow.keras as keras
keras.backend.set_image_data_format('channels_last')
#from keras import backend as K
import tensorflow as tf
#import tensorflow.signal as tf_signal
from tensorflow.keras.models import Model

tf.compat.v1.disable_eager_execution()

sys.path.append('../utils')
sys.path.append('..')

from tqdm import tqdm
import config
import misc
import plot
import psf
import utils
import kolmogorov
import zernike

import gen_images

import pickle
import os.path
import numpy.fft as fft

import time
#import scipy.signal as signal

in_dir = "images"
out_dir = "data_out"
image_file = None#"icont"
image_size = 96
tile=False
scale=1.0

jmax = 44
diameter = 100.0
wavelength = 5250.0
gamma = 1.0

# How many frames to generate per object
num_frames = 20
# How many objects to use in training
num_objs = 100#None

new_frames_after_every_obj = 2


fried_param = .1
noise_std_perc = 0.#.01

fried_or_zernike_aberr = True


def save_data(Ds, objects, pupil, modes, diversity, zernike_coefs):
    np.savez_compressed(out_dir + '/Ds', Ds=Ds, objs=objects, pupil=pupil, modes=modes, diversity=diversity, zernike_coefs=zernike_coefs)


def get_params(nx):

    #arcsec_per_px = .03*(wavelength*1e-10)/(diameter*1e-2)*180/np.pi*3600
    arcsec_per_px = .25*(wavelength*1e-10)/(diameter*1e-2)*180/np.pi*3600
    print("arcsec_per_px=", arcsec_per_px)
    defocus = 2.*np.pi*1000
    #defocus = (0., 0.)
    return (arcsec_per_px, defocus)


def gen_data(images, num_frames, num_images = None, shuffle = True, new_frames_after_every_obj=2):
    nx = images[0].shape[0]
    images = np.asarray(images)
    if shuffle:
        random_indices = random.choice(len(images), size=len(images), replace=False)
        images = images[random_indices]
    print(f"nx = {nx}")
    if num_images is not None and len(images) > num_images:
        images = images[:num_images]

    arcsec_per_px, defocus = get_params(nx)
    coords, _, _ = utils.get_coords(nx, arcsec_per_px, diameter, wavelength)

    aperture_func = lambda xs: utils.aperture_circ(xs, coef=150, radius = np.max(coords))
    defocus_func = lambda xs: defocus*np.sum(xs*xs, axis=2)

    num_objects = len(images)
    new_frames_after_every_obj = min(num_objects, new_frames_after_every_obj)

    Ds = np.zeros((num_objects, num_frames, 2, nx, nx)) # in real space
    #true_coefs = np.zeros((num_frames, jmax))
    if fried_or_zernike_aberr:
        wavefront = kolmogorov.kolmogorov(fried = np.array([fried_param]), num_realizations=num_frames*num_objects//new_frames_after_every_obj, size=4*nx, sampling=1.)
    DFs = np.zeros((num_objects, num_frames, 2, 2*nx-1, 2*nx-1), dtype='complex')

    pa = psf.phase_aberration(jmax, start_index=0)
    pa.calc_terms(nx=nx)
    
    ctf_tmp = psf.coh_trans_func(aperture_func, phase_aberr=None, defocus_func=defocus_func)
    ctf_tmp.calc(coords)
    modes = pa.get_pol_values()
    pupil = ctf_tmp.get_pupil()
    diversity = ctf_tmp.get_defocus()

    for mode_no in np.arange(len(modes)):
        my_test_plot = plot.plot(nrows=1, ncols=1)
        my_test_plot.colormap(modes[mode_no], [0], show_colorbar=True, colorbar_prec=2)
        my_test_plot.save(f"{out_dir}/modes{mode_no}.png")
        my_test_plot.close()

    my_test_plot = plot.plot(nrows=1, ncols=1)
    my_test_plot.colormap(pupil, [0], show_colorbar=True, colorbar_prec=2)
    my_test_plot.save(out_dir + "/pupil.png")
    my_test_plot.close()

    frame_no_start = 0
    for obj_no in tqdm(np.arange(num_objects)):
        
        image = images[obj_no]
        # Omit for now (just technical issues)
        image = misc.sample_image(psf.critical_sampling(misc.sample_image(image, .99), arcsec_per_px, diameter, wavelength), 1.01010101)
        image -= np.mean(image)
        image /= np.std(image)
        images[obj_no] = image
        
        if obj_no % new_frames_after_every_obj == 0:
            psf_true = []
        
            for frame_no in np.arange(num_frames):
                #pa_true = psf.phase_aberration(np.minimum(np.maximum(np.random.normal(size=jmax)*10, -25), 25), start_index=0)
                
                zernike_coefs = np.zeros((num_frames, jmax))
                if fried_or_zernike_aberr:
                    ctf_true = psf.coh_trans_func(aperture_func, psf.wavefront(wavefront[0,frame_no_start+frame_no,:,:]), defocus_func)
                    zernike_coefs[frame_no] = ctf_true.dot(pa)
                else:
                    zernike_coefs[frame_no] = np.random.normal(size=jmax)*500
                    pa_true = psf.phase_aberration(zernike_coefs[frame_no], start_index=0)
                    ctf_true = psf.coh_trans_func(aperture_func, pa_true, defocus_func)
                #print("wavefront", np.max(wavefront[0,frame_no,:,:]), np.min(wavefront[0,frame_no,:,:]))
                #true_coefs[frame_no] = ctf_true.dot(pa)
                
                #true_coefs[frame_no] = pa_true.alphas
                #true_coefs[frame_no] -= np.mean(true_coefs[frame_no])
                #true_coefs[frame_no] /= np.std(true_coefs[frame_no])
                psf_true.append(psf.psf(ctf_true, nx, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength))
                #print(np.max(coords), np.min(coords))
                #zernike_coefs[frame_no] = np.random.normal(size=(jmax))*np.linspace(1, .10, jmax)
                #zernike_coefs[frame_no][25] = -1.
                #zernike_coefs[frame_no] = ctf_true.dot(pa)
                #print(zernike_coefs[frame_no])
                #######################################################################
                # Plot the wavefront
                if False:#fried_or_zernike_aberr:
                    pa_check = psf.phase_aberration(zernike_coefs[frame_no], start_index=0)
                    pa_check.calc_terms(nx=nx)
                    my_test_plot = plot.plot(nrows=1, ncols=3)
                    my_test_plot.colormap(wavefront[0,frame_no,:,:], [0], show_colorbar=True, colorbar_prec=2)
                    my_test_plot.colormap(pa_check(), [1])
                    my_test_plot.colormap(np.abs(wavefront[0,frame_no,:,:] - pa_check()), [2])
                    my_test_plot.save(f"{out_dir}/pa{frame_no_start}_{frame_no}.png")
                    my_test_plot.close()
                #######################################################################
        
                pa_check = psf.phase_aberration(jmax, start_index=0)
                ctf_check = psf.coh_trans_func(aperture_func, pa_check, defocus_func)
                psf_check = psf.psf(ctf_check, nx, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)
        
                #######################################################################
                # Just checking if true_coefs are calculated correctly
                #pa_check = psf.phase_aberration(jmax, start_index=0)
                #ctf_check = psf.coh_trans_func(aperture_func, pa_check, defocus_func)
                #psf_check = psf.psf(ctf_check, nx//2, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)
                #######################################################################
            frame_no_start += num_frames

        for frame_no in np.arange(num_frames):
            
            #my_test_plot = plot.plot()
            #my_test_plot.colormap(image)
            #my_test_plot.save("critical_sampling" + str(frame_no) + " " + str(obj_no) + ".png")
            #my_test_plot.close()
            
            
            #image1 = misc.sample_image(image, .99)
            #image1 -= np.mean(image)
            #image1 /= np.std(image)
            
            image1 = utils.upsample(image)

            D_D_d = psf_true[frame_no].convolve(image1)

            D = D_D_d[0, 0]
            D_d = D_D_d[0, 1]

            #fimage = fft.fft2(misc.sample_image(image, .99))
            #fimage = fft.fftshift(fimage)
    
        
            #DFs1 = psf_true.multiply(fimage)
            #DF = DFs1[0, 0]
            #DF_d = DFs1[0, 1]
            
            #DF = fft.ifftshift(DF)
            #DF_d = fft.ifftshift(DF_d)
        
            #D = fft.ifft2(DF).real
            #D_d = fft.ifft2(DF_d).real

            if noise_std_perc > 0.:
                noise = np.random.poisson(lam=noise_std_perc*np.std(D), size=(nx-1, nx-1))
                noise_d = np.random.poisson(lam=noise_std_perc*np.std(D_d), size=(nx-1, nx-1))

                D += noise
                D_d += noise_d

            ###################################################################
            # Just checking if true_coefs are calculated correctly
            #if frame_no < 5 and obj_no < 5:
            #    my_test_plot = plot.plot(nrows=1, ncols=4)
            #    my_test_plot.colormap(image, [0], show_colorbar=True, colorbar_prec=2)
            #    my_test_plot.colormap(D, [1])
            #    my_test_plot.colormap(D_d, [2])
            #    my_test_plot.save(dir_name + "/check" + str(frame_no) + "_" + str(obj_no) + ".png")
            #    my_test_plot.close()
            ###################################################################
            #D -= np.mean(D)
            #D_d -= np.mean(D_d)
            #D /= np.std(D)
            #D_d /= np.std(D_d)

            DFs[obj_no, frame_no, 0] = fft.fft2(D)
            DFs[obj_no, frame_no, 1] = fft.fft2(D_d)

            D = misc.sample_image(D, 0.5)
            D_d = misc.sample_image(D_d, 0.5)

            Ds[obj_no, frame_no, 0] = D#misc.sample_image(D, 1.01010101)
            Ds[obj_no, frame_no, 1] = D_d#misc.sample_image(D_d, 1.01010101)


    for obj_no in np.arange(min(5, num_objects)):
        my_test_plot = plot.plot(nrows=1, ncols=4)
        my_test_plot.colormap(images[obj_no], [0], show_colorbar=True, colorbar_prec=2)
        my_test_plot.colormap(Ds[obj_no, 0, 0], [1])
        my_test_plot.colormap(Ds[obj_no, 0, 1], [2])
        ###############################################################
    
        obj_reconstr = psf_check.deconvolve(DFs[obj_no, :, :, :, :], alphas=zernike_coefs, gamma=gamma, do_fft = True, fft_shift_before = False, ret_all=False, a_est=None, normalize = False)
        obj_reconstr = fft.ifftshift(obj_reconstr)
        my_test_plot.colormap(obj_reconstr, [3])

        my_test_plot.save(out_dir + "/check_reconstr" + str(obj_no) + ".png")
        my_test_plot.close()


    return Ds, images, pupil, modes, diversity, zernike_coefs



if __name__ == '__main__':
    
    num_angles = 10
    num_subimages = num_objs//num_angles

    images = gen_images.gen_images(in_dir, None, image_file, image_size, tile, scale, num_subimages, num_angles, ret=True)
    Ds, images, pupil, modes, diversity, zernike_coefs = gen_data(images, num_frames, num_images=num_objs)
    save_data(Ds, images, pupil, modes, diversity, zernike_coefs)

    for obj_no in np.arange(min(5, num_objs)):
        my_test_plot = plot.plot(nrows=1, ncols=3)
        my_test_plot.colormap(images[obj_no], [0], show_colorbar=True, colorbar_prec=2)
        my_test_plot.colormap(Ds[obj_no, 0, 0], [1])
        my_test_plot.colormap(Ds[obj_no, 0, 1], [2])
        ###############################################################
    
        my_test_plot.save(out_dir + "/check" + str(obj_no) + ".png")
        my_test_plot.close()
