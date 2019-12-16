import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import numpy.random as random
import sys
sys.setrecursionlimit(10000)
sys.path.append('../utils')
import keras
keras.backend.set_image_data_format('channels_first')

import psf
import utils

import plot

import matplotlib.pyplot as plt
import matplotlib.colorbar as cb
from matplotlib import cm
import pickle
import os.path
import numpy.fft as fft

import time
import kolmogorov

jmax = 20
diameter = 50.0
wavelength = 5250.0
gamma = 1.0

num_frames = 10
fried_param = 0.2
noise_std_perc = 0.#.01

num_sets = 10000
n_epochs = 10

def reverse_colourmap(cmap, name = 'my_cmap_r'):
     return mpl.colors.LinearSegmentedColormap(name, cm.revcmap(cmap._segmentdata))

my_cmap = reverse_colourmap(plt.get_cmap('binary'))#plt.get_cmap('winter')

class nn_model:

    def __init__(self, nx, jmax, nx_orig):
        
        print("Creating model")
    
        image_input = keras.layers.Input((2, nx, nx), name='image_input') # Channels first
    
        hidden_layer = keras.layers.convolutional.Convolution2D(32, 8, 8, subsample=(2, 2), activation='relu')(image_input)#(normalized)
        #hidden_layer = keras.layers.convolutional.Convolution2D(24, 6, 6, subsample=(2, 2), activation='relu')(image_input)#(normalized)
        hidden_layer = keras.layers.convolutional.Convolution2D(16, 4, 4, subsample=(2, 2), activation='relu')(hidden_layer)
        hidden_layer = keras.layers.core.Flatten()(hidden_layer)
        hidden_layer = keras.layers.Dense(256, activation='relu')(hidden_layer)
        output = keras.layers.Dense(jmax, activation='tanh')(hidden_layer)
        #filtered_output = keras.layers.multiply([output, actions_input])#, mode='mul')
    
        model = keras.models.Model(input=[image_input], output=output)
        #optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        model.compile(optimizer='adadelta', loss='mse')
        
    
        #model = keras.models.Model(input=coefs, output=output)
        #optimizer = keras.optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False)
        #optimizer = keras.optimizers.RMSprop(lr=0.0025, rho=0.95, epsilon=0.01)
        #model.compile(optimizer, loss='mean_absolute_error')
        #model.compile(optimizer='adadelta', loss='binary_crossentropy')
        #model.compile(optimizer=optimizer, loss='binary_crossentropy')
        #model.compile(optimizer='adadelta', loss='mean_absolute_error')
        
        self.model = model
        self.nx_orig = nx_orig


    def train(self, Ds_train, coefs_train, Ds_test, coefs_test):
        self.validation_losses = []
        model = self.model
        #print("Training")
        num_frames_train = Ds_train.shape[0]
        num_objects_train = Ds_train.shape[1]
        Ds_train = np.reshape(Ds_train, (num_frames_train*num_objects_train, Ds_train.shape[2], Ds_train.shape[3], Ds_train.shape[4]))
        jmax = coefs_train.shape[1]
        coefs_train = np.reshape(np.tile(coefs_train, (1, num_objects_train)), (num_objects_train*num_frames_train, jmax))

        self.scale_factor = np.std(coefs_train)

        if np.shape(coefs_test)[0] > 0:
            num_frames_test = Ds_test.shape[0]
            num_objects_test = Ds_test.shape[1]
            Ds_test = np.reshape(Ds_test, (num_frames_test*num_objects_test, Ds_test.shape[2], Ds_test.shape[3], Ds_test.shape[4]))
            coefs_test = np.reshape(np.tile(coefs_test, (1, num_objects_test)), (num_objects_test*num_frames_test, jmax))
            history = model.fit(Ds_train, coefs_train/self.scale_factor,
                        epochs=n_epochs,
                        batch_size=1000,
                        shuffle=True,
                        validation_data=(Ds_test, coefs_test),
                        #callbacks=[keras.callbacks.TensorBoard(log_dir='model_log')],
                        verbose=1)
            
            self.validation_losses.append(history.history['val_loss'])
            print("Average validation loss: " + str(np.mean(self.validation_losses[-10:])))
    
        else:
            history = model.fit(Ds_train, coefs_train,
                        epochs=n_epochs,
                        batch_size=1000,
                        shuffle=True,
                        #callbacks=[keras.callbacks.TensorBoard(log_dir='model_log')],
                        verbose=1)
    
        #######################################################################
        # Plot some of the training data results
        n_test = 5
        predicted_coefs = model.predict(Ds_train[0:n_test])
    
    
        arcsec_per_px, defocus = get_params(self.nx_orig)
    
        aperture_func = lambda xs: utils.aperture_circ(xs, coef=15, radius =1.)
        defocus_func = lambda xs: defocus*np.sum(xs*xs, axis=2)
    
        pa_check = psf.phase_aberration(jmax, start_index=0)
        ctf_check = psf.coh_trans_func(aperture_func, pa_check, defocus_func)
        psf_check = psf.psf(ctf_check, self.nx_orig, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)
        for i in np.arange(n_test):
            print("True coefs", coefs_train[i])
            print("Predicted coefs", predicted_coefs[i]*self.scale_factor)
            DF = fft.fft2(Ds_train[i, 0])
            DF_d = fft.fft2(Ds_train[i, 1])
            image_reconstr = psf_check.deconvolve(np.array([[DF, DF_d]]), alphas=np.array([predicted_coefs[i]*self.scale_factor]), gamma=gamma, do_fft = True, fft_shift_before = False, ret_all=False, a_est=None, normalize = False)
            image_true = psf_check.deconvolve(np.array([[DF, DF_d]]), alphas=np.array([coefs[i]]), gamma=gamma, do_fft = True, fft_shift_before = False, ret_all=False, a_est=None, normalize = False)
            #D1 = psf_check.convolve(image, alphas=true_coefs[frame_no])
            my_test_plot = plot.plot(nrows=1, ncols=2)
            my_test_plot.colormap(fft.ifftshift(image_true[0]), [0])
            my_test_plot.colormap(fft.ifftshift(image_reconstr[0]), [1])
            #my_test_plot.colormap(D, [1, 0])
            #my_test_plot.colormap(D_d, [1, 1])
            #my_test_plot.colormap(D1[0, 0], [2, 0])
            #my_test_plot.colormap(D1[0, 1], [2, 1])
            my_test_plot.save("train_results" + str(i) + ".png")
            my_test_plot.close()
    
        #my_plot = plot.plot()
        #my_plot.plot(np.arange(jmax), coefs, params="r-")
        #my_plot.plot(np.arange(jmax), predicted_coefs, params="b-")
        #my_plot.save("learn_wf_results.png")
        #my_plot.close()
        #######################################################################
            
    
    def test(self):
        
        model = self.model
        n_test = 5
        
        Ds, DFs, coefs, nx_orig = gen_data(num_frames=n_test)
    
        start = time.time()    
        predicted_coefs = model.predict(np.reshape(Ds, (Ds.shape[0]*Ds.shape[1], Ds.shape[2], Ds.shape[3], Ds.shape[4])))
        end = time.time()
        print("Prediction time" + str(end - start))

        
        arcsec_per_px, defocus = get_params(self.nx_orig)
    
        aperture_func = lambda xs: utils.aperture_circ(xs, coef=15, radius =1.)
        defocus_func = lambda xs: defocus*np.sum(xs*xs, axis=2)
    
        pa_check = psf.phase_aberration(jmax, start_index=0)
        ctf_check = psf.coh_trans_func(aperture_func, pa_check, defocus_func)
        psf_check = psf.psf(ctf_check, self.nx_orig, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)

        for i in np.arange(n_test):
            print("True coefs", coefs_train[i])
            print("Predicted coefs", predicted_coefs[i]*self.scale_factor)
            DF = DFs[i, 0]
            DF_d = DFs[i, 1]
            image_reconstr = psf_check.deconvolve(np.array([[DF, DF_d]]), alphas=np.array([predicted_coefs[i]*self.scale_factor]), gamma=gamma, do_fft = True, fft_shift_before = False, ret_all=False, a_est=None, normalize = False)
            image_true = psf_check.deconvolve(np.array([[DF, DF_d]]), alphas=np.array([coefs[i]]), gamma=gamma, do_fft = True, fft_shift_before = False, ret_all=False, a_est=None, normalize = False)
            #D1 = psf_check.convolve(image, alphas=true_coefs[frame_no])
            my_test_plot = plot.plot(nrows=1, ncols=2)
            my_test_plot.colormap(fft.ifftshift(image_true[0]), [0])
            my_test_plot.colormap(fft.ifftshift(image_reconstr[0]), [1])
            #my_test_plot.colormap(D, [1, 0])
            #my_test_plot.colormap(D_d, [1, 1])
            #my_test_plot.colormap(D1[0, 0], [2, 0])
            #my_test_plot.colormap(D1[0, 1], [2, 1])
            my_test_plot.save("train_results" + str(i) + ".png")
            my_test_plot.close()
    
            #my_plot = plot.plot()
            #my_plot.plot(np.arange(jmax), coefs, params="r-")
            #my_plot.plot(np.arange(jmax), predicted_coefs, params="b-")
            #my_plot.save("learn_wf_results.png")
            #my_plot.close()
        


def get_params(nx):

    arcsec_per_px = .25*(wavelength*1e-10)/(diameter*1e-2)*180/np.pi*3600
    print("arcsec_per_px=", arcsec_per_px)
    defocus = 2.*np.pi*100
    #defocus = (0., 0.)
    return (arcsec_per_px, defocus)

def gen_data(num_frames, num_images = None):
    image_file = 'icont'
    dir = "images"
    images, _, nx, nx_orig = utils.read_images(dir, image_file, is_planet = False, image_size=50, tile=True)
    print("nx, nx_orig", nx, nx_orig)
    if num_images is not None and len(images) > num_images:
        images = images[:num_images]

    arcsec_per_px, defocus = get_params(nx_orig)

    aperture_func = lambda xs: utils.aperture_circ(xs, coef=15, radius =1.)
    defocus_func = lambda xs: defocus*np.sum(xs*xs, axis=2)

    coords, _, _ = utils.get_coords(nx_orig, arcsec_per_px, diameter, wavelength)

    num_objects = len(images)

    Ds = np.zeros((num_frames, num_objects, 2, nx, nx)) # in real space
    DFs = np.zeros((num_frames, num_objects, 2, nx, nx), dtype='complex') # in Fourier space
    true_coefs = np.zeros((num_frames, jmax))
    pa = psf.phase_aberration(jmax, start_index=0)
    pa.calc_terms(coords)
    #wavefront = kolmogorov.kolmogorov(fried = np.array([fried_param]), num_realizations=num_frames, size=4*nx_orig, sampling=1.)
    #pa = psf.phase_aberration(np.random.normal(size=jmax))
    for frame_no in np.arange(num_frames):
        pa_true = psf.phase_aberration(np.minimum(np.maximum(np.random.normal(size=jmax)*50, -50), 50), start_index=0)
        #ctf_true = psf.coh_trans_func(aperture_func, psf.wavefront(wavefront[0,frame_no,:,:]), defocus_func)
        ctf_true = psf.coh_trans_func(aperture_func, pa_true, defocus_func)
        #print("wavefront", np.max(wavefront[0,frame_no,:,:]), np.min(wavefront[0,frame_no,:,:]))
        #true_coefs[frame_no] = ctf_true.dot(pa)
        true_coefs[frame_no] = pa_true.alphas
        #true_coefs[frame_no] -= np.mean(true_coefs[frame_no])
        #true_coefs[frame_no] /= np.std(true_coefs[frame_no])
        psf_true = psf.psf(ctf_true, nx_orig, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)

        #######################################################################
        # Just checking if true_coefs are calculated correctly
        pa_check = psf.phase_aberration(jmax, start_index=0)
        ctf_check = psf.coh_trans_func(aperture_func, pa_check, defocus_func)
        psf_check = psf.psf(ctf_check, nx_orig, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)
        #######################################################################
        for obj_no in np.arange(num_objects):
            images[obj_no] = psf.critical_sampling(images[obj_no], arcsec_per_px, diameter, wavelength)
            image = images[obj_no]
            image -= np.mean(image)
            image /= np.std(image)
            #my_test_plot = plot.plot()
            #my_test_plot.colormap(image)
            #my_test_plot.save("critical_sampling" + str(frame_no) + " " + str(obj_no) + ".png")
            #my_test_plot.close()
            fimage = fft.fft2(images[obj_no])
            fimage = fft.fftshift(fimage)
    
        
            DFs1 = psf_true.multiply(fimage)
            DF = DFs1[0, 0]
            DF_d = DFs1[0, 1]
            
            if noise_std_perc > 0.:
                print("np.mean(image)", np.mean(image), np.min(image), np.max(image))
                noise = np.random.poisson(lam=noise_std_perc*np.std(image), size=(nx, nx))
                fnoise = fft.fft2(noise)
                fnoise = fft.fftshift(fnoise)
        
                noise_d = np.random.poisson(lam=noise_std_perc*np.std(image), size=(nx, nx))
                fnoise_d = fft.fft2(noise_d)
                fnoise_d = fft.fftshift(fnoise_d)
        
                DF += fnoise
                DF_d += fnoise_d
        
            DF = fft.ifftshift(DF)
            DF_d = fft.ifftshift(DF_d)
        
            D = fft.ifft2(DF).real
            D_d = fft.ifft2(DF_d).real

            ###################################################################
            # Just checking if true_coefs are calculated correctly
            if frame_no < 10:
                image_reconstr = psf_check.deconvolve(np.array([[DF, DF_d]]), alphas=np.array([true_coefs[frame_no]]), gamma=gamma, do_fft = True, fft_shift_before = False, ret_all=False, a_est=None, normalize = False)
                D1 = psf_check.convolve(image, alphas=true_coefs[frame_no])
                my_test_plot = plot.plot(nrows=3, ncols=2)
                my_test_plot.colormap(image, [0, 0])
                my_test_plot.colormap(fft.ifftshift(image_reconstr[0]), [0, 1])
                my_test_plot.colormap(D, [1, 0])
                my_test_plot.colormap(D_d, [1, 1])
                my_test_plot.colormap(D1[0, 0], [2, 0])
                my_test_plot.colormap(D1[0, 1], [2, 1])
                my_test_plot.save("check" + str(frame_no) + ".png")
                my_test_plot.close()
            ###################################################################

            Ds[frame_no, obj_no, 0] = D
            Ds[frame_no, obj_no, 1] = D_d

            DFs[frame_no, obj_no, 0] = DF
            DFs[frame_no, obj_no, 1] = DF_d


    return Ds, DFs, true_coefs, nx_orig


def load_data():
    data_file = 'learn_wavefront_data.pkl'
    if load_data and os.path.isfile(data_file):
        return pickle.load(open(data_file, 'rb'))
    else:
        return None

def save_data(data):
    with open('learn_wavefront_data.pkl', 'wb') as f:
        pickle.dump(data, f)



data = load_data()
if data is None:
    print("Generating training data")
    Ds, DFs, coefs, nx_orig = gen_data(num_frames)
    save_data((Ds, DFs, coefs, nx_orig))
else:
    Ds, DFs, coefs, nx_orig = data

my_test_plot = plot.plot()
my_test_plot.colormap(Ds[0, 0, 0])
my_test_plot.save("D0.png")
my_test_plot.close()

my_test_plot = plot.plot()
my_test_plot.colormap(Ds[0, 0, 1])
my_test_plot.save("D0_d.png")
my_test_plot.close()

num_objs = Ds.shape[1]
nx = Ds.shape[3]


model = nn_model(nx, jmax, nx_orig)

n_train = int(num_frames*.75)

num_reps = 3
for rep in np.arange(0, num_reps):
    print("Rep no: " + str(rep))

    Ds_train = Ds[:n_train] 
    Ds_test = Ds[n_train:]
    DFs_train = DFs[:n_train] 
    DFs_test = DFs[n_train:]
    coefs_train = coefs[:n_train] 
    coefs_test = coefs[n_train:]
    
    if rep == num_reps-1:
        # In the laast iteration train on the full set
        model.train(Ds, coefs, np.array([]), np.array([]), np.array([]))
    else:
        model.train(Ds_train, coefs_train, Ds_test, coefs_test)

    model.test()
