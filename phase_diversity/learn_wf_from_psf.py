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

jmax = 200
diameter = 100.0
wavelength = 5250.0
gamma = 1.0

num_frames = 10000
fried_param = 0.2
noise_std_perc = 0.#.01

n_epochs = 100
num_iters = 10
num_reps = 20

def reverse_colourmap(cmap, name = 'my_cmap_r'):
     return mpl.colors.LinearSegmentedColormap(name, cm.revcmap(cmap._segmentdata))

my_cmap = reverse_colourmap(plt.get_cmap('binary'))#plt.get_cmap('winter')

class nn_model:

    def __init__(self, nx, jmax, nx_orig):
        
        print("Creating model")
    
        image_input = keras.layers.Input((2, nx, nx), name='image_input') # Channels first
    
        #hidden_layer = keras.layers.convolutional.Convolution2D(32, 8, 8, subsample=(2, 2), activation='relu')(image_input)#(normalized)
        #hidden_layer = keras.layers.convolutional.Conv2D(64, (8, 8), subsample=(2, 2), activation='relu')(image_input)#(normalized)
        #hidden_layer = keras.layers.UpSampling2D((2, 2))(hidden_layer)
        hidden_layer = keras.layers.MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None)(image_input)
        #hidden_layer = keras.layers.UpSampling2D((2, 2))(hidden_layer)
        #hidden_layer = keras.layers.convolutional.Convolution2D(24, 6, 6, subsample=(2, 2), activation='relu')(image_input)#(normalized)
        #hidden_layer = keras.layers.convolutional.Conv2D(16, (4, 4), subsample=(2, 2), activation='relu')(hidden_layer)
        hidden_layer = keras.layers.core.Flatten()(hidden_layer)
        hidden_layer = keras.layers.Dense(512, activation='relu')(hidden_layer)
        hidden_layer = keras.layers.Dense(512, activation='relu')(hidden_layer)
        hidden_layer = keras.layers.Dense(512, activation='relu')(hidden_layer)
        hidden_layer = keras.layers.Dense(512, activation='relu')(hidden_layer)
        hidden_layer = keras.layers.Dense(512, activation='relu')(hidden_layer)
        hidden_layer = keras.layers.Dense(512, activation='relu')(hidden_layer)
        #hidden_layer = keras.layers.Dense(256, activation='relu')(hidden_layer)
        hidden_layer = keras.layers.Dense(256, activation='relu')(hidden_layer)
        hidden_layer = keras.layers.Dense(256, activation='relu')(hidden_layer)
        hidden_layer = keras.layers.Dense(256, activation='relu')(hidden_layer)
        #hidden_layer = keras.layers.Dense(256, activation='relu')(hidden_layer)
        #hidden_layer = keras.layers.Dense(256, activation='relu')(hidden_layer)
        #hidden_layer = keras.layers.Dense(256, activation='relu')(hidden_layer)
        #hidden_layer = keras.layers.Dense(256, activation='tanh')(hidden_layer)
        output = keras.layers.Dense(jmax, activation='linear')(hidden_layer)
        #filtered_output = keras.layers.multiply([output, actions_input])#, mode='mul')
    
        model = keras.models.Model(input=[image_input], output=output)
        #optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        model.compile(optimizer='adadelta', loss='mse')
        
    
        #model = keras.models.Model(input=coefs, output=output)
        #optimizer = keras.optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False)
        #optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        #model.compile(optimizer, loss='mse')
        #model.compile(optimizer='adadelta', loss='binary_crossentropy')
        #model.compile(optimizer=optimizer, loss='binary_crossentropy')
        #model.compile(optimizer='adadelta', loss='mean_absolute_error')
        
        self.model = model
        self.validation_losses = []
        self.nx = nx
        self.nx_orig = nx_orig
        
    def set_data(self, Ds, coefs, train_perc=.75):
        print("Ds", Ds.shape)
        self.Ds = Ds
        self.coefs = coefs
        
        self.scale_factor = np.std(coefs)
        self.coefs /= self.scale_factor
        
        n_train = int(len(self.Ds)*train_perc)
        self.Ds_train = self.Ds[:n_train] 
        self.Ds_validation = self.Ds[n_train:]
        self.coefs_train = self.coefs[:n_train] 
        self.coefs_validation = self.coefs[n_train:]

        #num_frames_train = Ds_train.shape[0]
        #num_objects_train = Ds_train.shape[1]
        #self.Ds_train = np.reshape(Ds_train, (num_frames_train*num_objects_train, Ds_train.shape[2], Ds_train.shape[3], Ds_train.shape[4]))
        #self.coefs_train = np.reshape(np.tile(coefs_train, (1, num_objects_train)), (num_objects_train*num_frames_train, jmax))

        #self.coefs_train /= self.scale_factor

        #num_frames_validation = Ds_validation.shape[0]
        #num_objects_validation = Ds_validation.shape[1]
        #self.Ds_validation = np.reshape(Ds_validation, (num_frames_validation*num_objects_validation, Ds_validation.shape[2], Ds_validation.shape[3], Ds_validation.shape[4]))
        #self.coefs_validation = np.reshape(np.tile(coefs_validation, (1, num_objects_validation)), (num_objects_validation*num_frames_validation, jmax))

        #self.coefs_validation /= self.scale_factor
        

    def train(self, full=False):
        model = self.model

        #if not full:
        history = model.fit(self.Ds_train, self.coefs_train,
                    epochs=n_epochs,
                    batch_size=32,
                    shuffle=True,
                    validation_data=(self.Ds_validation, self.coefs_validation),
                    #callbacks=[keras.callbacks.TensorBoard(log_dir='model_log')],
                    verbose=1)
        
        self.validation_losses.append(history.history['val_loss'])
        print("Average validation loss: " + str(np.mean(self.validation_losses[-10:])))
    
        #else:
        #    history = model.fit(self.Ds, self.coefs,
        #                epochs=n_epochs,
        #                batch_size=1000,
        #                shuffle=True,
        #                #callbacks=[keras.callbacks.TensorBoard(log_dir='model_log')],
        #                verbose=1)
    
        #######################################################################
        # Plot some of the training data results
        n_test = 5
        predicted_coefs = model.predict(self.Ds)
        #predicted_coefs = model.predict(Ds_train[0:n_test])
    
    
        arcsec_per_px, defocus = get_params(self.nx_orig)

        aperture_func = lambda xs: utils.aperture_circ(xs, coef=15, radius =1.)
        defocus_func = lambda xs: defocus*np.sum(xs*xs, axis=2)

        for i in np.arange(self.Ds.shape[0]):

            if i < n_test:
                print("True coefs", self.coefs[i])
                print("Predicted coefs", predicted_coefs[i]*self.scale_factor)

                pa_reconstr = psf.phase_aberration(predicted_coefs[i]*self.scale_factor, start_index=0)
                ctf = psf.coh_trans_func(aperture_func, pa_reconstr, defocus_func)#np.sum(u**2))
                Ds_reconstr = psf.psf(ctf, self.nx_orig, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength).calc()

                my_test_plot = plot.plot(nrows=2, ncols=2)
                my_test_plot.colormap(self.Ds[i, 0], [0, 0])
                my_test_plot.colormap(self.Ds[i, 1], [0, 1])
                my_test_plot.colormap(Ds_reconstr[0, 0], [1, 0])
                my_test_plot.colormap(Ds_reconstr[0, 1], [1, 1])
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
        
        Ds, coefs, _ = gen_data(num_frames=n_test)
    
        start = time.time()    
        predicted_coefs = model.predict(Ds)
        end = time.time()
        print("Prediction time" + str(end - start))

        
        arcsec_per_px, defocus = get_params(self.nx_orig)

        aperture_func = lambda xs: utils.aperture_circ(xs, coef=15, radius =1.)
        defocus_func = lambda xs: defocus*np.sum(xs*xs, axis=2)

        for i in np.arange(Ds.shape[0]):

            if i < n_test:
                print("True coefs", self.coefs[i])
                print("Predicted coefs", predicted_coefs[i]*self.scale_factor)

                pa_reconstr = psf.phase_aberration(predicted_coefs[i]*self.scale_factor, start_index=0)
                ctf = psf.coh_trans_func(aperture_func, pa_reconstr, defocus_func)#np.sum(u**2))
                Ds_reconstr = psf.psf(ctf, self.nx_orig, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength).calc()

                my_test_plot = plot.plot(nrows=2, ncols=2)
                my_test_plot.colormap(Ds[i, 0], [0, 0])
                my_test_plot.colormap(Ds[i, 1], [0, 1])
                my_test_plot.colormap(Ds_reconstr[0, 0], [1, 0])
                my_test_plot.colormap(Ds_reconstr[0, 1], [1, 1])
                #my_test_plot.colormap(D, [1, 0])
                #my_test_plot.colormap(D_d, [1, 1])
                #my_test_plot.colormap(D1[0, 0], [2, 0])
                #my_test_plot.colormap(D1[0, 1], [2, 1])
                my_test_plot.save("test_results" + str(i) + ".png")
                my_test_plot.close()
                
    
        


def get_params(nx):

    arcsec_per_px = .03*(wavelength*1e-10)/(diameter*1e-2)*180/np.pi*3600
    print("arcsec_per_px=", arcsec_per_px)
    defocus = 2.*np.pi
    #defocus = (0., 0.)
    return (arcsec_per_px, defocus)

def gen_data(num_frames, num_images = None):

    nx_orig = 50
    nx = 99
    arcsec_per_px, defocus = get_params(nx)

    aperture_func = lambda xs: utils.aperture_circ(xs, coef=15, radius =1.)
    defocus_func = lambda xs: defocus*np.sum(xs*xs, axis=2)

    Ds = np.zeros((num_frames, 2, nx, nx)) # in real space
    true_coefs = np.zeros((num_frames, jmax))

    for frame_no in np.arange(num_frames):
        #coefs = np.zeros(jmax)
        coefs = np.random.normal(size=jmax)#*np.linspace(.1, 10., 100)**2
        pa_true = psf.phase_aberration(coefs, start_index=0)#size=jmax), start_index=0)
        ctf = psf.coh_trans_func(aperture_func, pa_true, defocus_func)#np.sum(u**2))
        psf_vals = psf.psf(ctf, nx_orig, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength).calc()
        
        Ds[frame_no] = psf_vals
        true_coefs[frame_no] = pa_true.alphas

        ###################################################################
        if frame_no < 10:
            my_test_plot = plot.plot(nrows=1, ncols=2)
            my_test_plot.colormap(Ds[frame_no, 0], [0])
            my_test_plot.colormap(Ds[frame_no, 1], [1])
            my_test_plot.save("psf" + str(frame_no) + ".png")
            my_test_plot.close()
        ###################################################################



    return Ds, true_coefs, nx_orig


def load_data():
    data_file = 'learn_wf_from_psf_data.pkl'
    if load_data and os.path.isfile(data_file):
        return pickle.load(open(data_file, 'rb'))
    else:
        return None

def save_data(data):
    with open('learn_wf_from_psf_data.pkl', 'wb') as f:
        pickle.dump(data, f)



data = load_data()
if data is None:
    print("Generating training data")
    Ds, coefs, nx_orig = gen_data(num_frames)
    save_data((Ds, coefs, nx_orig))
else:
    Ds, coefs, nx_orig = data

nx = Ds.shape[2]
assert(nx == Ds.shape[3])

model = nn_model(nx, jmax, nx_orig)

model.set_data(Ds[:1000], coefs[:1000])

for rep in np.arange(0, num_reps):
    print("Rep no: " + str(rep))

    if rep == num_reps-1:
        # In the laast iteration train on the full set
        model.train(full=True)
    else:
        model.train()

    model.test()

    #if np.mean(model.validation_losses[-10:]) > np.mean(model.validation_losses[-20:-10]):
    #    print("Validation loss increasing, stopping training.")
    #    break
    model.validation_losses = model.validation_losses[-20:]