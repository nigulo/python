import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import numpy.random as random
import sys
sys.setrecursionlimit(10000)
sys.path.append('../utils')
import keras
keras.backend.set_image_data_format('channels_first')
from keras import backend as K
import TensorFlow as tf

import psf
import utils
import math
import misc

import plot

import matplotlib.pyplot as plt
import matplotlib.colorbar as cb
from matplotlib import cm
import pickle
import os.path
import numpy.fft as fft

import time
import kolmogorov
import zernike

jmax = 200
diameter = 100.0
wavelength = 5250.0
gamma = 1.0

num_frames_gen = 10
num_frames = 10
fried_param = 0.1
noise_std_perc = 0.#.01

n_epochs = 10
num_iters = 10
num_reps = 20

MODE_1 = 1
MODE_2 = 2
nn_mode = MODE_1

def reverse_colourmap(cmap, name = 'my_cmap_r'):
     return mpl.colors.LinearSegmentedColormap(name, cm.revcmap(cmap._segmentdata))

my_cmap = reverse_colourmap(plt.get_cmap('binary'))#plt.get_cmap('winter')


class phase_aberration():
    
    def __init__(self, alphas, start_index = 3):
        self.start_index= start_index
        if len(tf.shape(alphas)) == 0:
            # alphas is an integer, representing jmax
            self.create_pols(alphas)
            self.jmax = alphas
        else:
            self.create_pols(tf.shape(alphas)[0])
            self.set_alphas(alphas)
            self.jmax = tf.shape(alphas)[0]
    
    def create_pols(self, num):
        pols = np.zeros(num)
        for i in np.arange(self.start_index+1, self.start_index+num+1):
            n, m = zernike.get_nm(i)
            z = zernike.zernike(n, m)
            pols[i-self.start_index-1] = z
        self.pols = tf.constant(pols)

    def calc_terms(self, xs):
        self.terms = np.zeros(np.concatenate(([len(self.pols)], np.shape(xs)[:-1])))
        i = 0
        rhos_phis = utils.cart_to_polar(xs)
        for z in self.pols:
            self.terms[i] = z.get_value(rhos_phis)
            i += 1

    def set_alphas(self, alphas):
        if len(self.pols) != tf.shape(alphas)[0]:
            self.create_pols(tf.shape(alphas)[0])
        self.alphas = alphas
        self.jmax = tf.shape(self.alphas)[0]
    
            
    def __call__(self):
        vals = np.zeros(np.shape(self.terms)[1:])
        for i in np.arange(0, len(self.terms)):
            vals += self.terms[i] * self.alphas[i]
        return vals
    

'''
Coherent transfer function, also called as generalized pupil function
'''
class coh_trans_func():

    def __init__(self, pupil_func, phase_aberr, defocus_func = None):
        self.pupil_func = pupil_func
        self.phase_aberr = phase_aberr
        self.defocus_func = defocus_func
        
    def calc(self, xs):
        self.phase_aberr.calc_terms(xs)
        self.pupil = self.pupil_func(xs)
        if self.defocus_func is not None:
            self.defocus = self.defocus_func(xs)
        else:
            self.defocus = 0.
        
    def __call__(self):
        self.phase = self.phase_aberr()

        return np.array([self.pupil*np.exp(1.j * self.phase), self.pupil*np.exp(1.j * (self.phase + self.defocus))])


class psf():

    '''
        diameter in centimeters
        wavelength in Angstroms
    '''
    def __init__(self, coh_trans_func, nx, arcsec_per_px, diameter, wavelength, corr_or_fft=True):
        self.nx= nx
        coords, rc, x_limit = utils.get_coords(nx, arcsec_per_px, diameter, wavelength)
        self.coords = coords
        x_min = np.min(self.coords, axis=(0,1))
        x_max = np.max(self.coords, axis=(0,1))
        print("psf_coords", x_min, x_max, np.shape(self.coords))
        np.testing.assert_array_almost_equal(x_min, -x_max)
        self.incoh_vals = dict()
        self.otf_vals = dict()
        self.corr = dict() # only for testing purposes
        self.coh_trans_func = coh_trans_func
        self.coh_trans_func.calc(self.coords)
        
        # Repeat the same for bigger grid
        xs1 = np.linspace(-x_limit, x_limit, self.nx*2-1)
        coords1 = np.dstack(np.meshgrid(xs1, xs1)[::-1])
        self.coords1 = coords1

        self.nx1 = self.nx * 2 - 1

        self.corr_or_fft = corr_or_fft
        
        
    def calc(self, alphas=None, normalize = True):
        l = tf.shape(alphas)[0]
        self.incoh_vals = tf.zeros((l, 2, self.nx1, self.nx1))
        self.otf_vals = tf.zeros((l, 2, self.nx1, self.nx1), dtype='complex')
        
        for i in np.arange(0, l):
            if alphas is not None:
                self.coh_trans_func.phase_aberr.set_alphas(alphas[i])
            coh_vals = self.coh_trans_func()
        
            if self.corr_or_fft:
                #corr = signal.correlate2d(coh_vals, coh_vals, mode='full')/(self.nx*self.nx)
                corr = signal.fftconvolve(coh_vals, coh_vals[:, ::-1, ::-1].conj(), mode='full', axes=(-2, -1))/(self.nx*self.nx)
                vals = fft.fftshift(fft.ifft2(fft.ifftshift(corr, axes=(-2, -1))), axes=(-2, -1)).real
            else:
                vals = fft.ifft2(coh_vals, axes=(-2, -1))
                vals = (vals*vals.conjugate()).real
                vals = fft.ifftshift(vals, axes=(-2, -1))
                vals = np.array([utils.upsample(vals[0]), utils.upsample(vals[1])])
                # In principle there shouldn't be negative values, but ...
                vals[vals < 0] = 0. # Set negative values to zero
                corr = fft.fftshift(fft.fft2(fft.ifftshift(vals, axes=(-2, -1))), axes=(-2, -1))
    
            if normalize:
                norm = np.sum(vals, axis = (1, 2)).repeat(vals.shape[1]*vals.shape[2]).reshape((vals.shape[0], vals.shape[1], vals.shape[2]))
                vals /= norm
            self.incoh_vals[i] = vals
            self.otf_vals[i] = corr
        return self.incoh_vals

    '''
    dat_F.shape = [l, 2, nx, nx]
    alphas.shape = [l, jmax]
    '''
    def multiply(self, dat_F, alphas):

        if len(self.otf_vals) == 0:
            self.calc(alphas=alphas)
        return tf.multiply(dat_F, self.otf_vals)


class nn_model:

    def __init__(self, nx, num_frames):
        
        print("Creating model")
    
        self.num_frames = num_frames
        num_channels = 2#self.num_frames*2
        image_input = keras.layers.Input((num_channels, nx, nx), name='image_input') # Channels first
    
        if nn_mode == MODE_1:
            ###################################################################
            # Representation in a higher dim space
            ###################################################################
            #hidden_layer = keras.layers.convolutional.Convolution2D(32, 8, 8, subsample=(2, 2), activation='relu')(image_input)#(normalized)
            hidden_layer = keras.layers.convolutional.Conv2D(16, (64, 64), activation='relu', padding='same', subsample=(4, 4))(image_input)#(normalized)
            #hidden_layer = keras.layers.convolutional.Conv2D(16, (nx//4, nx//4), padding='same', activation='linear')(hidden_layer)#(normalized)
            #hidden_layer = keras.layers.Lambda(lambda x:K.mean(x, axis=0))(hidden_layer)
            #hidden_layer = keras.layers.UpSampling2D((2, 2))(hidden_layer)
            hidden_layer = keras.layers.core.Flatten()(hidden_layer)
            #output = keras.layers.core.Flatten()(hidden_layer)
            output = keras.layers.Dense(nx*nx, activation='linear')(hidden_layer)
            #output = keras.layers.Dense(nx*nx, activation='linear')(hidden_layer)
            #output = keras.layers.Reshape((nx, nx))(hidden_layer)
            #output = keras.layers.convolutional.Conv2D(64, (8, 8), activation='relu')(image_input)#(normalized)
            #output = keras.layers.add(hidden_layer)(image_input)#(normalized)
        else:
            
            def aberrate(x):
                return x
                
            object_input = keras.layers.Input((1, nx, nx), name='object_input') # Channels first
            ###################################################################
            # Autoencoder
            ###################################################################
            hidden_layer = keras.layers.convolutional.Conv2D(16, (64, 64), activation='relu', padding='same', subsample=(2, 2))(image_input)#(normalized)
            hidden_layer = keras.layers.convolutional.Conv2D(16, (32, 32), activation='relu', padding='same', subsample=(2, 2))(hidden_layer)#(normalized)
            hidden_layer = keras.layers.convolutional.Conv2D(16, (16, 16), activation='relu', padding='same', subsample=(2, 2))(hidden_layer)#(normalized)
            hidden_layer = keras.layers.convolutional.Conv2D(16, (8, 8), activation='relu', padding='same', subsample=(2, 2))(hidden_layer)#(normalized)
            hidden_layer = keras.layers.convolutional.Conv2D(16, (4, 4), activation='relu', padding='same', subsample=(2, 2))(hidden_layer)#(normalized)
            hidden_layer = keras.layers.core.Flatten()(hidden_layer)
            hidden_layer = keras.layers.Dense(jmax, activation='relu')(hidden_layer)
            hidden_layer = keras.layers.concatenate([hidden_layer, object_input])
            output = keras.layers.Lambda(aberrate)(hidden_layer)
           
            
    
        model = keras.models.Model(inputs=image_input, outputs=output)
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
        self.nx = nx
        self.validation_losses = []
        

    def set_data(self, Ds, objs, train_perc=.75):
        assert(self.num_frames <= Ds.shape[0])
        num_objects = Ds.shape[1]
        assert(Ds.shape[2] == 2)
        Ds = Ds[:self.num_frames]
        self.Ds = np.reshape(Ds, (self.num_frames*num_objects, Ds.shape[2], Ds.shape[3], Ds.shape[4]))
        #self.Ds = np.zeros((num_objects, 2*self.num_frames, Ds.shape[3], Ds.shape[4]))
        #for i in np.arange(num_objects):
        #    for j in np.arange(self.num_frames):
        #        self.Ds[i, 2*j] = Ds[j, i, 0]
        #        self.Ds[i, 2*j+1] = Ds[j, i, 1]
        self.objs = np.asarray(objs)
        #self.objs = np.zeros((len(objs), self.nx+1, self.nx+1))
        #for i in np.arange(len(objs)):
        #    self.objs[i] = misc.sample_image(objs[i], 1.01010101)
        print("objs", self.objs.shape)
        self.objs = np.tile(self.objs, (self.num_frames, 1, 1))

        self.objs = np.reshape(self.objs, (len(self.objs), -1))
        #self.objs = np.reshape(np.tile(objs, (1, num_frames)), (num_objects*num_frames, objs.shape[1]))
                      
        n_train = int(math.ceil(len(self.Ds)*train_perc))
        self.Ds_train = self.Ds[:n_train] 
        self.Ds_validation = self.Ds[n_train:]
        self.objs_train = self.objs[:n_train] 
        self.objs_validation = self.objs[n_train:]

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

        print(self.Ds_train.shape, self.objs_train.shape, self.Ds_validation.shape, self.objs_validation.shape)
        #if not full:
        history = model.fit(self.Ds_train, self.objs_train,
                    epochs=n_epochs,
                    batch_size=1,
                    shuffle=True,
                    validation_data=(self.Ds_validation, self.objs_validation),
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
        pred_objs = model.predict(self.Ds)
        #predicted_coefs = model.predict(Ds_train[0:n_test])
    
        #self.Ds_reconstr = np.array(self.Ds_train.shape[0], 1, self.Ds_train.shape[2], self.Ds_train.shape[3])
        for i in np.arange(self.Ds.shape[0]):
            if i < n_test:
                #D1 = psf_check.convolve(image, alphas=true_coefs[frame_no])
                my_test_plot = plot.plot(nrows=1, ncols=2)
                #my_test_plot.colormap(np.reshape(self.objs[i], (self.nx+1, self.nx+1)), [0])
                #my_test_plot.colormap(np.reshape(pred_objs[i], (self.nx+1, self.nx+1)), [1])
                my_test_plot.colormap(np.reshape(self.objs[i], (self.nx, self.nx)), [0])
                my_test_plot.colormap(np.reshape(pred_objs[i], (self.nx, self.nx)), [1])
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
        
        Ds_, objs, nx_orig = gen_data(num_frames=n_test)
        num_frames = Ds_.shape[0]
        num_objects = Ds_.shape[1]
        Ds = np.reshape(Ds_, (num_frames*num_objects, Ds_.shape[2], Ds_.shape[3], Ds_.shape[4]))
        #Ds = np.zeros((num_objects, 2*num_frames, Ds_.shape[3], Ds_.shape[4]))
        #for i in np.arange(num_objects):
        #    for j in np.arange(num_frames):
        #        Ds[i, 2*j] = Ds_[j, i, 0]
        #        Ds[i, 2*j+1] = Ds_[j, i, 1]

        start = time.time()    
        pred_objs = model.predict(Ds)
        end = time.time()
        print("Prediction time" + str(end - start))

        for i in np.arange(n_test):
            #D1 = psf_check.convolve(image, alphas=true_coefs[frame_no])
            my_test_plot = plot.plot(nrows=1, ncols=2)
            my_test_plot.colormap(np.reshape(objs[i], (self.nx, self.nx)), [0])
            my_test_plot.colormap(np.reshape(pred_objs[i], (self.nx, self.nx)), [1])
            #my_test_plot.colormap(np.reshape(pred_objs[i], (self.nx+1, self.nx+1)), [1])
            #my_test_plot.colormap(D, [1, 0])
            #my_test_plot.colormap(D_d, [1, 1])
            #my_test_plot.colormap(D1[0, 0], [2, 0])
            #my_test_plot.colormap(D1[0, 1], [2, 1])
            my_test_plot.save("test_results" + str(i) + ".png")
            my_test_plot.close()



def get_params(nx):

    #arcsec_per_px = .03*(wavelength*1e-10)/(diameter*1e-2)*180/np.pi*3600
    arcsec_per_px = .25*(wavelength*1e-10)/(diameter*1e-2)*180/np.pi*3600
    print("arcsec_per_px=", arcsec_per_px)
    defocus = 2.*np.pi*100
    #defocus = (0., 0.)
    return (arcsec_per_px, defocus)

def gen_data(num_frames, num_images = None):
    image_file = None
    dir = "images_in"
    images, _, nx, nx_orig = utils.read_images(dir, image_file, is_planet = False, image_size=None, tile=True)
    print("nx, nx_orig", nx, nx_orig)
    if num_images is not None and len(images) > num_images:
        images = images[:num_images]

    arcsec_per_px, defocus = get_params(nx_orig)

    aperture_func = lambda xs: utils.aperture_circ(xs, coef=15, radius =1.)
    defocus_func = lambda xs: defocus*np.sum(xs*xs, axis=2)

    coords, _, _ = utils.get_coords(nx//2, arcsec_per_px, diameter, wavelength)

    num_objects = len(images)

    Ds = np.zeros((num_frames, num_objects, 2, nx, nx)) # in real space
    #true_coefs = np.zeros((num_frames, jmax))
    pa = psf.phase_aberration(jmax, start_index=0)
    pa.calc_terms(coords)
    wavefront = kolmogorov.kolmogorov(fried = np.array([fried_param]), num_realizations=num_frames, size=4*nx//2, sampling=1.)
    #pa = psf.phase_aberration(np.random.normal(size=jmax))
    for frame_no in np.arange(num_frames):
        #pa_true = psf.phase_aberration(np.minimum(np.maximum(np.random.normal(size=jmax)*10, -25), 25), start_index=0)
        #pa_true = psf.phase_aberration(np.minimum(np.maximum(np.random.normal(size=jmax)*25, -25), 25), start_index=0)
        ctf_true = psf.coh_trans_func(aperture_func, psf.wavefront(wavefront[0,frame_no,:,:]), defocus_func)
        #ctf_true = psf.coh_trans_func(aperture_func, pa_true, defocus_func)
        #print("wavefront", np.max(wavefront[0,frame_no,:,:]), np.min(wavefront[0,frame_no,:,:]))
        #true_coefs[frame_no] = ctf_true.dot(pa)
        
        #true_coefs[frame_no] = pa_true.alphas
        #true_coefs[frame_no] -= np.mean(true_coefs[frame_no])
        #true_coefs[frame_no] /= np.std(true_coefs[frame_no])
        psf_true = psf.psf(ctf_true, nx//2, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)

        #######################################################################
        # Just checking if true_coefs are calculated correctly
        pa_check = psf.phase_aberration(jmax, start_index=0)
        ctf_check = psf.coh_trans_func(aperture_func, pa_check, defocus_func)
        psf_check = psf.psf(ctf_check, nx//2, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)
        #######################################################################
        for obj_no in np.arange(num_objects):
            # Omit for now (just technical issues)
            #images[obj_no] = psf.critical_sampling(images[obj_no], arcsec_per_px, diameter, wavelength)
            image = images[obj_no]
            image -= np.mean(image)
            image /= np.std(image)
            #my_test_plot = plot.plot()
            #my_test_plot.colormap(image)
            #my_test_plot.save("critical_sampling" + str(frame_no) + " " + str(obj_no) + ".png")
            #my_test_plot.close()
            fimage = fft.fft2(misc.sample_image(image, .99))
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
            if frame_no < 1 and obj_no < 5:
                my_test_plot = plot.plot(nrows=1, ncols=3)
                my_test_plot.colormap(image, [0])
                my_test_plot.colormap(D, [1])
                my_test_plot.colormap(D_d, [2])
                my_test_plot.save("check" + str(frame_no) + "_" + str(obj_no) + ".png")
                my_test_plot.close()
            ###################################################################

            Ds[frame_no, obj_no, 0] = misc.sample_image(D, 1.01010101)
            Ds[frame_no, obj_no, 1] = misc.sample_image(D_d, 1.01010101)


    return Ds, images, nx_orig


'''
def load_data():
    data_file = 'learn_object_data.pkl'
    if load_data and os.path.isfile(data_file):
        return pickle.load(open(data_file, 'rb'))
    else:
        return None

def save_data(data):
    with open('learn_object_data.pkl', 'wb') as f:
        pickle.dump(data, f, protocol=4)
'''

def load_data():
    data_file = 'learn_object_Ds.dat'
    if os.path.isfile(data_file):
        Ds = np.load(data_file)
        data_file = 'learn_object_objs.dat'
        if os.path.isfile(data_file):
            objs = np.load(data_file)
        return Ds, objs
    else:
        return None, None

def save_data(Ds, objects):
    with open('learn_object_Ds.dat', 'wb') as f:
        np.save(f, Ds)
    with open('learn_object_objs.dat', 'wb') as f:
        np.save(f, objs)


Ds, objs = load_data()
if Ds is None:
    Ds, objs, nx_orig = gen_data(num_frames_gen)
    save_data(Ds, objs)

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

# Shuffle the data
random_indices = random.choice(len(Ds), size=len(Ds), replace=False)
Ds = Ds[random_indices]
objs = objs[random_indices]

model = nn_model(nx, num_frames)

model.set_data(Ds, objs)

for rep in np.arange(0, num_reps):
    print("Rep no: " + str(rep))

    if rep == num_reps-1:
        # In the laast iteration train on the full set
        model.train(full=True)
    else:
        model.train()

    model.test()

    #if np.mean(model.validation_losses[-10:]) > np.mean(model.validation_losses[-20:-10]):
    #    break
    model.validation_losses = model.validation_losses[-20:]