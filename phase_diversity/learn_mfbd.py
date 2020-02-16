import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import numpy.random as random
import sys
sys.setrecursionlimit(10000)
import tensorflow.keras as keras
keras.backend.set_image_data_format('channels_last')
#from keras import backend as K
import tensorflow as tf
#import tensorflow.signal as tf_signal
from tensorflow.keras.models import Model

tf.compat.v1.disable_eager_execution()

import math

import pickle
import os.path
import numpy.fft as fft

import time
#import scipy.signal as signal

jmax = 200
diameter = 100.0
wavelength = 5250.0
gamma = 1.0


# How many frames to use in training
num_frames = 100
# How many objects to use in training
num_objs = 10#None

n_epochs = 10
num_iters = 10
num_reps = 1000


#logfile = open(dir_name + '/log.txt', 'w')
#def print(*xs):
#    for x in xs:
#        logfile.write('%s' % x)
#    logfile.write("\n")
#    logfile.flush()
    
train = True
if len(sys.argv) > 1:
    if sys.argv[1].upper() == "TEST":
        train = False
        
n_test_frames = 10
if len(sys.argv) > 2:
    n_test_frames = int(sys.argv[2])

n_test_objects = 1
if len(sys.argv) > 3:
    n_test_objects = int(sys.argv[3])


if train:
    dir_name = "results" + time.strftime("%Y%m%d-%H%M%S")
    os.mkdir(dir_name)
    sys.stdout = open(dir_name + '/log.txt', 'w')
    
    f = open(dir_name + '/params.txt', 'w')
    f.write('fried num_frames_gen num_frames num_objs\n')
    f.write('%f %d %d %d' % (num_frames, num_objs) + "\n")
    f.flush()
    f.close()
    images_dir = "images_in"
    sys.path.append('../utils')
    sys.path.append('..')
    
else:
    dir_name = "."
    images_dir = "../images_in_old"

    sys.path.append('../../utils')
    sys.path.append('../..')

import config
import misc
import plot
import psf
import utils
import zernike


def load_data():
    data_file = dir_name + '/learn_mfbd_Ds.dat.npz'
    if os.path.exists(data_file):
        loaded = np.load(data_file)
        Ds = loaded['a']
        return Ds

    return None

def save_data(Ds):
    np.savez_compressed(dir_name + '/learn_mfbd_Ds.dat', a=Ds)


def load_model():
    model_file = dir_name + '/learn_mfbd_model.dat'
    if os.path.exists(model_file):
        model = tf.keras.models.load_model(model_file)
        #nn_mode = pickle.load(open(dir_name + '/learn_mfbd_params.dat', 'rb'))
        return model#, nn_mode
    return None#, None

def save_model(model):
    tf.keras.models.save_model(model, dir_name + '/learn_mfbd_model.dat')
    #with open(dir_name + '/learn_mfbd_params.dat', 'wb') as f:
    #    pickle.dump(nn_mode, f, protocol=4)


def get_params(nx):

    #arcsec_per_px = .03*(wavelength*1e-10)/(diameter*1e-2)*180/np.pi*3600
    arcsec_per_px = .25*(wavelength*1e-10)/(diameter*1e-2)*180/np.pi*3600
    print("arcsec_per_px=", arcsec_per_px)
    defocus = 2.*np.pi*100
    #defocus = (0., 0.)
    return (arcsec_per_px, defocus)



class phase_aberration_tf():
    
    def __init__(self, alphas, start_index = 3):
        self.start_index= start_index
        if len(np.shape(alphas)) == 0:
            # alphas is an integer, representing jmax
            self.create_pols(alphas)
            self.jmax = alphas
        else:
            self.create_pols(len(alphas))
            self.set_alphas(tf.constant(alphas))
            self.jmax = len(alphas)
    
    def create_pols(self, num):
        self.pols = []
        for i in np.arange(self.start_index+1, self.start_index+num+1):
            n, m = zernike.get_nm(i)
            z = zernike.zernike(n, m)
            self.pols.append(z)

    def calc_terms(self, xs):
        terms = np.zeros(np.concatenate(([len(self.pols)], np.shape(xs)[:-1])))
        i = 0
        rhos_phis = utils.cart_to_polar(xs)
        for z in self.pols:
            terms[i] = z.get_value(rhos_phis)
            i += 1
        self.terms = tf.constant(terms, dtype='float32')

    def set_alphas(self, alphas):
        if len(self.pols) != self.jmax:
            self.create_pols(self.jmax)
        self.alphas = alphas
        #self.jmax = tf.shape(self.alphas).eval()[0]
    
            
    def __call__(self):
        #vals = np.zeros(tf.shape(self.terms)[1:])
        #for i in np.arange(0, len(self.terms)):
        #    vals += self.terms[i] * self.alphas[i]
        nx = self.terms.shape[1]
        alphas = tf.tile(tf.reshape(self.alphas, [self.jmax, 1, 1]), multiples=[1, nx, nx])
        #alphas1 = tf.complex(alphas1, tf.zeros((self.jmax, nx, nx)))
        vals = tf.math.reduce_sum(tf.math.multiply(self.terms, alphas), 0)
        #vals = tf.math.reduce_sum(tf.math.multiply(self.terms, tf.reshape(self.alphas, [self.jmax, 1, 1])), 0)
        return vals
    

'''
Coherent transfer function, also called as generalized pupil function
'''
class coh_trans_func_tf():

    def __init__(self, pupil_func, phase_aberr, defocus_func = None):
        self.pupil_func = pupil_func
        self.phase_aberr = phase_aberr
        self.defocus_func = defocus_func
        
    def calc(self, xs):
        self.phase_aberr.calc_terms(xs)
        pupil = self.pupil_func(xs)
        if self.defocus_func is not None:
            defocus = self.defocus_func(xs)
        else:
            assert(False)
        self.defocus = tf.complex(tf.constant(defocus, dtype='float32'), tf.zeros((defocus.shape[0], defocus.shape[1]), dtype='float32'))
        
        self.i = tf.constant(1.j, dtype='complex64')
        self.pupil = tf.constant(pupil, dtype='float32')
        self.pupil = tf.complex(self.pupil, tf.zeros((pupil.shape[0], pupil.shape[1]), dtype='float32'))
        
    def __call__(self):
        self.phase = self.phase_aberr()
        self.phase = tf.complex(self.phase, tf.zeros((self.phase.shape[0], self.phase.shape[1]), dtype='float32'))

        focus_val = tf.math.multiply(self.pupil, tf.math.exp(tf.math.scalar_mul(self.i, self.phase)))
        defocus_val = tf.math.multiply(self.pupil, tf.math.exp(tf.math.scalar_mul(self.i, (tf.math.add(self.phase, self.defocus)))))

        #return tf.concat(tf.reshape(focus_val, [1, focus_val.shape[0], focus_val.shape[1]]), tf.reshape(defocus_val, [1, defocus_val.shape[0], defocus_val.shape[1]]), 0)
        return tf.stack([focus_val, defocus_val])

    def get_defocus_val(self, focus_val):
        return tf.math.multiply(focus_val, tf.math.exp(tf.math.scalar_mul(self.i, self.defocus)))

class psf_tf():

    '''
        diameter in centimeters
        wavelength in Angstroms
    '''
    def __init__(self, coh_trans_func, nx, arcsec_per_px, diameter, wavelength):
        self.nx= nx
        coords, rc, x_limit = utils.get_coords(nx, arcsec_per_px, diameter, wavelength)
        self.coords = coords
        x_min = np.min(self.coords, axis=(0,1))
        x_max = np.max(self.coords, axis=(0,1))
        print("psf_coords", x_min, x_max, np.shape(self.coords))
        np.testing.assert_array_almost_equal(x_min, -x_max)
        self.incoh_vals = None
        self.otf_vals = None
        self.corr = None # only for testing purposes
        self.coh_trans_func = coh_trans_func
        self.coh_trans_func.calc(self.coords)
        

        
        
    def calc(self, alphas=None):
        #self.incoh_vals = tf.zeros((2, self.nx1, self.nx1))
        #self.otf_vals = tf.zeros((2, self.nx1, self.nx1), dtype='complex')
        
        if alphas is not None:
            self.coh_trans_func.phase_aberr.set_alphas(alphas)
        coh_vals = self.coh_trans_func()
    
        vals = tf.signal.ifft2d(coh_vals)
        vals = tf.math.real(tf.multiply(vals, tf.math.conj(vals)))
        vals = tf.signal.ifftshift(vals, axes=(1, 2))
        
        vals = tf.transpose(vals, (1, 2, 0))
        #vals = np.array([utils.upsample(vals[0]), utils.upsample(vals[1])])
        # Maybe have to add channels axis first
        vals = tf.image.resize(vals, size=(tf.shape(vals)[0]*2, tf.shape(vals)[1]*2))
        vals = tf.transpose(vals, (2, 0, 1))
        # In principle there shouldn't be negative values, but ...
        #vals[vals < 0] = 0. # Set negative values to zero
        vals = tf.cast(vals, dtype='complex64')
        corr = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(vals, axes=(1, 2))), axes=(1, 2))

        #if normalize:
        #    norm = np.sum(vals, axis = (1, 2)).repeat(vals.shape[1]*vals.shape[2]).reshape((vals.shape[0], vals.shape[1], vals.shape[2]))
        #    vals /= norm
        self.incoh_vals = vals
        self.otf_vals = corr
        return self.incoh_vals


    '''
    dat_F.shape = [l, 2, nx, nx]
    alphas.shape = [l, jmax]
    '''
    def multiply(self, dat_F, alphas):

        if self.otf_vals is None:
            self.calc(alphas=alphas)
        return tf.math.multiply(dat_F, self.otf_vals)


class nn_model:

    '''
        Aberrates object with wavefront coefs
    '''
    def aberrate(self, x):
        x = tf.reshape(x, [jmax + nx*nx])
        alphas = tf.slice(x, [0], [jmax])
        obj = tf.reshape(tf.slice(x, [jmax], [nx*nx]), [nx, nx])
        
        fobj = tf.signal.fft2d(tf.complex(obj, tf.zeros((nx, nx))))
        fobj = tf.signal.fftshift(fobj)

    
        DF = self.psf.multiply(fobj, alphas)
        DF = tf.signal.ifftshift(DF, axes = (1, 2))
        D = tf.math.real(tf.signal.ifft2d(DF))
        #D = tf.signal.fftshift(D, axes = (1, 2)) # Is it needed?
        D = tf.transpose(D, (1, 2, 0))
        D = tf.reshape(D, [1, D.shape[0], D.shape[1], D.shape[2]])
                    
        return D

    '''
        Aberrates object with psf
    '''
    def aberrate_psf(self, x):
        x = tf.reshape(x, [3*nx*nx])
        psf = tf.reshape(tf.slice(x, [0], [2*nx*nx]), [2, nx, nx])
        obj = tf.reshape(tf.slice(x, [2*nx*nx], [nx*nx]), [nx, nx])
        
        fobj = tf.signal.fft2d(tf.complex(obj, tf.zeros((nx, nx))))
        fobj = tf.signal.fftshift(fobj, axes = (1, 2))

        fpsf = tf.signal.fft2d(tf.complex(psf, tf.zeros((nx, nx))))
        fpsf = tf.signal.fftshift(fpsf, axes = (1, 2))
        
        DF = tf.math.multiply(fobj, fpsf)
        DF = tf.signal.ifftshift(DF, axes = (1, 2))#, axes = (1, 2))
        D = tf.math.real(tf.signal.ifft2d(DF))

        D = tf.transpose(D, (1, 2, 0))
        D = tf.reshape(D, [1, D.shape[0], D.shape[1], D.shape[2]])
                    
        return D
    
    def create_psf(self):
        arcsec_per_px, defocus = get_params(nx)
        aperture_func = lambda xs: utils.aperture_circ(xs, coef=15, radius =1.)
        defocus_func = lambda xs: defocus*np.sum(xs*xs, axis=2)

        pa = phase_aberration_tf(jmax, start_index=0)
        ctf = coh_trans_func_tf(aperture_func, pa, defocus_func)
        self.psf = psf_tf(ctf, nx//2, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)
        
    
    def __init__(self, nx, num_frames, num_objs):
        
        self.num_frames = num_frames
        self.num_objs = num_objs
        num_channels = 2#self.num_frames*2
        image_input = keras.layers.Input((nx, nx, num_channels), name='image_input') # Channels first

        model, nn_mode_ = load_model()
        if model is None:
            print("Creating model")
            self.create_psf()
            object_input = keras.layers.Input((nx*nx), name='object_input') # Channels first
            #object_input  = keras.layers.Reshape((nx*nx))(object_input)
            ###################################################################
            # Autoencoder
            ###################################################################
            hidden_layer = keras.layers.Conv2D(32, (64, 64), activation='relu', padding='same')(image_input)#(normalized)
            hidden_layer = keras.layers.MaxPooling2D()(hidden_layer)
            hidden_layer = keras.layers.Conv2D(32, (32, 32), activation='relu', padding='same')(hidden_layer)#(normalized)
            hidden_layer = keras.layers.MaxPooling2D()(hidden_layer)
            hidden_layer = keras.layers.Conv2D(32, (16, 16), activation='relu', padding='same')(hidden_layer)#(normalized)
            hidden_layer = keras.layers.MaxPooling2D()(hidden_layer)
            hidden_layer = keras.layers.Conv2D(32, (8, 8), activation='relu', padding='same')(hidden_layer)#(normalized)
            hidden_layer = keras.layers.MaxPooling2D()(hidden_layer)
            #hidden_layer = keras.layers.Conv2D(64, (4, 4), activation='relu', padding='same')(hidden_layer)#(normalized)
            #hidden_layer = keras.layers.MaxPooling2D()(hidden_layer)
            hidden_layer = keras.layers.Flatten()(hidden_layer)
            hidden_layer = keras.layers.Dense(1000, activation='relu')(hidden_layer)
            #hidden_layer = keras.layers.Dense(500, activation='relu')(hidden_layer)
            #hidden_layer = keras.layers.Dense(jmax, activation='relu')(hidden_layer)
            #hidden_layer = keras.layers.Dense(jmax, activation='relu')(hidden_layer)
            #hidden_layer = keras.layers.Dense(jmax, activation='relu')(hidden_layer)
            #hidden_layer = keras.layers.Dense(2*jmax, activation='relu')(hidden_layer)
            hidden_layer = keras.layers.Dense(jmax, activation='linear', name='alphas_layer')(hidden_layer)
            #hidden_layer = keras.layers.Reshape((jmax))(hidden_layer)
            hidden_layer = keras.layers.concatenate([hidden_layer, object_input])
            output = keras.layers.Lambda(self.aberrate)(hidden_layer)
           
            model = keras.models.Model(inputs=[image_input, object_input], outputs=output)

        
        else:
            print("Loading model")
            self.create_psf()
            print("Mode 2")
            object_input = model.input[1]
            hidden_layer = keras.layers.concatenate([model.output, object_input])
            output = keras.layers.Lambda(self.aberrate)(hidden_layer)
            full_model = Model(inputs=model.input, outputs=output)
            model = full_model

        self.model = model
        self.model.compile(optimizer='adadelta', loss='mse')
        self.nx = nx
        self.validation_losses = []
        self.nn_mode = nn_mode_
            

    def set_data(self, Ds, objs, train_perc=.75):
        assert(self.num_frames <= Ds.shape[0])
        if self.num_objs is None or self.num_objs <= 0:
            self.num_objs = Ds.shape[1]
        assert(self.num_objs <= Ds.shape[1])
        assert(Ds.shape[2] == 2)
        Ds = Ds[:self.num_frames, :self.num_objs]
        num_objects = Ds.shape[1]
        self.Ds = np.transpose(np.reshape(Ds, (self.num_frames*num_objects, Ds.shape[2], Ds.shape[3], Ds.shape[4])), (0, 2, 3, 1))
        #self.Ds = np.reshape(Ds, (self.num_frames*num_objects, Ds.shape[2], Ds.shape[3], Ds.shape[4]))
        #self.Ds = np.zeros((num_objects, 2*self.num_frames, Ds.shape[3], Ds.shape[4]))
        #for i in np.arange(num_objects):
        #    for j in np.arange(self.num_frames):
        #        self.Ds[i, 2*j] = Ds[j, i, 0]
        #        self.Ds[i, 2*j+1] = Ds[j, i, 1]
        self.objs = objs[:self.num_objs]
        #self.objs = np.zeros((len(objs), self.nx+1, self.nx+1))
        #for i in np.arange(len(objs)):
        #    self.objs[i] = misc.sample_image(objs[i], 1.01010101)
        print("objs", self.objs.shape, self.num_objs, num_objects)
        self.objs = np.tile(self.objs, (self.num_frames, 1, 1))

        self.objs = np.reshape(self.objs, (len(self.objs), -1))
        #self.objs = np.reshape(np.tile(objs, (1, num_frames)), (num_objects*num_frames, objs.shape[1]))
                    
        # Shuffle the data
        random_indices = random.choice(len(self.Ds), size=len(self.Ds), replace=False)
        self.Ds = self.Ds[random_indices]
        self.objs = self.objs[random_indices]
        
        n_train = int(math.ceil(len(self.Ds)*train_perc))

        self.Ds_train = self.Ds[:n_train] 
        self.objs_train = self.objs[:n_train] 

        self.Ds_validation = self.Ds[n_train:]
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
        
        for epoch in np.arange(n_epochs):
            history = model.fit([self.Ds_train, self.objs_train], self.Ds_train,
                        epochs=1,
                        batch_size=1,
                        shuffle=True,
                        validation_data=([self.Ds_validation, self.objs_validation], self.Ds_validation),
                        #callbacks=[keras.callbacks.TensorBoard(log_dir='model_log')],
                        verbose=1)
            intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("alphas_layer").output)
            save_model(intermediate_layer_model)
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
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("alphas_layer").output)
        pred_alphas = intermediate_layer_model.predict([self.Ds, self.objs], batch_size=1)
        pred_Ds = model.predict([self.Ds, self.objs], batch_size=1)
        #pred_Ds = model.predict([self.Ds, self.objs], batch_size=1)
        #predicted_coefs = model.predict(Ds_train[0:n_test])
    
        arcsec_per_px, defocus = get_params(self.nx)
    
        aperture_func = lambda xs: utils.aperture_circ(xs, coef=15, radius =1.)
        defocus_func = lambda xs: defocus*np.sum(xs*xs, axis=2)
    
        pa_check = psf.phase_aberration(jmax, start_index=0)
        ctf_check = psf.coh_trans_func(aperture_func, pa_check, defocus_func)
        psf_check = psf.psf(ctf_check, self.nx//2, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)
        #self.Ds_reconstr = np.array(self.Ds_train.shape[0], 1, self.Ds_train.shape[2], self.Ds_train.shape[3])
    
        #self.Ds_reconstr = np.array(self.Ds_train.shape[0], 1, self.Ds_train.shape[2], self.Ds_train.shape[3])
        objs_test = []
        objs_reconstr = []
        i = 0
        while len(objs_test) < n_test:
            D = misc.sample_image(self.Ds[i, :, :, 0], .99)
            D_d = misc.sample_image(self.Ds[i, :, :, 1], .99)
            DF = fft.fft2(D)
            DF_d = fft.fft2(D_d)
            
            obj = np.reshape(self.objs[i], (self.nx, self.nx))
            found = False
            for obj_test in objs_test:
                if np.all(obj_test == obj):
                    found = True
                    break
            if found:
                i += 1
                continue
            objs_test.append(obj)
            
            obj_reconstr = psf_check.deconvolve(np.array([[DF, DF_d]]), alphas=np.array([pred_alphas[i]]), gamma=gamma, do_fft = True, fft_shift_before = False, ret_all=False, a_est=None, normalize = False)
            obj_reconstr = fft.ifftshift(obj_reconstr[0])
            objs_reconstr.append(obj_reconstr)


            #D1 = psf_check.convolve(image, alphas=true_coefs[frame_no])
            my_test_plot = plot.plot(nrows=3, ncols=2)
            #my_test_plot.colormap(np.reshape(self.objs[i], (self.nx+1, self.nx+1)), [0])
            #my_test_plot.colormap(np.reshape(pred_objs[i], (self.nx+1, self.nx+1)), [1])
            my_test_plot.colormap(obj, [0, 0])
            my_test_plot.colormap(obj_reconstr, [0, 1])
            my_test_plot.colormap(self.Ds[i, :, :, 0], [1, 0])
            my_test_plot.colormap(pred_Ds[i, :, :, 0], [1, 1])
            my_test_plot.colormap(self.Ds[i, :, :, 1], [2, 0])
            my_test_plot.colormap(pred_Ds[i, :, :, 1], [2, 1])
            #my_test_plot.colormap(D, [1, 0])
            #my_test_plot.colormap(D_d, [1, 1])
            #my_test_plot.colormap(D1[0, 0], [2, 0])
            #my_test_plot.colormap(D1[0, 1], [2, 1])
            my_test_plot.save(dir_name + "/train_results" + str(i) + ".png")
            my_test_plot.close()
            
            i += 1

        # Plot average reconstructions
        for i in np.arange(n_test, self.Ds.shape[0]):
            D = misc.sample_image(self.Ds[i, :, :, 0], .99)
            D_d = misc.sample_image(self.Ds[i, :, :, 1], .99)
            DF = fft.fft2(D)
            DF_d = fft.fft2(D_d)
            
            obj = np.reshape(self.objs[i], (self.nx, self.nx))
            found = False
            for j in np.arange(len(objs_test)):
                if np.all(objs_test[j] == obj):
                    found = True
                    break
            if not found:
                continue
            
            obj_reconstr = psf_check.deconvolve(np.array([[DF, DF_d]]), alphas=np.array([pred_alphas[i]]), gamma=gamma, do_fft = True, fft_shift_before = False, ret_all=False, a_est=None, normalize = False)
            obj_reconstr = fft.ifftshift(obj_reconstr[0])
            objs_reconstr[j] += obj_reconstr

        for i in np.arange(len(objs_test)):
            
            obj = objs_test[i]
            obj_reconstr = objs_reconstr[i]

            #D1 = psf_check.convolve(image, alphas=true_coefs[frame_no])
            my_test_plot = plot.plot(nrows=1, ncols=2)
            #my_test_plot.colormap(np.reshape(self.objs[i], (self.nx+1, self.nx+1)), [0])
            #my_test_plot.colormap(np.reshape(pred_objs[i], (self.nx+1, self.nx+1)), [1])
            my_test_plot.colormap(obj, [0])
            my_test_plot.colormap(obj_reconstr, [1])
            my_test_plot.save(dir_name + "/train_results_mean" + str(i) + ".png")
            my_test_plot.close()

                    
    
    def test(self):
        
        model = self.model
        
        Ds_, objs, nx_orig = gen_data(num_frames=n_test_frames, num_images=n_test_objects)
        print("test_1")
        num_frames = Ds_.shape[0]
        num_objects = Ds_.shape[1]

        Ds = np.transpose(np.reshape(Ds_, (num_frames*num_objects, Ds_.shape[2], Ds_.shape[3], Ds_.shape[4])), (0, 2, 3, 1))
        objs = objs[:num_objects]
        objs = np.tile(objs, (num_frames, 1, 1))
        objs = np.reshape(objs, (len(objs), -1))
        print("test_2")

        # Shuffle the data
        random_indices = random.choice(len(Ds), size=len(Ds), replace=False)
        Ds = Ds[random_indices]
        objs = objs[random_indices]
        print("test_3")

        #Ds = np.zeros((num_objects, 2*num_frames, Ds_.shape[3], Ds_.shape[4]))
        #for i in np.arange(num_objects):
        #    for j in np.arange(num_frames):
        #        Ds[i, 2*j] = Ds_[j, i, 0]
        #        Ds[i, 2*j+1] = Ds_[j, i, 1]

        print("test_4_2")
        start = time.time()    
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("alphas_layer").output)
        pred_alphas = intermediate_layer_model.predict([Ds, np.zeros_like(objs)], batch_size=1)
        end = time.time()
        print("Prediction time" + str(end - start))

        arcsec_per_px, defocus = get_params(self.nx)
    
        aperture_func = lambda xs: utils.aperture_circ(xs, coef=15, radius =1.)
        defocus_func = lambda xs: defocus*np.sum(xs*xs, axis=2)
    
        pa_check = psf.phase_aberration(jmax, start_index=0)
        ctf_check = psf.coh_trans_func(aperture_func, pa_check, defocus_func)
        psf_check = psf.psf(ctf_check, self.nx//2, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)

        #obj_reconstr_mean = np.zeros((self.nx-1, self.nx-1))
        DFs = np.zeros((len(objs), 2, self.nx-1, self.nx-1), dtype='complex') # in Fourier space
        for i in np.arange(len(objs)):
            D = misc.sample_image(Ds[i, :, :, 0], .99)
            D_d = misc.sample_image(Ds[i, :, :, 1], .99)
            DF = fft.fft2(D)
            DF_d = fft.fft2(D_d)
            DFs[i, 0] = DF
            DFs[i, 1] = DF_d
            
            #obj_reconstr = psf_check.deconvolve(np.array([[DF, DF_d]]), alphas=np.array([pred_alphas[i]]), gamma=gamma, do_fft = True, fft_shift_before = False, ret_all=False, a_est=None, normalize = False)
            #obj_reconstr = fft.ifftshift(obj_reconstr[0])
            #obj_reconstr_mean += obj_reconstr

        obj_reconstr = psf_check.deconvolve(DFs, alphas=pred_alphas, gamma=gamma, do_fft = True, fft_shift_before = False, ret_all=False, a_est=None, normalize = False)
        obj_reconstr = fft.ifftshift(obj_reconstr[0])
        #obj_reconstr_mean += obj_reconstr

            #my_test_plot = plot.plot(nrows=1, ncols=2)
            #my_test_plot.colormap(np.reshape(objs[i], (self.nx, self.nx)), [0])
            #my_test_plot.colormap(obj_reconstr, [1])
            #my_test_plot.save("test_results_mode" + str(nn_mode) + "_" + str(i) + ".png")
            #my_test_plot.close()
            
        my_test_plot = plot.plot(nrows=2, ncols=2)
        my_test_plot.colormap(np.reshape(objs[i], (self.nx, self.nx)), [0, 0])
        my_test_plot.colormap(obj_reconstr, [0, 1])
        my_test_plot.colormap(Ds[i, :, :, 0], [1, 0])
        my_test_plot.colormap(Ds[i, :, :, 1], [1, 1])
        my_test_plot.save(dir_name + "/test_results_mean.png")
        my_test_plot.close()
            
            
#TODO: this is just a dummy method at the moment
def read_data_from_images(num_frames, num_images = None, shuffle = True):
    image_file = None
    images, _, nx, nx_orig = utils.read_images(images_dir, image_file, is_planet = False, image_size=None, tile=True)
    images = np.asarray(images)
    if shuffle:
        random_indices = random.choice(len(images), size=len(images), replace=False)
        images = images[random_indices]
    print("nx, nx_orig", nx, nx_orig)
    if num_images is not None and len(images) > num_images:
        images = images[:num_images]

    num_objects = len(images)

    Ds = np.zeros((num_frames, num_objects, 2, nx, nx)) # in real space

    return Ds



Ds = load_data()
if Ds is None:
    Ds = read_data_from_images()
    save_data(Ds)

my_test_plot = plot.plot()
my_test_plot.colormap(Ds[0, 0, 0])
my_test_plot.save(dir_name + "/D0.png")
my_test_plot.close()

my_test_plot = plot.plot()
my_test_plot.colormap(Ds[0, 0, 1])
my_test_plot.save(dir_name + "/D0_d.png")
my_test_plot.close()

nx = Ds.shape[3]

model = nn_model(nx, num_frames, num_objs)

model.set_data(Ds, objs)

if train:
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
else:
    model.test()

#logfile.close()