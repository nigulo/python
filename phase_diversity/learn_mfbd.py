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

import math

import pickle
import os.path
import numpy.fft as fft

import time
#import scipy.signal as signal

jmax = 50
diameter = 100.0
wavelength = 5250.0
gamma = 1.0

# How many frames to generate per object
num_frames_gen = 100

# How many frames to use in training
num_frames = 100
# How many objects to use in training
num_objs = 10#None

# How many frames of the same object are sent to NN input
# Must be power of 2
num_frames_input = 8

fried_param = .1
noise_std_perc = 0.#.01

n_epochs = 10
num_iters = 10
num_reps = 1000
shuffle = True

MODE_1 = 1 # aberrated images --> wavefront coefs --> MFBD loss
MODE_2 = 2 # aberrated images --> wavefront coefs --> object (using MFBD formula) --> aberrated images
nn_mode = MODE_1

batch_size = 8
n_channels = 128

#logfile = open(dir_name + '/log.txt', 'w')
#def print(*xs):
#    for x in xs:
#        logfile.write('%s' % x)
#    logfile.write("\n")
#    logfile.flush()
    

dir_name = None
if len(sys.argv) > 1:
    dir_name = sys.argv[1]

train = True
if len(sys.argv) > 2:
    if sys.argv[2].upper() == "TEST":
        train = False
        
n_test_frames = 10
if len(sys.argv) > 3:
    n_test_frames = int(sys.argv[3])

n_test_objects = 1
if len(sys.argv) > 4:
    n_test_objects = int(sys.argv[4])


if dir_name is None:
    dir_name = "results" + time.strftime("%Y%m%d-%H%M%S")
    os.mkdir(dir_name)
    
    f = open(dir_name + '/params.txt', 'w')
    f.write('fried jmax num_frames_gen num_frames num_objs nn_mode\n')
    f.write('%f %d %d %d %d %d' % (fried_param, jmax, num_frames_gen, num_frames, num_objs, nn_mode) + "\n")
    f.flush()
    f.close()

images_dir_train = "images_in"
images_dir_test = "images_in"#images_in_test"

sys.path.append('../utils')
sys.path.append('..')

if train:
    sys.stdout = open(dir_name + '/log.txt', 'a')
    
#else:
#    dir_name = "."
#    images_dir = "../images_in_old"

#    sys.path.append('../../utils')
#    sys.path.append('../..')


import config
import misc
import plot
import psf
import psf_tf
import utils
import kolmogorov
import zernike


def load_data():
    data_file = dir_name + '/learn_object_Ds.dat.npz'
    if os.path.exists(data_file):
        loaded = np.load(data_file)
        Ds = loaded['a']
        objs = loaded['b']
        return Ds, objs

    data_file = dir_name + '/learn_object_Ds.dat'
    if os.path.exists(data_file):
        Ds = np.load(data_file)
        data_file = dir_name + '/learn_object_objs.dat'
        if os.path.exists(data_file):
            objs = np.load(data_file)
        return Ds, objs

    return None, None

def save_data(Ds, objects):
    np.savez_compressed(dir_name + '/learn_object_Ds.dat', a=Ds, b=objects)
    #with open(dir_name + '/learn_object_Ds.dat', 'wb') as f:
    #    np.save(f, Ds)
    #with open(dir_name + '/learn_object_objs.dat', 'wb') as f:
    #    np.save(f, objs)


def load_model():
    model_file = dir_name + '/learn_object_model.h5'
    if not os.path.exists(model_file):
        model_file = dir_name + '/learn_object_model.dat' # Just an old file suffix
    if os.path.exists(model_file):
        model = tf.keras.models.load_model(model_file)
        nn_mode = pickle.load(open(dir_name + '/learn_object_params.dat', 'rb'))
        return model, nn_mode
    return None, None

def save_model(model):
    tf.keras.models.save_model(model, dir_name + '/learn_object_model.h5')
    with open(dir_name + '/learn_object_params.dat', 'wb') as f:
        pickle.dump(nn_mode, f, protocol=4)


def load_weights(model):
    model_file = dir_name + '/learn_object_weights.h5'
    if os.path.exists(model_file):
        model.load_weights(model_file)
        nn_mode = pickle.load(open(dir_name + '/learn_object_params.dat', 'rb'))
        return nn_mode
    return None

def save_weights(model):
    model.save_weights(dir_name + '/learn_object_weights.h5')
    with open(dir_name + '/learn_object_params.dat', 'wb') as f:
        pickle.dump(nn_mode, f, protocol=4)


def get_params(nx):

    #arcsec_per_px = .03*(wavelength*1e-10)/(diameter*1e-2)*180/np.pi*3600
    arcsec_per_px = .25*(wavelength*1e-10)/(diameter*1e-2)*180/np.pi*3600
    print("arcsec_per_px=", arcsec_per_px)
    defocus = 2.*np.pi*1000
    #defocus = (0., 0.)
    return (arcsec_per_px, defocus)


def convert_data(Ds_in):            
    num_frames = Ds_in.shape[0]
    num_objects = Ds_in.shape[1]
    Ds_out = np.zeros(((num_frames-num_frames_input+1)*num_objects, Ds.shape[3], Ds.shape[4], Ds.shape[2]*num_frames_input))
    k = 0
    l = 0
    for i in np.arange(num_objects):
        for j in np.arange(num_frames):
            Ds_out[k, :, :, 2*l] = Ds_in[j, i, 0, :, :]
            Ds_out[k, :, :, 2*l+1] = Ds_in[j, i, 1, :, :]
            l += 1
            if l >= num_frames_input:
                l = 0
                k += 1
    Ds_out = Ds_out[:k]
    #assert(k == (num_frames-num_frames_input+1)*num_objects)
    return Ds_out, num_frames-num_frames_input+1

class nn_model:

    
    def create_psf(self):
        arcsec_per_px, defocus = get_params(nx)
        aperture_func = lambda xs: utils.aperture_circ(xs, coef=15, radius =1.)
        defocus_func = lambda xs: defocus*np.sum(xs*xs, axis=2)

        pa = psf_tf.phase_aberration_tf(jmax, start_index=0)
        ctf = psf_tf.coh_trans_func_tf(aperture_func, pa, defocus_func)
        self.psf = psf_tf.psf_tf(ctf, nx, arcsec_per_px=arcsec_per_px, diameter=diameter, wavelength=wavelength, num_frames=num_frames_input, batch_size=batch_size)
        
    
    
    def __init__(self, nx, num_frames, num_objs):
        
        self.num_frames = num_frames
        assert(num_frames_input <= self.num_frames)
        self.num_objs = num_objs
        num_defocus_channels = 2#self.num_frames*2
        image_input = keras.layers.Input((nx, nx, num_defocus_channels*num_frames_input), name='image_input') # Channels first

        model, nn_mode_ = load_model()
        
        if model is None:
            print("Creating model")
            nn_mode_ = nn_mode
                
            def tile(a, num):
                return tf.tile(a, [1, 1, 1, num])
                        
            def resize(x):
                #vals = tf.transpose(x, (1, 2, 0))
                vals = tf.image.resize(x, size=(25, 25))
                #vals = tf.transpose(vals, (2, 0, 1))
                return vals
            
            def untile(a, num):
                a1 = tf.slice(a, [0, 0, 0, 0], [1, tf.shape(a)[1], tf.shape(a)[2], num])                    
                a2 = tf.slice(a, [0, 0, 0, num], [1, tf.shape(a)[1], tf.shape(a)[2], num])
                return tf.add(a1, a2)
            
            
            def multiply(x, num):
                return tf.math.scalar_mul(tf.constant(num, dtype="float32"), x)


            def conv_layer(x, n_channels, kernel=(3, 3), max_pooling=True, batch_normalization=True, num_convs=3):
                for i in np.arange(num_convs):
                    x1 = keras.layers.Conv2D(n_channels, (1, 1), activation='linear', padding='same')(x)#(normalized)
                    x2 = keras.layers.Conv2D(n_channels, kernel, activation='relu', padding='same')(x)#(normalized)
                    x = keras.layers.add([x2, x1])#tf.keras.backend.tile(image_input, [1, 1, 1, 16])])
                    if batch_normalization:
                        x = keras.layers.BatchNormalization()(x)
                if max_pooling:
                    x = keras.layers.MaxPooling2D()(x)
                return x
                

            self.create_psf()
            
            hidden_layer = conv_layer(image_input, n_channels)
            hidden_layer = conv_layer(hidden_layer, 2*n_channels)
            hidden_layer = conv_layer(hidden_layer, 4*n_channels)
            hidden_layer = conv_layer(hidden_layer, 4*n_channels)
            hidden_layer = conv_layer(hidden_layer, 4*n_channels)

            hidden_layer = keras.layers.Flatten()(hidden_layer)
            hidden_layer = keras.layers.Dense(36*n_channels, activation='relu')(hidden_layer)
            #hidden_layer = keras.layers.Dense(1000, activation='relu')(hidden_layer)
            #hidden_layer = keras.layers.Dense(1000, activation='relu')(hidden_layer)
            
            #alphas_layer = keras.layers.Dense(500, activation='relu', name='tmp1')(hidden_layer)
            #obj_layer = keras.layers.Dense(500, activation='relu', name='tmp2')(hidden_layer)

            alphas_layer = hidden_layer
                            
            #alphas_layer = keras.layers.Flatten()(alphas_layer)
            #obj_layer = keras.layers.Flatten()(obj_layer)
            #alphas_layer = keras.layers.Dense(256, activation='relu')(alphas_layer)
            #alphas_layer = keras.layers.Dense(128, activation='relu')(alphas_layer)

            #alphas_layer1 = keras.layers.Reshape((6, 6, 256))(alphas_layer)
            #alphas_layer = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(alphas_layer1)#(normalized)
            #alphas_layer = keras.layers.add([alphas_layer, alphas_layer1])
            #alphas_layer = keras.layers.MaxPooling2D()(alphas_layer)
            alphas_layer = keras.layers.Flatten()(alphas_layer)
            #alphas_layer = keras.layers.Dense(2048, activation='relu')(alphas_layer)
            alphas_layer = keras.layers.Dense(1024, activation='relu')(alphas_layer)
            #alphas_layer = keras.layers.Dense(512, activation='relu')(alphas_layer)
            #alphas_layer = keras.layers.Dense(256, activation='relu')(alphas_layer)
            #alphas_layer = keras.layers.Dense(128, activation='relu')(alphas_layer)
            alphas_layer = keras.layers.Dense(jmax*num_frames_input, activation='linear', name='alphas_layer')(alphas_layer)
            #alphas_layer = keras.layers.Lambda(lambda x : multiply(x, 100.), name='alphas_layer')(alphas_layer)
            
            #obj_layer = keras.layers.Dense(256)(obj_layer)
            #obj_layer = keras.layers.Dense(128)(obj_layer)
            #obj_layer = keras.layers.Dense(64)(obj_layer)
            #obj_layer = keras.layers.Dense(128)(obj_layer)
            #obj_layer = keras.layers.Dense(256)(obj_layer)
            #obj_layer = keras.layers.Dense(512)(obj_layer)
            #obj_layer = keras.layers.Dense(1152)(obj_layer)
           
            
            if nn_mode == MODE_1:
                hidden_layer = keras.layers.concatenate([tf.reshape(alphas_layer, [batch_size*jmax*num_frames_input]), tf.reshape(image_input, [batch_size*num_frames_input*2*nx*nx])])
                output = keras.layers.Lambda(self.psf.mfbd_loss)(hidden_layer)
                #output = keras.layers.Lambda(lambda x: tf.reshape(tf.math.reduce_sum(x), [1]))(output)
                #output = keras.layers.Flatten()(output)
                #output = keras.layers.Lambda(lambda x: tf.math.reduce_sum(x))(output)
            elif nn_mode == MODE_2:
                hidden_layer = keras.layers.concatenate([tf.reshape(alphas_layer, [batch_size*jmax*num_frames_input]), tf.reshape(image_input, [batch_size*num_frames_input*2*nx*nx])])
                output = keras.layers.Lambda(self.psf.deconvolve_aberrate)(hidden_layer)

            else:
                assert(False)
           
            model = keras.models.Model(inputs=image_input, outputs=output)

            nn_mode_ = load_weights(model)
            if nn_mode_ is not None:
                assert(nn_mode_ == nn_mode) # Model was saved with in mode
            else:
                nn_mode_ = nn_mode

           #optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
            
        
            #model = keras.models.Model(input=coefs, output=output)
            #optimizer = keras.optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False)
            #optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
            #model.compile(optimizer, loss='mse')
            #model.compile(optimizer='adadelta', loss='binary_crossentropy')
            #model.compile(optimizer=optimizer, loss='binary_crossentropy')
            #model.compile(optimizer='adadelta', loss='mean_absolute_error')
        else:
            print("Loading model")
            self.create_psf()
            print("Mode 2")
            object_input = model.input[1]
            hidden_layer = keras.layers.concatenate([model.output, object_input])
            output = keras.layers.Lambda(self.psf.aberrate)(hidden_layer)
            full_model = Model(inputs=model.input, outputs=output)
            model = full_model

            
        self.model = model
        
        def mfbd_loss(y_true, y_pred):
            return tf.reduce_sum(tf.subtract(y_true, y_pred))
            
        #self.model.compile(optimizer='adadelta', loss=mfbd_loss)#'mse')
        self.model.compile(optimizer='adam', loss=mfbd_loss)#'mse')
        self.nx = nx
        self.validation_losses = []
        self.nn_mode = nn_mode_


    def set_data(self, Ds, objs, train_perc=.75):
        assert(self.num_frames <= Ds.shape[0])
        if self.num_objs is None or self.num_objs <= 0:
            self.num_objs = Ds.shape[1]
        assert(self.num_objs <= Ds.shape[1])
        assert(Ds.shape[2] == 2)
        if shuffle:
            i1 = random.randint(0, Ds.shape[0] + 1 - self.num_frames)
            i2 = random.randint(0, Ds.shape[1] + 1 - self.num_objs)
        else:
            i1 = 0
            i2 = 0
        Ds = Ds[i1:i1+self.num_frames, i2:i2+self.num_objs]
        self.objs = objs[i2:i2+self.num_objs]
        num_objects = Ds.shape[1]
        self.Ds, num_frames = convert_data(Ds)
        
        #self.Ds = np.transpose(np.reshape(Ds, (self.num_frames*num_objects, Ds.shape[2], Ds.shape[3], Ds.shape[4])), (0, 2, 3, 1))
        #
        #self.Ds = np.reshape(Ds, (self.num_frames*num_objects, Ds.shape[2], Ds.shape[3], Ds.shape[4]))
        #self.Ds = np.zeros((num_objects, 2*self.num_frames, Ds.shape[3], Ds.shape[4]))
        #for i in np.arange(num_objects):
        #    for j in np.arange(self.num_frames):
        #        self.Ds[i, 2*j] = Ds[j, i, 0]
        #        self.Ds[i, 2*j+1] = Ds[j, i, 1]
        #self.objs = np.zeros((len(objs), self.nx+1, self.nx+1))
        #for i in np.arange(len(objs)):
        #    self.objs[i] = misc.sample_image(objs[i], 1.01010101)
        print("objs", self.objs.shape, self.num_objs, num_objects)
        self.objs = np.tile(self.objs, (num_frames, 1, 1))

        self.objs = np.reshape(self.objs, (len(self.objs), -1))
        #self.objs = np.reshape(np.tile(objs, (1, num_frames)), (num_objects*num_frames, objs.shape[1]))
                    
        # Shuffle the data
        random_indices = random.choice(len(self.Ds), size=len(self.Ds), replace=False)
        self.Ds = self.Ds[random_indices]
        self.objs = self.objs[random_indices]
        
        #for i in np.arange(len(self.Ds)):
        #    my_plot = plot.plot(nrows=self.Ds.shape[3]//2, ncols=2)
        #    for j in np.arange(self.Ds.shape[3]//2):
        #        my_plot.colormap(self.Ds[i, :, :, 2*j], [j, 0], show_colorbar=True, colorbar_prec=2)
        #        my_plot.colormap(self.Ds[i, :, :, 2*j+1], [j, 1], show_colorbar=True, colorbar_prec=2)
    
        #    my_plot.save(dir_name + "/Ds" + str(i) + ".png")
        #    my_plot.close()
                
        n_train = int(math.ceil(len(self.Ds)*train_perc))

        n_train -= n_train % batch_size
        
        n_validation = len(self.Ds) - n_train
        n_validation -= n_validation % batch_size

        self.Ds_train = self.Ds[:n_train]
        self.objs_train = self.objs[:n_train]

        self.Ds_validation = self.Ds[n_train:n_train+n_validation]
        self.objs_validation = self.objs[n_train:n_train+n_validation]

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
        

    def train(self):
        model = self.model

        print(self.Ds_train.shape, self.objs_train.shape, self.Ds_validation.shape, self.objs_validation.shape)
        
        for epoch in np.arange(n_epochs):
            if self.nn_mode == MODE_1:
                output_data_train = np.zeros((self.objs_train.shape[0], nx, nx))
                output_data_validation = np.zeros((self.objs_validation.shape[0], nx, nx))
                #output_data_train = np.zeros((self.objs_train.shape[0], 1))
                #output_data_validation = np.zeros((self.objs_validation.shape[0], 1))
            elif self.nn_mode == MODE_2:
                output_data_train = self.Ds_train
                output_data_validation = self.Ds_validation
            else:
                assert(False)

                #output_data_train = np.concatenate((output_data_train, self.Ds_train), axis=3)
                #output_data_validation = np.concatenate((output_data_validation, self.Ds_validation), axis=3)
            history = model.fit(self.Ds_train, output_data_train,
                        epochs=1,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(self.Ds_validation, output_data_validation),
                        #callbacks=[keras.callbacks.TensorBoard(log_dir='model_log')],
                        verbose=1)
            #history = model.fit([self.Ds_train, self.objs_train, np.tile(self.Ds_train, [1, 1, 1, 16])], self.Ds_train,
            #            epochs=1,
            #            batch_size=1,
            #            shuffle=True,
            #            validation_data=([self.Ds_validation, self.objs_validation, np.tile(self.Ds_validation, [1, 1, 1, 32])], self.Ds_validation),
            #            #callbacks=[keras.callbacks.TensorBoard(log_dir='model_log')],
            #            verbose=1)
            
            
            #intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("alphas_layer").output)
            #save_weights(intermediate_layer_model)
            save_weights(model)
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
        n_test = min(num_objs, 5)
        try:
            alphas_layer_model = Model(inputs=model.input, outputs=model.get_layer("alphas_layer").output)
            pred_alphas = alphas_layer_model.predict([self.Ds, np.zeros_like(self.objs)], batch_size=1)
        except ValueError:
            pred_alphas = None
        #pred_alphas = intermediate_layer_model.predict([self.Ds, self.objs, np.tile(self.Ds, [1, 1, 1, 16])], batch_size=1)
        if nn_mode == MODE_2:
            pred_Ds = model.predict([self.Ds, self.objs], batch_size=1)
        else:
            pred_Ds = None
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
            DF = np.zeros((num_frames_input, 2, self.nx-1, self.nx-1), dtype="complex")
            for l in np.arange(num_frames_input):
                D = misc.sample_image(self.Ds[i, :, :, 2*l], .99)
                D_d = misc.sample_image(self.Ds[i, :, :, 2*l+1], .99)
                DF[l, 0] = fft.fft2(D)
                DF[l, 1] = fft.fft2(D_d)
            
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
            
            if pred_alphas is not None:
                obj_reconstr = psf_check.deconvolve(DF, alphas=np.reshape(pred_alphas[i], (num_frames_input, jmax)), gamma=gamma, do_fft = True, fft_shift_before = False, ret_all=False, a_est=None, normalize = False)
                obj_reconstr = fft.ifftshift(obj_reconstr)
                objs_reconstr.append(obj_reconstr)
            #print("pred_alphas", i, pred_alphas[i])

            num_rows = 0
            if pred_alphas is not None:
                num_rows += 1
            if pred_Ds is not None:
                num_rows += 2
            my_test_plot = plot.plot(nrows=num_rows, ncols=3)
            #my_test_plot.colormap(np.reshape(self.objs[i], (self.nx+1, self.nx+1)), [0])
            #my_test_plot.colormap(np.reshape(pred_objs[i], (self.nx+1, self.nx+1)), [1])
            row = 0
            if pred_alphas is not None:
                my_test_plot.colormap(obj, [row, 0], show_colorbar=True, colorbar_prec=2)
                my_test_plot.colormap(obj_reconstr, [row, 1])
                my_test_plot.colormap(misc.sample_image(obj, .99) - obj_reconstr, [row, 2])
                row += 1
            if pred_Ds is not None:
                my_test_plot.colormap(self.Ds[i, :, :, 0], [row, 0])
                my_test_plot.colormap(pred_Ds[i, :, :, 0], [row, 1])
                my_test_plot.colormap(np.abs(self.Ds[i, :, :, 0] - pred_Ds[i, :, :, 0]), [row, 2])
                row += 1
                my_test_plot.colormap(self.Ds[i, :, :, 1], [row, 0])
                my_test_plot.colormap(pred_Ds[i, :, :, 1], [row, 1])
                my_test_plot.colormap(np.abs(self.Ds[i, :, :, 1] - pred_Ds[i, :, :, 1]), [row, 2])

            my_test_plot.save(dir_name + "/train_results" + str(i) + ".png")
            my_test_plot.close()
            
            i += 1

 
        #######################################################################
                    
    
    def test(self):
        
        model = self.model
        
        Ds_, objs, nx_orig = gen_data(num_frames=n_test_frames, images_dir = images_dir_test, num_images=n_test_objects)
        print("test_1")
        num_frames = Ds_.shape[0]
        num_objects = Ds_.shape[1]

        Ds, num_frames = convert_data(Ds_)
        #Ds = np.transpose(np.reshape(Ds_, (num_frames*num_objects, Ds_.shape[2], Ds_.shape[3], Ds_.shape[4])), (0, 2, 3, 1))
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
        #pred_alphas = intermediate_layer_model.predict([Ds, np.zeros_like(objs), np.tile(Ds, [1, 1, 1, 16])], batch_size=1)
        try:
            intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("alphas_layer").output)
            pred_alphas = intermediate_layer_model.predict(Ds, batch_size=batch_size)
        except ValueError:
            pred_alphas = None
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

        if pred_alphas is not None:
            obj_reconstr = psf_check.deconvolve(DFs, alphas=pred_alphas, gamma=gamma, do_fft = True, fft_shift_before = False, ret_all=False, a_est=None, normalize = False)
            obj_reconstr = fft.ifftshift(obj_reconstr)
            #obj_reconstr_mean += obj_reconstr

            #my_test_plot = plot.plot(nrows=1, ncols=2)
            #my_test_plot.colormap(np.reshape(objs[i], (self.nx, self.nx)), [0])
            #my_test_plot.colormap(obj_reconstr, [1])
            #my_test_plot.save("test_results_mode" + str(nn_mode) + "_" + str(i) + ".png")
            #my_test_plot.close()
        
        n_rows = 1
        if pred_alphas is not None:
            n_rows += 1
        my_test_plot = plot.plot(nrows=n_rows, ncols=2)
        row = 0
        obj = np.reshape(objs[i], (self.nx, self.nx))
        if pred_alphas is not None:
            my_test_plot.colormap(obj, [row, 0], show_colorbar=True, colorbar_prec=2)
            my_test_plot.colormap(obj_reconstr, [row, 1])
            row += 1
        my_test_plot.colormap(Ds[i, :, :, 0], [row, 0])
        my_test_plot.colormap(Ds[i, :, :, 1], [row, 1])
        my_test_plot.save(dir_name + "/test_results_mean.png")
        my_test_plot.close()
        

def gen_data(num_frames, images_dir = images_dir_train, num_images = None, shuffle = True):
    image_file = None
    images, _, nx, nx_orig = utils.read_images(images_dir, image_file, is_planet = False, image_size=None, tile=True)
    images = np.asarray(images)
    if shuffle:
        random_indices = random.choice(len(images), size=len(images), replace=False)
        images = images[random_indices]
    print("nx, nx_orig", nx, nx_orig)
    if num_images is not None and len(images) > num_images:
        images = images[:num_images]

    arcsec_per_px, defocus = get_params(nx_orig)
    aperture_func = lambda xs: utils.aperture_circ(xs, coef=15, radius =1.)
    defocus_func = lambda xs: defocus*np.sum(xs*xs, axis=2)

    coords, _, _ = utils.get_coords(nx, arcsec_per_px, diameter, wavelength)

    num_objects = len(images)

    Ds = np.zeros((num_frames, num_objects, 2, nx, nx)) # in real space
    #true_coefs = np.zeros((num_frames, jmax))
    pa = psf.phase_aberration(jmax, start_index=0)
    pa.calc_terms(nx=nx)
    wavefront = kolmogorov.kolmogorov(fried = np.array([fried_param]), num_realizations=num_frames, size=4*nx, sampling=1.)
    DFs = np.zeros((num_frames, num_objects, 2, 2*nx-1, 2*nx-1), dtype='complex')
    zernike_coefs = np.zeros((num_frames, jmax))
    #pa = psf.phase_aberration(np.random.normal(size=jmax))
    for frame_no in np.arange(num_frames):
        #pa_true = psf.phase_aberration(np.minimum(np.maximum(np.random.normal(size=jmax)*10, -25), 25), start_index=0)
        zernike_coefs[frame_no] = np.random.normal(size=jmax)*500
        pa_true = psf.phase_aberration(zernike_coefs[frame_no], start_index=0)
        #ctf_true = psf.coh_trans_func(aperture_func, psf.wavefront(wavefront[0,frame_no,:,:]), defocus_func)
        ctf_true = psf.coh_trans_func(aperture_func, pa_true, defocus_func)
        #print("wavefront", np.max(wavefront[0,frame_no,:,:]), np.min(wavefront[0,frame_no,:,:]))
        #true_coefs[frame_no] = ctf_true.dot(pa)
        
        #true_coefs[frame_no] = pa_true.alphas
        #true_coefs[frame_no] -= np.mean(true_coefs[frame_no])
        #true_coefs[frame_no] /= np.std(true_coefs[frame_no])
        psf_true = psf.psf(ctf_true, nx, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength)
        print(np.max(coords), np.min(coords))
        #zernike_coefs[frame_no] = np.random.normal(size=(jmax))*np.linspace(1, .10, jmax)
        #zernike_coefs[frame_no][25] = -1.
        #zernike_coefs[frame_no] = ctf_true.dot(pa)
        print(zernike_coefs[frame_no])
        #######################################################################
        # Plot the wavefront
        pa_check = psf.phase_aberration(zernike_coefs[frame_no], start_index=0)
        pa_check.calc_terms(nx=nx)
        my_test_plot = plot.plot(nrows=1, ncols=3)
        my_test_plot.colormap(wavefront[0,frame_no,:,:], [0], show_colorbar=True, colorbar_prec=2)
        my_test_plot.colormap(pa_check(), [1])
        my_test_plot.colormap(np.abs(wavefront[0,frame_no,:,:] - pa_check()), [2])
        my_test_plot.save(dir_name + "/pa" + str(frame_no) + ".png")
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
        for obj_no in np.arange(num_objects):
            image = images[obj_no]
            # Omit for now (just technical issues)
            image = misc.sample_image(psf.critical_sampling(misc.sample_image(image, .99), arcsec_per_px, diameter, wavelength), 1.01010101)
            image -= np.mean(image)
            image /= np.std(image)
            images[obj_no] = image
            
            #my_test_plot = plot.plot()
            #my_test_plot.colormap(image)
            #my_test_plot.save("critical_sampling" + str(frame_no) + " " + str(obj_no) + ".png")
            #my_test_plot.close()
            
            
            #image1 = misc.sample_image(image, .99)
            #image1 -= np.mean(image)
            #image1 /= np.std(image)
            
            image1 = utils.upsample(image)

            D_D_d = psf_true.convolve(image1)

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

            DFs[frame_no, obj_no, 0] = fft.fft2(D)
            DFs[frame_no, obj_no, 1] = fft.fft2(D_d)

            D = misc.sample_image(D, 0.5)
            D_d = misc.sample_image(D_d, 0.5)

            Ds[frame_no, obj_no, 0] = D#misc.sample_image(D, 1.01010101)
            Ds[frame_no, obj_no, 1] = D_d#misc.sample_image(D_d, 1.01010101)

        print("Finished aberrating with wavefront", frame_no)

    for obj_no in np.arange(min(5, num_objects)):
        my_test_plot = plot.plot(nrows=1, ncols=4)
        my_test_plot.colormap(images[obj_no], [0], show_colorbar=True, colorbar_prec=2)
        my_test_plot.colormap(Ds[0, obj_no, 0], [1])
        my_test_plot.colormap(Ds[0, obj_no, 1], [2])
        ###############################################################
    
        obj_reconstr = psf_check.deconvolve(DFs[:, obj_no,:, :, :], alphas=zernike_coefs, gamma=gamma, do_fft = True, fft_shift_before = False, ret_all=False, a_est=None, normalize = False)
        obj_reconstr = fft.ifftshift(obj_reconstr)
        my_test_plot.colormap(obj_reconstr, [3])

        my_test_plot.save(dir_name + "/check" + str(obj_no) + ".png")
        my_test_plot.close()


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



Ds, objs = load_data()
if Ds is None:
    Ds, objs, nx_orig = gen_data(num_frames_gen)
    save_data(Ds, objs)


nx = Ds.shape[3]

#Ds_mean = np.mean(Ds, axis=(2,3))
#Ds_std = np.std(Ds, axis=(2,3))
#Ds -= np.tile(np.reshape(Ds_mean, (Ds_mean.shape[0], Ds_mean.shape[0], 1, 1)), (1, 1, nx, nx))
#Ds /= np.tile(np.reshape(Ds_std, (Ds_mean.shape[0], Ds_mean.shape[0], 1, 1)), (1, 1, nx, nx))

my_test_plot = plot.plot()
my_test_plot.colormap(Ds[0, 0, 0])
my_test_plot.save(dir_name + "/D0.png")
my_test_plot.close()

my_test_plot = plot.plot()
my_test_plot.colormap(Ds[0, 0, 1])
my_test_plot.save(dir_name + "/D0_d.png")
my_test_plot.close()


model = nn_model(nx, num_frames, num_objs)

model.set_data(Ds, objs)

if train:
    for rep in np.arange(0, num_reps):
        print("Rep no: " + str(rep))
    
        model.train()

        model.test()
        
        model.set_data(Ds, objs)
            
    
        #if np.mean(model.validation_losses[-10:]) > np.mean(model.validation_losses[-20:-10]):
        #    break
        model.validation_losses = model.validation_losses[-20:]
else:
    model.test()

#logfile.close()