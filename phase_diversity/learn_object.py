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

fried_param = 0.1
noise_std_perc = 0.#.01

n_epochs = 10
num_iters = 10
num_reps = 1000
suffle = True

MODE_1 = 1 # aberrated images --> object
MODE_2 = 2 # aberrated images --> wavefront coefs (+object as second input) --> aberrated images
MODE_3 = 3 # aberrated images --> psf (+object as second input) --> aberrated images
nn_mode = MODE_2

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
    model_file = dir_name + '/learn_object_model.dat'
    if os.path.exists(model_file):
        model = tf.keras.models.load_model(model_file)
        nn_mode = pickle.load(open(dir_name + '/learn_object_params.dat', 'rb'))
        return model, nn_mode
    return None, None

def save_model(model):
    tf.keras.models.save_model(model, dir_name + '/learn_object_model.dat')
    with open(dir_name + '/learn_object_params.dat', 'wb') as f:
        pickle.dump(nn_mode, f, protocol=4)


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
            nn_mode_ = nn_mode
            if nn_mode == MODE_1:
                ###################################################################
                # Representation in a higher dim space
                ###################################################################
                #hidden_layer = keras.layers.convolutional.Convolution2D(32, 8, 8, subsample=(2, 2), activation='relu')(image_input)#(normalized)
                hidden_layer = keras.layers.Conv2D(16, (64, 64), activation='relu', padding='same')(image_input)#(normalized)
                hidden_layer = keras.layers.MaxPooling2D(pool_size=(4,4))(hidden_layer)
                #hidden_layer = keras.layers.convolutional.Conv2D(16, (nx//4, nx//4), padding='same', activation='linear')(hidden_layer)#(normalized)
                #hidden_layer = keras.layers.Lambda(lambda x:K.mean(x, axis=0))(hidden_layer)
                #hidden_layer = keras.layers.UpSampling2D((2, 2))(hidden_layer)
                hidden_layer = keras.layers.Flatten()(hidden_layer)
                #output = keras.layers.core.Flatten()(hidden_layer)
                output = keras.layers.Dense(nx*nx, activation='linear')(hidden_layer)
                #output = keras.layers.Dense(nx*nx, activation='linear')(hidden_layer)
                #output = keras.layers.Reshape((nx, nx))(hidden_layer)
                #output = keras.layers.convolutional.Conv2D(64, (8, 8), activation='relu')(image_input)#(normalized)
                #output = keras.layers.add(hidden_layer)(image_input)#(normalized)
    
                model = keras.models.Model(inputs=image_input, outputs=output)
            elif nn_mode == MODE_2:
                self.create_psf()
                object_input = keras.layers.Input((nx*nx), name='object_input') # Channels first
                #object_input  = keras.layers.Reshape((nx*nx))(object_input)
                ###################################################################
                # Autoencoder
                ###################################################################
                hidden_layer0 = keras.layers.Conv2D(64, (64, 64), activation='relu', padding='same')(image_input)#(normalized)
                #hidden_layer0 = keras.layers.Conv2D(32, (64, 64), activation='relu', padding='same')(hidden_layer0)#(normalized)
                #hidden_layer0 = keras.layers.BatchNormalization()(hidden_layer0)
                #hidden_layer0 = keras.layers.add([hidden_layer0, tf.keras.backend.tile(image_input, [1, 1, 1, 16])])
                hidden_layer0 = keras.layers.concatenate([hidden_layer0, image_input], name='h0')
                hidden_layer1 = keras.layers.MaxPooling2D()(hidden_layer0)
                hidden_layer2 = keras.layers.Conv2D(64, (32, 32), activation='relu', padding='same')(hidden_layer1)#(normalized)
                #hidden_layer2 = keras.layers.Conv2D(32, (32, 32), activation='relu', padding='same')(hidden_layer2)#(normalized)
                #hidden_layer2 = keras.layers.BatchNormalization()(hidden_layer2)
                #hidden_layer2 = keras.layers.add([hidden_layer2, hidden_layer1])
                hidden_layer2 = keras.layers.concatenate([hidden_layer2, hidden_layer1], name='h2')
                
                hidden_layer3 = keras.layers.MaxPooling2D()(hidden_layer2)
                hidden_layer4 = keras.layers.Conv2D(32, (16, 16), activation='relu', padding='same')(hidden_layer3)#(normalized)
                #hidden_layer4 = keras.layers.Conv2D(32, (16, 16), activation='relu', padding='same')(hidden_layer4)#(normalized)
                #hidden_layer4 = keras.layers.BatchNormalization()(hidden_layer4)
                #hidden_layer4 = keras.layers.add([hidden_layer4, hidden_layer3])
                hidden_layer4 = keras.layers.concatenate([hidden_layer4, hidden_layer3], name='h4')
                
                hidden_layer5 = keras.layers.MaxPooling2D()(hidden_layer4)
                hidden_layer6 = keras.layers.Conv2D(32, (8, 8), activation='relu', padding='same')(hidden_layer5)#(normalized)
                #hidden_layer6 = keras.layers.Conv2D(32, (8, 8), activation='relu', padding='same')(hidden_layer6)#(normalized)
                #hidden_layer6 = keras.layers.BatchNormalization()(hidden_layer6)
                #hidden_layer6 = keras.layers.add([hidden_layer6, hidden_layer5])
                hidden_layer6 = keras.layers.concatenate([hidden_layer6, hidden_layer5], name='h6')
                hidden_layer7 = keras.layers.MaxPooling2D()(hidden_layer6)

                #hidden_layer = keras.layers.Conv2D(64, (7, 7), activation='relu', padding='same')(image_input)#(normalized)
                #hidden_layer = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(image_input)#(normalized)
                #hidden_layer = keras.layers.MaxPooling2D()(hidden_layer)
                #hidden_layer = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(image_input)#(normalized)
                #hidden_layer = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(image_input)#(normalized)
                #hidden_layer = keras.layers.MaxPooling2D()(hidden_layer)
                #hidden_layer = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(image_input)#(normalized)
                #hidden_layer = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(image_input)#(normalized)
                #hidden_layer = keras.layers.MaxPooling2D()(hidden_layer)
                #hidden_layer = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(image_input)#(normalized)
                #hidden_layer = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(image_input)#(normalized)
                #hidden_layer = keras.layers.MaxPooling2D()(hidden_layer)
                
                hidden_layer = keras.layers.Flatten()(hidden_layer7)
                hidden_layer = keras.layers.Dense(1000, activation='relu')(hidden_layer)
                hidden_layer = keras.layers.Dense(jmax, activation='linear', name='alphas_layer')(hidden_layer)
                hidden_layer = keras.layers.concatenate([hidden_layer, object_input])
                output = keras.layers.Lambda(self.aberrate)(hidden_layer)
               
                model = keras.models.Model(inputs=[image_input, object_input], outputs=output)

            elif nn_mode == MODE_3:
                self.create_psf()
                object_input = keras.layers.Input((nx*nx), name='object_input') # Channels first
                #object_input  = keras.layers.Reshape((nx*nx))(object_input)
                ###################################################################
                # Autoencoder
                ###################################################################
                hidden_layer = keras.layers.Conv2D(4, (64, 64), activation='relu', padding='same')(image_input)#(normalized)
                hidden_layer = keras.layers.MaxPooling2D()(hidden_layer)
                hidden_layer = keras.layers.Conv2D(8, (32, 32), activation='relu', padding='same')(hidden_layer)#(normalized)
                hidden_layer = keras.layers.MaxPooling2D()(hidden_layer)
                hidden_layer = keras.layers.Conv2D(16, (16, 16), activation='relu', padding='same')(hidden_layer)#(normalized)
                hidden_layer = keras.layers.MaxPooling2D()(hidden_layer)
                hidden_layer = keras.layers.Conv2D(32, (8, 8), activation='relu', padding='same')(hidden_layer)#(normalized)
                hidden_layer = keras.layers.MaxPooling2D()(hidden_layer)
                hidden_layer = keras.layers.Conv2D(64, (4, 4), activation='relu', padding='same')(hidden_layer)#(normalized)
                hidden_layer = keras.layers.MaxPooling2D()(hidden_layer)
                hidden_layer = keras.layers.UpSampling2D((2, 2))(hidden_layer)
                hidden_layer = keras.layers.Conv2D(32, (4, 4), activation='relu', padding='same')(hidden_layer)#(normalized)
                hidden_layer = keras.layers.UpSampling2D((2, 2))(hidden_layer)
                hidden_layer = keras.layers.Conv2D(16, (8, 8), activation='relu', padding='same')(hidden_layer)#(normalized)
                hidden_layer = keras.layers.UpSampling2D((2, 2))(hidden_layer)
                hidden_layer = keras.layers.Conv2D(8, (16, 16), activation='relu', padding='same')(hidden_layer)#(normalized)
                hidden_layer = keras.layers.UpSampling2D((2, 2))(hidden_layer)
                hidden_layer = keras.layers.Conv2D(4, (32, 32), activation='relu', padding='same')(hidden_layer)#(normalized)
                hidden_layer = keras.layers.UpSampling2D((2, 2))(hidden_layer)
                hidden_layer = keras.layers.Conv2D(2, (64, 64), activation='relu', padding='same', name='psf_layer')(image_input)#(normalized)
                hidden_layer = keras.layers.Flatten()(hidden_layer)
                hidden_layer = keras.layers.concatenate([hidden_layer, object_input])
                output = keras.layers.Lambda(self.aberrate_psf)(hidden_layer)
               
                model = keras.models.Model(inputs=[image_input, object_input], outputs=output)
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
            if nn_mode_ == MODE_2:
                self.create_psf()
                print("Mode 2")
                object_input = model.input[1]
                hidden_layer = keras.layers.concatenate([model.output, object_input])
                output = keras.layers.Lambda(self.aberrate)(hidden_layer)
                full_model = Model(inputs=model.input, outputs=output)
                model = full_model
            if nn_mode_ == MODE_3:
                print("Mode 3")
                object_input = model.input[1]
                hidden_layer = keras.layers.concatenate([model.output, object_input])
                output = keras.layers.Lambda(self.aberrate_psf)(hidden_layer)
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
        i1 = random.randint(0, Ds.shape[0] + 1 - self.num_frames)
        i2 = random.randint(0, Ds.shape[1] + 1 - self.num_objs)
        Ds = Ds[i1:i1+self.num_frames, i2:i2+self.num_objs]
        self.objs = objs[i2:i2+self.num_objs]
        num_objects = Ds.shape[1]
        self.Ds = np.transpose(np.reshape(Ds, (self.num_frames*num_objects, Ds.shape[2], Ds.shape[3], Ds.shape[4])), (0, 2, 3, 1))
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
        

    def train(self):
        model = self.model

        print(self.Ds_train.shape, self.objs_train.shape, self.Ds_validation.shape, self.objs_validation.shape)
        
        for epoch in np.arange(n_epochs):
            if self.nn_mode == MODE_1:
                history = model.fit(self.Ds_train, self.objs_train,
                            epochs=1,
                            batch_size=1,
                            shuffle=True,
                            validation_data=(self.Ds_validation, self.objs_validation),
                            #callbacks=[keras.callbacks.TensorBoard(log_dir='model_log')],
                            verbose=1)
                save_model(model)
            elif self.nn_mode == MODE_2:
                history = model.fit([self.Ds_train, self.objs_train], self.Ds_train,
                            epochs=1,
                            batch_size=1,
                            shuffle=True,
                            validation_data=([self.Ds_validation, self.objs_validation], self.Ds_validation),
                            #callbacks=[keras.callbacks.TensorBoard(log_dir='model_log')],
                            verbose=1)
                intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("alphas_layer").output)
                save_model(intermediate_layer_model)
            elif self.nn_mode == MODE_3:
                history = model.fit([self.Ds_train, self.objs_train], self.Ds_train,
                            epochs=1,
                            batch_size=1,
                            shuffle=True,
                            validation_data=([self.Ds_validation, self.objs_validation], self.Ds_validation),
                            #callbacks=[keras.callbacks.TensorBoard(log_dir='model_log')],
                            verbose=1)
                intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("psf_layer").output)
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
        n_test = min(num_objs, 5)
        if self.nn_mode == MODE_1:
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
                    my_test_plot.save(dir_name + "/train_results" + str(i) + ".png")
                    my_test_plot.close()
        elif self.nn_mode == MODE_2:
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
                obj_reconstr = fft.ifftshift(obj_reconstr)
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

        elif self.nn_mode == MODE_3:
            intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("psf_layer").output)
            pred_psf = intermediate_layer_model.predict([self.Ds, self.objs], batch_size=1)
            pred_Ds = model.predict([self.Ds, self.objs], batch_size=1)
            for i in np.arange(self.Ds.shape[0]):
                if i < n_test:
                    D = self.Ds[i, :, :, 0]
                    D_d = self.Ds[i, :, :, 1]
                    DF = fft.fft2(D)
                    DF_d = fft.fft2(D_d)
                    
                    psfF = fft.fft2(pred_psf)
                    
                    obj_reconstr_1 = fft.ifft2(DF/psfF[0])
                    obj_reconstr_2 = fft.ifft2(DF_d/psfF[1])

                    my_test_plot = plot.plot(nrows=4, ncols=2)
                    my_test_plot.colormap(obj, [0, 0])
                    my_test_plot.colormap(obj_reconstr_1, [0, 1])
                    my_test_plot.colormap(obj, [1, 0])
                    my_test_plot.colormap(obj_reconstr_2, [1, 1])
                    my_test_plot.colormap(self.Ds[i, :, :, 0], [2, 0])
                    my_test_plot.colormap(pred_Ds[i, :, :, 0], [2, 1])
                    my_test_plot.colormap(self.Ds[i, :, :, 1], [3, 0])
                    my_test_plot.colormap(pred_Ds[i, :, :, 1], [3, 1])
                    my_test_plot.save(dir_name + "/train_results" + str(i) + ".png")
                    my_test_plot.close()
    
        #######################################################################
                    
    
    def test(self):
        
        model = self.model
        
        Ds_, objs, nx_orig = gen_data(num_frames=n_test_frames, images_dir = images_dir_test, num_images=n_test_objects)
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

        if self.nn_mode == MODE_1:
            start = time.time()    
            pred_objs = model.predict(Ds)
            end = time.time()
            print("Prediction time" + str(end - start))
    
            for i in np.arange(num_frames):
                #D1 = psf_check.convolve(image, alphas=true_coefs[frame_no])
                my_test_plot = plot.plot(nrows=1, ncols=2)
                my_test_plot.colormap(np.reshape(objs[i], (self.nx, self.nx)), [0])
                my_test_plot.colormap(np.reshape(pred_objs[i], (self.nx, self.nx)), [1])
                #my_test_plot.colormap(np.reshape(pred_objs[i], (self.nx+1, self.nx+1)), [1])
                #my_test_plot.colormap(D, [1, 0])
                #my_test_plot.colormap(D_d, [1, 1])
                #my_test_plot.colormap(D1[0, 0], [2, 0])
                #my_test_plot.colormap(D1[0, 1], [2, 1])
                my_test_plot.save(dir_name + "/test_results" + str(i) + ".png")
                my_test_plot.close()
        elif self.nn_mode == MODE_2:
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
            obj_reconstr = fft.ifftshift(obj_reconstr)
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
            
        elif self.nn_mode == MODE_3:
            start = time.time()    
            intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("psf_layer").output)
            pred_psf = intermediate_layer_model.predict([Ds, np.zeros_like(objs)], batch_size=1)
            end = time.time()
            print("Prediction time" + str(end - start))
            for i in np.arange(len(objs)):
                D = Ds[i, :, :, 0]
                D_d = Ds[i, :, :, 1]
                DF = fft.fft2(D)
                DF_d = fft.fft2(D_d)
                
                psfF = fft.fft2(pred_psf)
                
                obj_reconstr_1 = fft.ifft2(DF/psfF[0])
                obj_reconstr_2 = fft.ifft2(DF_d/psfF[1])

                my_test_plot = plot.plot(nrows=1, ncols=3)
                my_test_plot.colormap(np.reshape(objs[i], (self.nx, self.nx)), [0])
                my_test_plot.colormap(obj_reconstr_1, [1])
                my_test_plot.colormap(obj_reconstr_2, [2])
                my_test_plot.save(dir_name + "/test_results" + str(i) + ".png")
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
            fimage = fft.fft2(misc.sample_image(image, .99))
            fimage = fft.fftshift(fimage)
    
        
            DFs1 = psf_true.multiply(fimage)
            DF = DFs1[0, 0]
            DF_d = DFs1[0, 1]
            
            DF = fft.ifftshift(DF)
            DF_d = fft.ifftshift(DF_d)
        
            D = fft.ifft2(DF).real
            D_d = fft.ifft2(DF_d).real

            if noise_std_perc > 0.:
                noise = np.random.poisson(lam=noise_std_perc*np.std(D), size=(nx-1, nx-1))
                noise_d = np.random.poisson(lam=noise_std_perc*np.std(D_d), size=(nx-1, nx-1))

                D += noise
                D_d += noise_d

            ###################################################################
            # Just checking if true_coefs are calculated correctly
            if frame_no < 1 and obj_no < 5:
                my_test_plot = plot.plot(nrows=1, ncols=3)
                my_test_plot.colormap(image, [0])
                my_test_plot.colormap(D, [1])
                my_test_plot.colormap(D_d, [2])
                my_test_plot.save(dir_name + "/check" + str(frame_no) + "_" + str(obj_no) + ".png")
                my_test_plot.close()
            ###################################################################

            Ds[frame_no, obj_no, 0] = misc.sample_image(D, 1.01010101)
            Ds[frame_no, obj_no, 1] = misc.sample_image(D_d, 1.01010101)
        print("Finished aberrating with wavefront", frame_no)

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
    
        model.train()

        model.test()
        
        model.set_data(Ds, objs)
            
    
        #if np.mean(model.validation_losses[-10:]) > np.mean(model.validation_losses[-20:-10]):
        #    break
        model.validation_losses = model.validation_losses[-20:]
else:
    model.test()

#logfile.close()