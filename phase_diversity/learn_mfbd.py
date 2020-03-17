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
import matplotlib.pyplot as plt

import time
#import scipy.signal as signal

gamma = 1.0

# How many frames to use in training
num_frames = 20#100
# How many objects to use in training
num_objs = 10#75#None

# How many frames of the same object are sent to NN input
# Must be power of 2
num_frames_input = 8

n_epochs_2 = 1
n_epochs_1 = 1
num_reps = 1000
shuffle = True

MODE_1 = 1 # aberrated images --> wavefront coefs --> MFBD loss
MODE_2 = 2 # aberrated images --> wavefront coefs --> object (using MFBD formula) --> aberrated images
nn_mode = MODE_1

batch_size = 8
n_channels = 256

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

n_test_frames = None
if len(sys.argv) > 3:
    n_test_frames = int(sys.argv[3])

n_test_objects = None
if len(sys.argv) > 4:
    n_test_objects = int(sys.argv[4])

test_data_file = "Ds_test.npz"
if len(sys.argv) > 5:
    test_data_file = sys.argv[5]

if dir_name is None:
    dir_name = "results" + time.strftime("%Y%m%d-%H%M%S")
    os.mkdir(dir_name)
    
    f = open(dir_name + '/params.txt', 'w')
    f.write('fried jmax num_frames_gen num_frames num_objs nn_mode\n')
    f.write('%d %d %d' % (num_frames, num_objs, nn_mode) + "\n")
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
#import gen_images
#import gen_data

#tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

n_gpus = len(gpus)

def load_data(data_file="Ds.npz"):
    data_file = dir_name + '/' + data_file
    if os.path.exists(data_file):
        loaded = np.load(data_file)
        Ds = loaded['Ds']
        try:
            objs = loaded['objs']
        except:
            objs = None
        pupil = loaded['pupil']
        modes = loaded['modes']
        diversity = loaded['diversity']
        try:
            coefs = loaded['coefs']
        except:
            coefs = None
        try:
            positions = loaded['positions']
        except:
            positions = None
        try:
            coords = loaded['coords']
        except:
            coords = None
        return Ds, objs, pupil, modes, diversity, coefs, positions, coords
    raise "No data found"


def load_model():
    model_file = dir_name + '/model.h5'
    if not os.path.exists(model_file):
        model_file = dir_name + '/model.dat' # Just an old file suffix
    if os.path.exists(model_file):
        model = tf.keras.models.load_model(model_file)
        nn_mode = pickle.load(open(dir_name + '/params.dat', 'rb'))
        return model, nn_mode
    return None, None

def save_model(model):
    tf.keras.models.save_model(model, dir_name + '/model.h5')
    with open(dir_name + '/params.dat', 'wb') as f:
        pickle.dump(nn_mode, f, protocol=4)


def load_weights(model):
    model_file = dir_name + '/weights.tf'
    if not os.path.exists(model_file):
        model_file = dir_name + '/weights.h5'
    if os.path.exists(model_file):
        model.load_weights(model_file)
        nn_mode = pickle.load(open(dir_name + '/params.dat', 'rb'))
        return nn_mode
    return None

def save_weights(model):
    model.save_weights(dir_name + '/weights.tf')
    with open(dir_name + '/params.dat', 'wb') as f:
        pickle.dump(nn_mode, f, protocol=4)

'''
def get_params(nx):

    #arcsec_per_px = .03*(wavelength*1e-10)/(diameter*1e-2)*180/np.pi*3600
    arcsec_per_px = .25*(wavelength*1e-10)/(diameter*1e-2)*180/np.pi*3600
    print("arcsec_per_px=", arcsec_per_px)
    defocus = 2.*np.pi*1000
    #defocus = (0., 0.)
    return (arcsec_per_px, defocus)
'''

def convert_data(Ds_in, objs_in, diversity_in=None, positions=None, coords=None):
    num_objects = Ds_in.shape[0]
    num_frames = Ds_in.shape[1]
    Ds_out = np.zeros(((num_frames-num_frames_input+1)*num_objects, Ds.shape[3], Ds.shape[4], Ds.shape[2]*num_frames_input))
    if objs_in is not None:
        objs_out = np.zeros(((num_frames-num_frames_input+1)*num_objects, objs_in.shape[1], objs_in.shape[2]))
    else:
        objs_out  = None
    if diversity_in is not None:
        diversity_out = np.zeros(((num_frames-num_frames_input+1)*num_objects, Ds.shape[3], Ds.shape[4], Ds.shape[2]*num_frames_input))
    else:
        diversity_out = None
    ids = np.zeros((num_frames-num_frames_input+1)*num_objects, dtype='int')
    positions_out= np.zeros(((num_frames-num_frames_input+1)*num_objects, 2), dtype='int')
    coords_out= np.zeros(((num_frames-num_frames_input+1)*num_objects, 2), dtype='int')
        
    k = 0
    l = 0
    for i in np.arange(num_objects):
        for j in np.arange(num_frames):
            Ds_out[k, :, :, 2*l] = Ds_in[i, j, 0, :, :]
            Ds_out[k, :, :, 2*l+1] = Ds_in[i, j, 1, :, :]
            if objs_out is not None:
                objs_out[k] = objs_in[i]
            if diversity_out is not None:
                if positions is None:
                    if len(diversity_in.shape) == 3:
                        # Diversities both for focus and defocus image
                        #diversity_out[k, :, :, 2*l] = diversity_in[0]
                        for div_i in np.arange(diversity_in.shape[0]):
                            diversity_out[k, :, :, 2*l+1] += diversity_in[div_i]
                    else:
                        assert(len(diversity_in.shape) == 2)
                        # Just a defocus
                        diversity_out[k, :, :, 2*l+1] = diversity_in
                else:
                    assert(len(diversity_in.shape) == 5)
                    #for div_i in np.arange(diversity_in.shape[2]):
                    #    #diversity_out[k, :, :, 2*l] = diversity_in[positions[i, 0], positions[i, 1], 0]
                    #    diversity_out[k, :, :, 2*l+1] += diversity_in[positions[i, 0], positions[i, 1], div_i]
                    #    #diversity_out[k, :, :, 2*l+1] = diversity_in[positions[i, 0], positions[i, 1], 1]
                    diversity_out[k, :, :, 2*l+1] += diversity_in[positions[i, 0], positions[i, 1], 1]
            ids[k] = i
            if positions is not None:
                positions_out[k] = positions[i]
            if coords is not None:
                coords_out[k] = coords[i]
            l += 1
            if l >= num_frames_input:
                l = 0
                k += 1
    Ds_out = Ds_out[:k]
    if objs_out is not None:
        objs_out = objs_out[:k]
    if diversity_out is not None:
        diversity_out = diversity_out[:k]
    ids = ids[:k]
    positions_out = positions_out[:k]
    coords_out = coords_out[:k]
    #assert(k == (num_frames-num_frames_input+1)*num_objects)
    return Ds_out, objs_out, diversity_out, num_frames - num_frames_input + 1, ids, positions_out, coords_out


class nn_model:       
    
    
    def __init__(self, jmax, nx, num_frames, num_objs, pupil, modes):
        
        self.jmax = jmax
        self.num_frames = num_frames
        assert(num_frames_input <= self.num_frames)
        self.num_objs = num_objs
        self.nx = nx
        self.validation_losses = []
        self.hanning = utils.hanning(nx, 10)
        self.filter = utils.create_filter(nx)
        #self.pupil = pupil[nx//4:nx*3//4,nx//4:nx*3//4]
        #self.modes = modes[:, nx//4:nx*3//4,nx//4:nx*3//4]

        self.pupil = pupil
        self.modes = modes

        self.modes_orig = modes
        self.pupil_orig = pupil

         
        pa_check = psf.phase_aberration(len(modes), start_index=1)
        pa_check.set_terms(modes)
        ctf_check = psf.coh_trans_func()
        ctf_check.set_phase_aberr(pa_check)
        ctf_check.set_pupil(pupil)
        #ctf_check.set_diversity(diversity[i, j])
        self.psf_check = psf.psf(ctf_check, corr_or_fft=False)
        
        pa = psf_tf.phase_aberration_tf(len(modes), start_index=1)
        pa.set_terms(modes)
        ctf = psf_tf.coh_trans_func_tf()
        ctf.set_phase_aberr(pa)
        ctf.set_pupil(pupil)
        #ctf.set_diversity(diversity[i, j])
        batch_size_per_gpu = batch_size//max(1, n_gpus)
        self.psf = psf_tf.psf_tf(ctf, num_frames=num_frames_input, batch_size=batch_size_per_gpu, set_diversity=True)
        
        
        num_defocus_channels = 2#self.num_frames*2

        self.strategy = tf.distribute.MirroredStrategy()
        with self.strategy.scope():

            image_input = keras.layers.Input((nx, nx, num_defocus_channels*num_frames_input), name='image_input')
            diversity_input = keras.layers.Input((nx, nx, num_defocus_channels*num_frames_input), name='diversity_input')
    
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
    
    
                def conv_layer(x, n_channels, kernel=(3, 3), max_pooling=True, batch_normalization=True, num_convs=2):
                    for i in np.arange(num_convs):
                        x1 = keras.layers.Conv2D(n_channels, (1, 1), activation='linear', padding='same')(x)#(normalized)
                        x2 = keras.layers.Conv2D(n_channels, kernel, activation='relu', padding='same')(x)#(normalized)
                        x = keras.layers.add([x2, x1])#tf.keras.backend.tile(image_input, [1, 1, 1, 16])])
                        if batch_normalization:
                            x = keras.layers.BatchNormalization()(x)
                    if max_pooling:
                        x = keras.layers.MaxPooling2D()(x)
                    return x
                
                
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
                alphas_layer = keras.layers.Dense(jmax*num_frames_input, activation='linear')(alphas_layer)
                alphas_layer = keras.layers.Lambda(lambda x : multiply(x, 1.), name='alphas_layer')(alphas_layer)
                
                #obj_layer = keras.layers.Dense(256)(obj_layer)
                #obj_layer = keras.layers.Dense(128)(obj_layer)
                #obj_layer = keras.layers.Dense(64)(obj_layer)
                #obj_layer = keras.layers.Dense(128)(obj_layer)
                #obj_layer = keras.layers.Dense(256)(obj_layer)
                #obj_layer = keras.layers.Dense(512)(obj_layer)
                #obj_layer = keras.layers.Dense(1152)(obj_layer)
               
                
                if nn_mode == MODE_1:
                    
                    a1 = tf.reshape(alphas_layer, [batch_size_per_gpu, num_frames_input, jmax])                    
                    a2 = tf.reshape(tf.transpose(diversity_input, [0, 3, 1, 2]), [batch_size_per_gpu, num_frames_input, 2*nx*nx])
                    a3 = tf.concat([a1, a2], axis=2)
                    
                    hidden_layer = keras.layers.concatenate([tf.reshape(a3, [batch_size_per_gpu*num_frames_input*(jmax+2*nx*nx)]), tf.reshape(image_input, [batch_size_per_gpu*num_frames_input*2*nx*nx])])
                    #hidden_layer = keras.layers.concatenate([tf.reshape(alphas_layer, [batch_size*jmax*num_frames_input]), tf.reshape(image_input, [batch_size*num_frames_input*2*nx*nx]), tf.reshape(diversity_input, [batch_size*num_frames_input*2*nx*nx])])
                    output = keras.layers.Lambda(self.psf.mfbd_loss)(hidden_layer)
                    #output = keras.layers.Lambda(lambda x: tf.reshape(tf.math.reduce_sum(x), [1]))(output)
                    #output = keras.layers.Flatten()(output)
                    #output = keras.layers.Lambda(lambda x: tf.math.reduce_sum(x))(output)
                elif nn_mode == MODE_2:
                    hidden_layer = keras.layers.concatenate([tf.reshape(alphas_layer, [batch_size_per_gpu*jmax*num_frames_input]), tf.reshape(image_input, [batch_size*num_frames_input*2*nx*nx])])
                    output = keras.layers.Lambda(self.psf.deconvolve_aberrate)(hidden_layer)
    
                else:
                    assert(False)
               
                model = keras.models.Model(inputs=[image_input, diversity_input], outputs=output)
    
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
                return tf.reduce_sum(tf.subtract(y_pred, y_true))/(nx*nx)
                
            #self.model.compile(optimizer='adadelta', loss=mfbd_loss)#'mse')
            self.model.compile(optimizer='adadelta', loss=mfbd_loss)#'mse')

        nn_mode_ = load_weights(model)
        if nn_mode_ is not None:
            assert(nn_mode_ == nn_mode) # Model was saved with in mode
        else:
            nn_mode_ = nn_mode

        self.nn_mode = nn_mode_

    def set_data(self, Ds, objs, diversity, positions, train_perc=.75):
        if self.num_frames is None or self.num_frames <= 0:
            self.num_frames = Ds.shape[1]
        self.num_frames = min(self.num_frames, Ds.shape[1])
        #assert(self.num_frames <= Ds.shape[1])
        if self.num_objs is None or self.num_objs <= 0:
            self.num_objs = Ds.shape[0]
        self.num_objs = min(self.num_objs, Ds.shape[0])
        #assert(self.num_objs <= Ds.shape[0])
        assert(Ds.shape[2] == 2)
        if objs is None:
            # Just generate dummy array in case we don't have true object data
            objs = np.zeros((Ds.shape[0], Ds.shape[3], Ds.shape[4]))
        if shuffle:
            i1 = random.randint(0, Ds.shape[0] + 1 - self.num_objs)
            i2 = random.randint(0, Ds.shape[1] + 1 - self.num_frames)
        else:
            i1 = 0
            i2 = 0
        Ds = Ds[i1:i1+self.num_objs, i2:i2+self.num_frames]
        objs = objs[i1:i1+self.num_objs]
        if positions is not None:
            positions = positions[i1:i1+self.num_objs]
        num_objects = Ds.shape[1]
        
        self.Ds, self.objs, self.diversities, num_frames, self.obj_ids, self.positions, _s = convert_data(Ds, objs, diversity, positions)
        med = np.median(self.Ds, axis=(1, 2), keepdims=True)
        std = np.std(self.Ds, axis=(1, 2), keepdims=True)
        self.Ds -= med
        self.Ds = self.hanning.multiply(self.Ds, axis=1)
        #self.Ds += med
        self.Ds /= std
        
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
        #print("objs", self.objs.shape, self.num_objs, num_objects)
        #self.objs = np.reshape(np.repeat(self.objs, num_frames, axis=0), (num_frames*self.objs.shape[0], self.objs.shape[1], self.objs.shape[2]))

        #self.objs = np.reshape(self.objs, (len(self.objs), -1))
        #self.objs = np.reshape(np.tile(objs, (1, num_frames)), (num_objects*num_frames, objs.shape[1]))
                    
        # Shuffle the data
        random_indices = random.choice(len(self.Ds), size=len(self.Ds), replace=False)
        self.Ds = self.Ds[random_indices]
        if self.objs is not None:
            self.objs = self.objs[random_indices]
        if self.diversities is not None:
            self.diversities = self.diversities[random_indices]
        self.obj_ids = self.obj_ids[random_indices]
        self.positions = self.positions[random_indices]
        
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
        self.Ds_validation = self.Ds[n_train:n_train+n_validation]

        if self.objs is not None:
            self.objs_train = self.objs[:n_train]
            self.objs_validation = self.objs[n_train:n_train+n_validation]

        if self.diversities is not None:
            self.diversities_train = self.diversities[:n_train]
            self.diversities_validation = self.diversities[n_train:n_train+n_validation]

        self.obj_ids_train = self.obj_ids[:n_train]
        self.obj_ids_validation = self.obj_ids[n_train:n_train+n_validation]
        #for i in np.arange(len(self.objs)):
        #    my_test_plot = plot.plot(nrows=3, ncols=1)
        #    my_test_plot.colormap(self.objs[i], [0, 0], show_colorbar=True, colorbar_prec=2)
        #    my_test_plot.colormap(self.Ds[i, :, :, 0], [1, 0])
        #    my_test_plot.colormap(self.Ds[i, :, :, 1], [2, 0])
    
        #    my_test_plot.save(dir_name + "/data_check" + str(i) + ".png")
        #    my_test_plot.close()
        

    '''        
    def deconvolve(self, num_frames, alphas, diversity, Ds):
    
        pa = psf_tf.phase_aberration_tf(len(self.modes_orig), start_index=1)
        pa.set_terms(self.modes_orig)
        ctf = psf_tf.coh_trans_func_tf()
        ctf.set_phase_aberr(pa)
        ctf.set_pupil(self.pupil_orig)
        #ctf.set_diversity(diversity[i, j])
        psf_ = psf_tf.psf_tf(ctf, num_frames=num_frames, batch_size=1, set_diversity=True)
        #alphas = tf.constant(alphas, dtype="float32")
        #diversity = tf.constant(diversity, dtype="float32")
        #Ds = tf.constant(Ds, dtype="float32")
        
        alphas = tf.cast(alphas, tf.float32)
        diversity = tf.cast(diversity, tf.float32)
        Ds = tf.cast(Ds, tf.float32)

        print("num_frames", num_frames)
        print("alphas", alphas)
        print("diversity", diversity)
        print("Ds", Ds)
    
        a1 = tf.reshape(alphas, [1, num_frames, self.jmax])                    
        a2 = tf.reshape(tf.transpose(diversity, [2, 0, 1]), [1, num_frames_input, 2*self.nx*self.nx])
        a3 = tf.concat([a1, a2], axis=2)
        
        x = tf.concat([tf.reshape(a3, [num_frames*(self.jmax+2*self.nx*self.nx)]), tf.reshape(Ds, [num_frames*2*self.nx*self.nx])], axis=0)
        #hidden_layer = keras.layers.concatenate([tf.reshape(alphas_layer, [batch_size*jmax*num_frames_input]), tf.reshape(image_input, [batch_size*num_frames_input*2*nx*nx]), tf.reshape(diversity_input, [batch_size*num_frames_input*2*nx*nx])])
        obj, _ = psf_.deconvolve(x)
        print("obj", obj[0])
        return obj[0]
    '''

    def train(self):
        jmax = self.jmax
        model = self.model

        #print(self.Ds_train.shape, self.objs_train.shape, self.Ds_validation.shape, self.objs_validation.shape)
        
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
        
        # TODO: find out why it doesnt work with datasets
        #ds = tf.data.Dataset.from_tensors((self.Ds_train, output_data_train)).batch(batch_size)
        #ds_val = tf.data.Dataset.from_tensors((self.Ds_validation, output_data_validation)).batch(batch_size)
        
        
        for epoch in np.arange(n_epochs_2):

                #output_data_train = np.concatenate((output_data_train, self.Ds_train), axis=3)
                #output_data_validation = np.concatenate((output_data_validation, self.Ds_validation), axis=3)
            #history = model.fit(x=ds,
            #            epochs=1,
            #            #batch_size=batch_size,
            #            shuffle=True,
            #            validation_data=ds_val,
            #            #callbacks=[keras.callbacks.TensorBoard(log_dir='model_log')],
            #            verbose=1)
            history = model.fit(x=[self.Ds_train, self.diversities_train], y=output_data_train,
                        epochs=n_epochs_1,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=[[self.Ds_validation, self.diversities_validation], output_data_validation],
                        #callbacks=[keras.callbacks.TensorBoard(log_dir='model_log')],
                        verbose=1,
                        steps_per_epoch=None)
            
            
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
            pred_alphas = alphas_layer_model.predict([self.Ds, self.diversities], batch_size=batch_size)
        except ValueError:
            pred_alphas = None
        #pred_alphas = intermediate_layer_model.predict([self.Ds, self.objs, np.tile(self.Ds, [1, 1, 1, 16])], batch_size=1)
        if nn_mode == MODE_2:
            pred_Ds = model.predict([self.Ds, self.diversities], batch_size=batch_size)
        else:
            pred_Ds = None
        #pred_Ds = model.predict([self.Ds, self.objs], batch_size=1)
        #predicted_coefs = model.predict(Ds_train[0:n_test])
    
        #self.Ds_reconstr = np.array(self.Ds_train.shape[0], 1, self.Ds_train.shape[2], self.Ds_train.shape[3])
    
        #self.Ds_reconstr = np.array(self.Ds_train.shape[0], 1, self.Ds_train.shape[2], self.Ds_train.shape[3])
        obj_ids_test = []
        objs_reconstr = []
        i = 0
        while len(obj_ids_test) < n_test and i < len(self.objs):
            
            obj = self.objs[i]#np.reshape(self.objs[i], (self.nx, self.nx))
            found = False

            ###################################################################            
            # Just to plot results only for different objects
            for obj_id in obj_ids_test:
                if obj_id == self.obj_ids[i]:
                    found = True
                    break
            if found:
                i += 1
                continue
            ###################################################################            
            obj_ids_test.append(self.obj_ids[i])

            DF = np.zeros((num_frames_input, 2, self.nx, self.nx), dtype="complex")
            #DF = np.zeros((num_frames_input, 2, 2*self.pupil.shape[0]-1, 2*self.pupil.shape[0]-1), dtype="complex")
            for l in np.arange(num_frames_input):
                D = self.Ds[i, :, :, 2*l]
                D_d = self.Ds[i, :, :, 2*l+1]
                #D = misc.sample_image(self.Ds[i, :, :, 2*l], (2.*self.pupil.shape[0] - 1)/nx)
                #D_d = misc.sample_image(self.Ds[i, :, :, 2*l+1], (2.*self.pupil.shape[0] - 1)/nx)
                DF[l, 0] = fft.fft2(D)
                DF[l, 1] = fft.fft2(D_d)
            
            if pred_alphas is not None:
                diversity = np.concatenate((self.diversities[i, :, :, 0], self.diversities[i, :, :, 1]))
                #diversity = np.concatenate((self.diversities[i, :, :, 0][nx//4:nx*3//4,nx//4:nx*3//4], self.diversities[i, :, :, 1][nx//4:nx*3//4,nx//4:nx*3//4]))
                self.psf_check.coh_trans_func.set_diversity(diversity)
                obj_reconstr = self.psf_check.deconvolve(DF, alphas=np.reshape(pred_alphas[i], (num_frames_input, jmax)), gamma=gamma, do_fft = True, fft_shift_before = False, ret_all=False, a_est=None, normalize = False)
                obj_reconstr = fft.ifftshift(obj_reconstr)
                
                #obj_reconstr = self.deconvolve(num_frames_input, pred_alphas[i], self.diversities[i], self.Ds[i])
                
                objs_reconstr.append(obj_reconstr)
                pred_Ds = self.psf_check.convolve(obj, alphas=np.reshape(pred_alphas[i], (num_frames_input, jmax)))
                #pred_Ds = self.psf_check.convolve(misc.sample_image(obj, (2.*self.pupil.shape[0] - 1)/nx), alphas=np.reshape(pred_alphas[i], (num_frames_input, jmax)))
                pred_Ds  = np.transpose(pred_Ds, (0, 2, 3, 1))
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
                my_test_plot.colormap(obj - obj_reconstr, [row, 2])
                #my_test_plot.colormap(misc.sample_image(obj, (2.*self.pupil.shape[0] - 1)/nx) - obj_reconstr, [row, 2])
                row += 1
            if pred_Ds is not None:
                my_test_plot.colormap(self.Ds[i, :, :, 0], [row, 0])
                my_test_plot.colormap(pred_Ds[0, :, :, 0], [row, 1])
                my_test_plot.colormap(np.abs(self.Ds[i, :, :, 0] - pred_Ds[0, :, :, 0]), [row, 2])
                #my_test_plot.colormap(np.abs(misc.sample_image(self.Ds[i, :, :, 0], (2.*self.pupil.shape[0] - 1)/nx) - pred_Ds[0, :, :, 0]), [row, 2])
                row += 1
                my_test_plot.colormap(self.Ds[i, :, :, 1], [row, 0])
                my_test_plot.colormap(pred_Ds[0, :, :, 1], [row, 1])
                my_test_plot.colormap(np.abs(self.Ds[i, :, :, 1] - pred_Ds[0, :, :, 1]), [row, 2])
                #my_test_plot.colormap(np.abs(misc.sample_image(self.Ds[i, :, :, 1], (2.*self.pupil.shape[0] - 1)/nx) - pred_Ds[0, :, :, 1]), [row, 2])

            my_test_plot.save(dir_name + "/train_results" + str(i) + ".png")
            my_test_plot.close()
            
            i += 1

 
        #######################################################################
    
    def coords_of_pos(self, coords, positions, pos):
        #print("pos", pos)
        max_pos = np.max(positions, axis = 0)
        if pos[0] < 0 or pos[1] < 0:
            # extrapolate left coord
            coord0 = self.coords_of_pos(coords, positions, [0, 0])
            if max_pos[0] == 0:
                if max_pos[1] == 0: # We have only single patch
                    coord1 = coord0 - [nx, nx] + [nx//10, nx//10]
                else:
                    coord1 = np.array([coord0[0] - nx + nx//10, 2*coord0[1] - self.coords_of_pos(coords, positions, [0, 1])[1]])
            elif max_pos[1] == 0:
                coord1 = np.array([2*coord0[0] - self.coords_of_pos(coords, positions, [1, 0])[0], coord0[1] - nx + nx//10])
            else:                
                coord1 = 2*coord0 - self.coords_of_pos(coords, positions, [1, 1])
            if pos[0] < 0:
                if pos[1] < 0:
                    return coord1
                else:
                    coord0 = self.coords_of_pos(coords, positions, [0, pos[1]])
                    return np.array([coord1[0], coord0[1]])
            else:
                coord0 = self.coords_of_pos(coords, positions, [pos[0], 0])
                return np.array([coord0[0], coord1[1]])
        #print("max_pos", max_pos, positions)
        if pos[0] > max_pos[0] or pos[1] > max_pos[1]:
            # extrapolate left coord
            coord0 = self.coords_of_pos(coords, positions, max_pos)
            if max_pos[0] == 0:
                if max_pos[1] == 0: # We have only single patch
                    coord1 = coord0 - [nx, nx] + [nx//10, nx//10]
                else:
                    coord1 = np.array([coord0[0] - nx + nx//10, 2*coord0[1] - self.coords_of_pos(coords, positions, max_pos - [0, 1])[1]])
            elif max_pos[1] == 0:
                coord1 = np.array([2*coord0[0] - self.coords_of_pos(coords, positions, max_pos - [1, 0])[0], coord0[1] - nx + nx//10])
            else:
                coord1 = 2*coord0 - self.coords_of_pos(coords, positions, max_pos - [1, 1])
            if pos[0] > max_pos[0]:
                if pos[1] > max_pos[1]:
                    return coord1
                else:
                    coord0 = self.coords_of_pos(coords, positions, [max_pos[0], pos[1]])
                    return np.array([coord1[0], coord0[1]])
            else:
                coord0 = self.coords_of_pos(coords, positions, [pos[0], max_pos[1]])
                return np.array([coord0[0], coord1[1]])
        filtr = np.all(positions == pos, axis=1)
        #print("pos, filtr", pos, filtr)
        return coords[filtr][0]
    
    def crop(self, i, coords, positions):
        nx = self.nx
        coord = coords[i]
        pos = positions[i]
        top_left_coord = self.coords_of_pos(coords, positions, pos - [1, 1]) + [nx, nx]
        bottom_right_coord = self.coords_of_pos(coords, positions, pos + [1, 1])
        print("top_left_coord, bottom_right_coord", top_left_coord, bottom_right_coord)
        
        top_left_coord  = (top_left_coord + coord)//2
        bottom_right_coord = (bottom_right_coord + coord + [nx, nx])//2
        top_left_delta = top_left_coord - coord 
        bottom_right_delta = bottom_right_coord - coord - [nx, nx]
    
        print("pos, coord, i", pos, coord, i)
        return top_left_coord, bottom_right_coord, top_left_delta, bottom_right_delta
    
    def test(self, Ds_, objs, diversity, positions, coords):
        estimate_full_image = True
        if coords is None:
            estimate_full_image = False
        #print("positions, coords", positions, coords)
        if objs is None:
            # Just generate dummy array in case we don't have true object data
            objs = np.zeros((Ds_.shape[0], Ds_.shape[3], Ds_.shape[4]))
        
        jmax = self.jmax
        model = self.model
        
        num_frames = Ds_.shape[1]
        #num_objects = Ds_.shape[0]

        Ds, objs, diversities, num_frames, obj_ids, positions, coords = convert_data(Ds_, objs, diversity, positions, coords)
        #print("positions1, coords1", positions, coords)
        med = np.median(Ds, axis=(1, 2), keepdims=True)
        std = np.std(Ds, axis=(1, 2), keepdims=True)
        Ds -= med
        Ds = self.hanning.multiply(Ds, axis=1)
        #Ds += med
        Ds /= std
        #Ds = np.transpose(np.reshape(Ds_, (num_frames*num_objects, Ds_.shape[2], Ds_.shape[3], Ds_.shape[4])), (0, 2, 3, 1))
        #objs = objs[:num_objects]
        #objs = np.reshape(np.repeat(objs, num_frames, axis=0), (num_frames*objs.shape[0], objs.shape[1], objs.shape[2]))
        
        #objs = np.tile(objs, (num_frames, 1, 1))
        #objs = np.reshape(objs, (len(objs), -1))

        # Shuffle the data
        #random_indices = random.choice(len(Ds), size=len(Ds), replace=False)
        #Ds = Ds[random_indices]
        #objs = objs[random_indices]
        #diversities = diversities[random_indices]

        #Ds = np.zeros((num_objects, 2*num_frames, Ds_.shape[3], Ds_.shape[4]))
        #for i in np.arange(num_objects):
        #    for j in np.arange(num_frames):
        #        Ds[i, 2*j] = Ds_[j, i, 0]
        #        Ds[i, 2*j+1] = Ds_[j, i, 1]

        start = time.time()    
        #pred_alphas = intermediate_layer_model.predict([Ds, np.zeros_like(objs), np.tile(Ds, [1, 1, 1, 16])], batch_size=1)
        try:
            intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("alphas_layer").output)
            pred_alphas = intermediate_layer_model.predict([Ds, diversities], batch_size=batch_size)
        except ValueError:
            pred_alphas = None
            
        Ds *= std
            
        end = time.time()
        print("Prediction time: " + str(end - start))

        #obj_reconstr_mean = np.zeros((self.nx-1, self.nx-1))
        #DFs = np.zeros((len(objs), 2, 2*self.nx-1, 2*self.nx-1), dtype='complex') # in Fourier space
        
        obj_ids_test = []
        
        cropped_objs = []
        cropped_reconstrs = []
        cropped_coords = []
        
        full_shape = np.zeros(2, dtype="int")
        
        #print("coords, pos", coords, positions)
        
        for i in np.arange(len(objs)):
            #if len(obj_ids_test) >= n_test_objects:
            #    break
            obj = objs[i]#np.reshape(self.objs[i], (self.nx, self.nx))
            found = False
            ###################################################################            
            # Just to plot results only for different objects
            for obj_id in obj_ids_test:
                if obj_id == obj_ids[i]:
                    found = True
                    break
            if found:
                continue
            ###################################################################            
            obj_ids_test.append(obj_ids[i])
            
            if estimate_full_image:
                top_left_coord, bottom_right_coord, top_left_delta, bottom_right_delta = self.crop(i, coords, positions)
                print("Crop:", top_left_coord, bottom_right_coord, top_left_delta, bottom_right_delta)
                cropped_obj = obj[top_left_delta[0]:bottom_right_delta[0], top_left_delta[1]:bottom_right_delta[1]]
                cropped_objs.append(cropped_obj)
                cropped_coords.append(top_left_coord)
                full_shape += cropped_obj.shape
                print("cropped_obj.shape", cropped_obj.shape, top_left_coord)

            # Find all other realizations of the same object
            DFs = []
            alphas = []
            if pred_alphas is not None:
                for j in np.arange(i, len(objs)):
                    if obj_ids[j] == obj_ids[i]:
                        for l in np.arange(num_frames_input):
                            D = Ds[j, :, :, 2*l]
                            D_d = Ds[j, :, :, 2*l+1]
                            #D = misc.sample_image(Ds[j, :, :, 2*l], (2.*self.pupil.shape[0] - 1)/nx)
                            #D_d = misc.sample_image(Ds[j, :, :, 2*l+1], (2.*self.pupil.shape[0] - 1)/nx)
                            DF = fft.fft2(D)
                            DF_d = fft.fft2(D_d)
                            DFs.append(np.array([DF, DF_d]))
                            alphas.append(pred_alphas[j, l*jmax:(l+1)*jmax])
                            #if n_test_frames is not None and len(alphas) >= n_test_frames:
                            #    break
                    #if len(alphas) >= n_test_frames:
                    #    break
            DFs = np.asarray(DFs, dtype="complex")
            alphas = np.asarray(alphas)
            print("alphas", len(alphas), len(DFs))
            
            #obj_reconstr = psf_check.deconvolve(np.array([[DF, DF_d]]), alphas=np.array([pred_alphas[i]]), gamma=gamma, do_fft = True, fft_shift_before = False, ret_all=False, a_est=None, normalize = False)
            #obj_reconstr = fft.ifftshift(obj_reconstr[0])
            #obj_reconstr_mean += obj_reconstr

            if len(alphas) > 0:
                diversity = np.concatenate((diversities[i, :, :, 0], diversities[i, :, :, 1]))
                #diversity = np.concatenate((diversities[i, :, :, 0][nx//4:nx*3//4,nx//4:nx*3//4], diversities[i, :, :, 1][nx//4:nx*3//4,nx//4:nx*3//4]))
                self.psf_check.coh_trans_func.set_diversity(diversity)
                obj_reconstr = self.psf_check.deconvolve(DFs, alphas=alphas, gamma=gamma, do_fft = True, fft_shift_before = False, 
                                                         ret_all=False, a_est=None, normalize = False, fltr=self.filter)
                obj_reconstr = fft.ifftshift(obj_reconstr)

                if estimate_full_image:
                    cropped_reconstrs.append(obj_reconstr[top_left_delta[0]:bottom_right_delta[0], top_left_delta[1]:bottom_right_delta[1]])
                #obj_reconstr_mean += obj_reconstr
    
                #my_test_plot = plot.plot(nrows=1, ncols=2)
                #my_test_plot.colormap(np.reshape(objs[i], (self.nx, self.nx)), [0])
                #my_test_plot.colormap(obj_reconstr, [1])
                #my_test_plot.save("test_results_mode" + str(nn_mode) + "_" + str(i) + ".png")
                #my_test_plot.close()
            else:
                obj_reconstr  = None
            n_rows = 1
            if obj_reconstr is not None:
                n_rows += 1
            my_test_plot = plot.plot(nrows=n_rows, ncols=2)
            row = 0
            obj = objs[i]#np.reshape(objs[i], (self.nx, self.nx))
            if obj_reconstr is not None:
                my_test_plot.colormap(obj, [row, 0], show_colorbar=True, colorbar_prec=2)
                my_test_plot.colormap(obj_reconstr, [row, 1])
                row += 1
            my_test_plot.colormap(Ds[i, :, :, 0], [row, 0])
            my_test_plot.colormap(Ds[i, :, :, 1], [row, 1])
            my_test_plot.save(dir_name + "/test_results" + str(i) +".png")
            my_test_plot.close()

        if estimate_full_image:
            max_pos = np.max(positions, axis = 0)
            min_coord = np.min(cropped_coords, axis = 0)
            full_shape[0] = full_shape[0] // (max_pos[1] + 1)
            full_shape[1] = full_shape[1] // (max_pos[0] + 1)
            print("full_shape", full_shape)
            full_obj = np.zeros(full_shape)
            full_reconstr = np.zeros(full_shape)
            for i in np.arange(len(cropped_objs)):
                x = cropped_coords[i][0]-min_coord[0]
                y = cropped_coords[i][1]-min_coord[1]
                s = cropped_objs[i].shape
                print(x, y, s)
                full_obj[x:x+s[0],y:y+s[1]] = cropped_objs[i]
                full_reconstr[x:x+s[0],y:y+s[1]] = cropped_reconstrs[i]
            my_test_plot = plot.plot(nrows=1, ncols=2)
            my_test_plot.colormap(full_obj, [0], show_colorbar=True, colorbar_prec=2)
            my_test_plot.colormap(full_reconstr, [1])
            my_test_plot.save(dir_name + "/test_results.png")
            my_test_plot.close()
            


if train:

    Ds, objs, pupil, modes, diversity, true_coefs, positions, coords = load_data()


    #mean = np.mean(Ds, axis=(3, 4), keepdims=True)
    #std = np.std(Ds, axis=(3, 4), keepdims=True)
    #Ds -= mean
    #Ds /= np.median(Ds, axis=(3, 4), keepdims=True)
    
    n_train = int(len(Ds)*.75)
    print("n_train, n_test", n_train, len(Ds) - n_train)
    print("num_frames", Ds.shape[1])
    
    Ds_train = Ds[:n_train]
    Ds_test = Ds[n_train:, :10] # Take only 10 frames for testing purposes
    if objs is not None:
        objs_train = objs[:n_train]
        objs_test = objs[n_train:]
    else:
        objs_train = None
        objs_test = None
    
    if positions is not None:
        positions_train = positions[:n_train]
        positions_test = positions[n_train:]
    else:
        positions_train = None
        positions_test = None

    #if coords is not None:
    #    coords_test = coords[n_train:]
    #else:
    coords_test = None
    
    nx = Ds.shape[3]
    jmax = len(modes)

    #Ds_mean = np.mean(Ds, axis=(2,3))
    #Ds_std = np.std(Ds, axis=(2,3))
    #Ds -= np.tile(np.reshape(Ds_mean, (Ds_mean.shape[0], Ds_mean.shape[0], 1, 1)), (1, 1, nx, nx))
    #Ds /= np.tile(np.reshape(Ds_std, (Ds_mean.shape[0], Ds_mean.shape[0], 1, 1)), (1, 1, nx, nx))
    
    my_test_plot = plot.plot()
    my_test_plot.colormap(Ds[0, 0, 0], show_colorbar=True)
    my_test_plot.save(dir_name + "/D0.png")
    my_test_plot.close()
    
    my_test_plot = plot.plot()
    my_test_plot.colormap(Ds[0, 0, 1])
    my_test_plot.save(dir_name + "/D0_d.png")
    my_test_plot.close()
    
    pupil_check = pupil[nx//4:nx*3//4,nx//4:nx*3//4]
    #pupil_check[np.where(pupil_check < 0.001)] = 0.
    #pupil_check[np.where(pupil_check > 0.1)] = 1.
    #pupil_check = np.ones_like(pupil_check)
    modes_check = modes[:, nx//4:nx*3//4,nx//4:nx*3//4]
    
    my_test_plot = plot.plot()
    if len(diversity.shape) == 5:
        my_test_plot.colormap(diversity[0, 0, 1], show_colorbar=True)
    elif len(diversity.shape) == 3:
        my_test_plot.colormap(diversity[1], show_colorbar=True)
    else:
        my_test_plot.colormap(diversity, show_colorbar=True)
    my_test_plot.save(dir_name + "/diversity.png")
    my_test_plot.close()

    my_test_plot = plot.plot()
    my_test_plot.colormap(pupil, show_colorbar=True)
    my_test_plot.save(dir_name + "/pupil.png")
    my_test_plot.close()
    
    #for i in np.arange(len(modes)):
    #    my_test_plot = plot.plot()
    #    my_test_plot.colormap(modes[i], show_colorbar=True)
    #    my_test_plot.save(dir_name + f"/mode{i}.png")
    #    my_test_plot.close()
        

    ###########################################################################
    # Null check of deconvolution
    pa_check = psf.phase_aberration([])#len(modes), start_index=1)
    pa_check.set_terms(np.array([]))#np.zeros((jmax, nx//2, nx//2)))#modes)
    ctf_check = psf.coh_trans_func()
    ctf_check.set_phase_aberr(pa_check)
    ctf_check.set_pupil(pupil_check)
    #ctf_check.set_diversity(diversity[i, j])
    psf_check = psf.psf(ctf_check)

    D = misc.sample_image(Ds[0, 0, 0], (nx - 1)/nx)
    #D = plt.imread("tests/psf_tf_test_input.png")[0:95, 0:95]
    #D = psf_check.critical_sampling(D, threshold=1e-3)

    #hanning = utils.hanning(D.shape[0], 20)
    #med = np.median(D)
    #D -= med
    #D = hanning.multiply(D)
    #D += med
    #D /= med
    
    D_d = D

    #D = misc.sample_image(Ds[0, 0, 0], (2.*nx - 1)/nx)
    #D_d = misc.sample_image(Ds[0, 0, 1], (2.*nx - 1)/nx)
    DF = fft.fft2(D)
    #DF[np.where(np.abs(DF) < np.std(D)/10)] = 0.
    DF[-90:, -90:] = 0.
    D1 = fft.ifft2(DF).real
    DF_d = DF#fft.fft2(D)#fft.fft2(D_d)
            
    #diversity = np.concatenate((diversity[0, :, :, 0], diversity[0, :, :, 1]))
    #self.psf_check.coh_trans_func.set_diversity(diversity)
    psf_check.coh_trans_func.set_diversity(np.zeros((2, nx//2, nx//2)))
    obj_reconstr = psf_check.deconvolve(np.array([[DF, DF_d]]), alphas=None, gamma=gamma, do_fft = True, fft_shift_before = False, 
                                        ret_all=False, a_est=None, normalize = False)
    obj_reconstr = fft.ifftshift(obj_reconstr)
    #D1 = psf_check.convolve(D)

    my_test_plot = plot.plot(nrows=1, ncols=3)
    my_test_plot.colormap(D, [0], show_colorbar=True)
    my_test_plot.colormap(D1, [1])
    my_test_plot.colormap(obj_reconstr, [2])

    my_test_plot.save(dir_name + "/null_deconv.png")
    my_test_plot.close()

    ###########################################################################

    model = nn_model(jmax, nx, num_frames, num_objs, pupil, modes)

    for rep in np.arange(0, num_reps):
        model.set_data(Ds_train, objs_train, diversity, positions_train)
        print("Rep no: " + str(rep))
    
        model.train()

        model.test(Ds_test, objs_test, diversity, positions_test, coords_test)
        
        #if np.mean(model.validation_losses[-10:]) > np.mean(model.validation_losses[-20:-10]):
        #    break
        model.validation_losses = model.validation_losses[-20:]
else:

    #in_dir = "images"
    #image_file = None#"icont"
    #image_size = nx
    #tile=False
    #scale=1.0

    #num_angles = 1
    #num_subimages = n_test_objects

    #images = gen_images.gen_images(in_dir, None, image_file, image_size, tile, scale, num_subimages, num_angles, ret=True)
    #Ds, images, pupil, modes, diversity, true_coefs = gen_data.gen_data(images, n_test_frames, num_images=num_objs)

    
    Ds, objs, pupil, modes, diversity, true_coefs, positions, coords = load_data(test_data_file)

    if n_test_objects is None:
        n_test_objects = Ds.shape[0]
    if n_test_frames is None:
        n_test_frames = Ds.shape[1]
    n_test_objects = min(Ds.shape[0], n_test_objects)
    n_test_frames = min(Ds.shape[1], n_test_frames)
    
    max_pos = np.max(positions, axis = 0)

    print("n_test_objects", n_test_objects)
    print(max_pos)
    max_pos = np.round(max_pos*np.sqrt(n_test_objects/len(Ds))).astype(int)
    print(max_pos)
    filtr = np.all(positions < max_pos, axis=1)

    Ds = Ds[filtr, :n_test_frames]
    objs = objs[filtr]
    positions = positions[filtr]
    coords = coords[filtr]
    
    print(positions)
    print(coords)

    #mean = np.mean(Ds, axis=(3, 4), keepdims=True)
    #std = np.std(Ds, axis=(3, 4), keepdims=True)
    #Ds -= mean
    #Ds /= np.median(Ds, axis=(3, 4), keepdims=True)
    
    nx = Ds.shape[3]
    jmax = len(modes)
    
    
    model = nn_model(jmax, nx, num_frames, num_objs, pupil, modes)
    
    model.test(Ds, objs, diversity, positions, coords)

#logfile.close()
