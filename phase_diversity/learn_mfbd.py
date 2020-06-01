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

#tf.compat.v1.disable_eager_execution()
#from multiprocessing import Process

import math

import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import zarr
import pickle
import os.path
import numpy.fft as fft
import matplotlib.pyplot as plt

import time
#import scipy.signal as signal

gamma = 1.0

MODE_1 = 1 # aberrated images --> wavefront coefs --> MFBD loss
MODE_2 = 2 # aberrated images --> wavefront coefs --> object (using MFBD formula) --> aberrated images
MODE_3 = 3 # aberrated images --> wavefront coefs --> object (using MFBD formula) --> aberrated images
nn_mode = MODE_2

# Meant for smoothing alpha curves (seems useless)
smooth_window = 0 

#logfile = open(dir_name + '/log.txt', 'w')
#def print(*xs):
#    for x in xs:
#        logfile.write('%s' % x)
#    logfile.write("\n")
#    logfile.flush()
    
i = 1
dir_name = None
if len(sys.argv) > i:
    dir_name = sys.argv[i]
i +=1

if len(sys.argv) > i:
    nn_mode = int(sys.argv[i])
i +=1

train = True
if len(sys.argv) > i:
    if sys.argv[i].upper() == "TEST":
        train = False
i +=1

if train:
    data_file = "Ds"
else:
    data_file = "Ds_test"

if len(sys.argv) > i:
    data_file = sys.argv[i]
i +=1

n_test_frames = None
if len(sys.argv) > i:
    n_test_frames = int(sys.argv[i])
i +=1

n_test_objects = None
if len(sys.argv) > i:
    n_test_objects = int(sys.argv[i])

if nn_mode == MODE_1:
    
    shuffle0 = False
    shuffle1 = False
    shuffle2 = False
    
    num_reps = 1000

    n_epochs_2 = 10
    n_epochs_1 = 1
    
    # How many frames to use in training
    num_frames = 320
    # How many objects to use in training
    num_objs = 80#0#None
    
    # How many frames of the same object are sent to NN input
    # Must be power of 2
    num_frames_input = 1
    
    batch_size = 64
    n_channels = 8
    
    sum_over_batch = True
    
    zero_avg_tiptilt = True
    
elif nn_mode == MODE_2:

    shuffle0 = False
    shuffle1 = False
    shuffle2 = False
    
    num_reps = 1000
    
    n_epochs_2 = 20
    n_epochs_1 = 1
    
    n_epochs_mode_2 = 10
    mode_2_index = 1
    
    # How many frames to use in training
    num_frames = 640
    # How many objects to use in training
    num_objs = 80#None
    
    # How many frames of the same object are sent to NN input
    # Must be power of 2
    num_frames_input = 1
    
    batch_size = 32
    n_channels = 8
    
    sum_over_batch = True
    
    zero_avg_tiptilt = True

    num_alphas_input = 10

else:
    
    shuffle0 = False
    shuffle1 = True
    shuffle2 = False
    
    num_reps = 1000
    
    n_epochs_2 = 4
    n_epochs_1 = 1
    
    n_epochs_mode_2 = 10
    
    # How many frames to use in training
    num_frames = 640
    # How many objects to use in training
    num_objs = 10#None
    
    # How many frames of the same object are sent to NN input
    # Must be power of 2
    num_frames_input = 1
    
    batch_size = 32
    n_channels = 8
    
    sum_over_batch = True

    zero_avg_tiptilt = True
    num_alphas_input = 10

no_shuffle = not shuffle0 and not shuffle1 and not shuffle2

assert(num_frames % num_frames_input == 0)
if sum_over_batch:
    if not train:
        assert((n_test_frames // num_frames_input) % batch_size == 0)

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

gpu_id = '/device:GPU:0'

if train:
    sys.stdout = open(dir_name + '/log.txt', 'a')
else:
    gpu_id = '/device:GPU:1'

    
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

n_gpus = 1#len(gpus)

if n_gpus >= 1:
    from numba import cuda

def load_data(data_file):
    f = dir_name + '/' + data_file + ".zarr"
    if os.path.exists(f):
        loaded = zarr.open(f, 'r')
    else:
        f = dir_name + '/' + data_file + ".npz"
        if os.path.exists(f):
            loaded = np.load(f)
        else:
            raise "No data found"
    Ds = loaded['Ds']
    try:
        objs = loaded['objs']
    except:
        objs = None
    pupil = loaded['pupil']
    modes = loaded['modes']
    diversity = loaded['diversity']
    try:
        coefs = loaded['alphas']
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
    try:
        model.load_weights(model_file)
    except:
        model_file = dir_name + '/weights.h5'
        try:
            model.load_weights(model_file)
        except:
            print("No model weights found")
            return None, None
    nn_mode, params = pickle.load(open(dir_name + '/params.dat', 'rb'))
    return nn_mode, params

def save_weights(model, params):
    model.save_weights(dir_name + '/weights.tf')
    with open(dir_name + '/params.dat', 'wb') as f:
        pickle.dump((nn_mode, params), f, protocol=4)

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
    assert(Ds_in.shape[2] == 2)
    assert(Ds_in.shape[0] == objs_in.shape[0])
    num_objects = Ds_in.shape[0]
    num_frames = Ds_in.shape[1]
    Ds_out = np.zeros(((num_frames-num_frames_input+1)*num_objects, Ds_in.shape[3], Ds_in.shape[4], Ds_in.shape[2]*num_frames_input))
    if objs_in is not None:
        objs_out = np.zeros(((num_frames-num_frames_input+1)*num_objects, objs_in.shape[1], objs_in.shape[2]))
    else:
        objs_out  = None
    if diversity_in is not None:
        diversity_out = np.zeros(((num_frames-num_frames_input+1)*num_objects, Ds_in.shape[3], Ds_in.shape[4], Ds_in.shape[2]))
    else:
        diversity_out = None
    ids = np.zeros((num_frames-num_frames_input+1)*num_objects, dtype='int')
    positions_out = np.zeros(((num_frames-num_frames_input+1)*num_objects, 2), dtype='int')
    coords_out = np.zeros(((num_frames-num_frames_input+1)*num_objects, 2), dtype='int')
        
    k = 0
    for i in np.arange(num_objects):
        l = 0
        Ds_k = np.zeros((Ds.shape[3], Ds_in.shape[4], Ds_in.shape[2]*num_frames_input))
        diversity_k = np.zeros((Ds_in.shape[3], Ds_in.shape[4], Ds_in.shape[2]))
        for j in np.arange(num_frames):
            Ds_k[:, :, 2*l] = Ds_in[i, j, 0, :, :]
            Ds_k[:, :, 2*l+1] = Ds_in[i, j, 1, :, :]
            if diversity_out is not None and l == 0:
                if positions is None:
                    if len(diversity_in.shape) == 3:
                        # Diversities both for focus and defocus image
                        #diversity_out[k, :, :, 0] = diversity_in[0]
                        for div_i in np.arange(diversity_in.shape[0]):
                            diversity_k[:, :, 1] += diversity_in[div_i]
                    else:
                        assert(len(diversity_in.shape) == 2)
                        # Just a defocus
                        diversity_k[:, :, 1] = diversity_in
                else:
                    assert(len(diversity_in.shape) == 5)
                    #for div_i in np.arange(diversity_in.shape[2]):
                    #    #diversity_out[k, :, :, 0] = diversity_in[positions[i, 0], positions[i, 1], 0]
                    #    diversity_out[k, :, :, 1] += diversity_in[positions[i, 0], positions[i, 1], div_i]
                    #    #diversity_out[k, :, :, 1] = diversity_in[positions[i, 0], positions[i, 1], 1]
                    diversity_k[:, :, 1] += diversity_in[positions[i, 0], positions[i, 1], 1]
            l += 1
            if l >= num_frames_input:
                Ds_out[k] = Ds_k
                if diversity_out is not None:
                    diversity_out[k] = diversity_k
                if objs_out is not None:
                    objs_out[k] = objs_in[i]
                ids[k] = i
                if positions is not None:
                    positions_out[k] = positions[i]
                if coords is not None:
                    coords_out[k] = coords[i]
                l = 0
                k += 1
                Ds_k = np.zeros((Ds.shape[3], Ds_in.shape[4], Ds_in.shape[2]*num_frames_input))
                diversity_k = np.zeros((Ds_in.shape[3], Ds_in.shape[4], Ds_in.shape[2]))
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


#import datetime

class MyCustomCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, model):
        self.model = model
        self.counter = 0

    #def on_train_batch_begin(self, batch, logs=None):
    #    print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

    #def on_train_batch_end(self, batch, logs=None):
    #    self.counter += batch*batch_size
    #    #print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))
    #    if self.counter >= 10000:
    #        save_weights(self.model)
    #        self.counter = 0

    #def on_test_batch_begin(self, batch, logs=None):
    #    print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

    #def on_test_batch_end(self, batch, logs=None):
    #    print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

class nn_model:       
    
    
    def __init__(self, jmax, nx, num_frames, num_objs, pupil, modes):
        
        self.jmax = jmax
        self.num_frames = num_frames
        assert(num_frames_input <= self.num_frames)
        self.num_objs = num_objs
        self.nx = nx
        self.hanning = utils.hanning(nx, 10)
        self.filter = utils.create_filter(nx, freq_limit = 0.4)
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
        batch_size_per_gpu = max(1, batch_size//max(1, n_gpus))
        self.psf = psf_tf.psf_tf(ctf, num_frames=num_frames_input, batch_size=batch_size_per_gpu, set_diversity=True, 
                                 mode=nn_mode, sum_over_batch=sum_over_batch, fltr=self.filter)
        print("batch_size_per_gpu, num_frames_input", batch_size_per_gpu, num_frames_input)
        
        self.psf_test = psf_tf.psf_tf(ctf, num_frames=n_test_frames, batch_size=1, set_diversity=True, 
                                      mode=nn_mode, sum_over_batch=sum_over_batch, fltr=self.filter)
        
        num_defocus_channels = 2#self.num_frames*2

        #self.strategy = tf.distribute.MirroredStrategy()
        #with self.strategy.scope():
        with tf.device(gpu_id):

            image_input = keras.layers.Input((nx, nx, num_defocus_channels*num_frames_input), name='image_input')
            diversity_input = keras.layers.Input((nx, nx, num_defocus_channels), name='diversity_input')
            image_input1 = image_input
            if nn_mode >= MODE_2:
                DD_DP_PP_input = keras.layers.Input((4, nx, nx), name='DD_DP_PP_input')
                tt_sums_input = keras.layers.Input((2), name='tt_sums_input')
                alphas_input = keras.layers.Input((num_alphas_input*num_frames_input*jmax), name='alphas_input')
                if nn_mode == MODE_3:
                    #alphas_input = keras.layers.Input((nx, num_defocus_channels*num_frames_input), name='alphas_input')
                    image_diff_input = keras.layers.Input((nx, nx, num_defocus_channels*num_frames_input), name='image_diff_input')
                    image_diff_input1 = tf.reshape(image_diff_input, [batch_size_per_gpu, nx, nx, num_defocus_channels*num_frames_input])
                    image_input1 = tf.concat([image_input, image_diff_input1], axis=3)
            #else:
            #    raise Exception("Unsupported mode")
    
            model, nn_mode_ = load_model()
            
            if model is None:
                print("Creating model")
                nn_mode_ = nn_mode
                
                '''
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
                '''
                
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
                
                def seq_block(x):
                    x_out = tf.slice(x, [0, 0], [1, 1024])
                    x_i = x_out
                    for i in np.arange(1, batch_size_per_gpu):
                        x_i1 = tf.concat([tf.slice(x, [i, 0], [1, 1024]), x_i], axis=1)
                        x_i = keras.layers.Dense(1024, activation='relu')(x_i1)
                        x_out = tf.concat([x_out, x_i], axis=0)
                    return x_out
                            
                #def tile(x):
                #    return tf.tile(tf.reshape(tf.reshape(x, [batch_size_per_gpu*1024]), [1, batch_size_per_gpu*1024]), [batch_size_per_gpu, 1])
                
                hidden_layer = conv_layer(image_input1, n_channels, num_convs=1)
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
                #alphas_layer = keras.layers.Dense(jmax*num_frames_input, activation='linear')(alphas_layer)
                if no_shuffle:
                    alphas_layer = tf.reshape(alphas_layer, [1, batch_size_per_gpu, 1024])
                    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True, stateful=True), merge_mode='sum')#, activation="relu")#, return_state=True)
                    #lstm2 = tf.keras.layers.LSTM(512, return_sequences=True, stateful=True, go_backwards=True)#, activation="relu")#, return_state=True)
                    alphas_layer = lstm(alphas_layer)
                    alphas_layer = tf.reshape(alphas_layer, [batch_size_per_gpu, 512])
                    #alphas_layer = tf.reshape(alphas_layer, [batch_size_per_gpu*2, 512])
                    #alphas_layer1 = tf.slice(alphas_layer, [0, 0], [batch_size_per_gpu, 512])
                    #alphas_layer2 = tf.slice(alphas_layer, [batch_size_per_gpu, 0], [batch_size_per_gpu, 512])
                    #alphas_layer = tf.concat([alphas_layer1, tf.reverse(alphas_layer2, axis=[0])], axis=1)
                    #alphas_layer = lstm2(alphas_layer)
                #alphas_layer = seq_block(alphas_layer)
                
                alphas_layer = keras.layers.Dense(512, activation='relu')(alphas_layer)
                alphas_layer = keras.layers.Dense(jmax*num_frames_input, activation='linear')(alphas_layer)
                #if nn_mode >= MODE_2:
                #    alphas_layer1 = keras.layers.Dense(1024, activation='relu')(alphas_input)
                #    alphas_layer1 = keras.layers.Dense(jmax*num_frames_input, activation='relu')(alphas_layer1)
                #    alphas_layer = keras.layers.add([alphas_layer, alphas_layer1])
                
                if zero_avg_tiptilt:
                    def zero_avg(x):
                        alphas = tf.slice(x, [0], [batch_size_per_gpu*num_frames_input*jmax])
                        alphas = tf.reshape(alphas, [batch_size_per_gpu, num_frames_input, jmax])
                        if sum_over_batch:
                            alpha_sums = tf.tile(tf.math.reduce_sum(alphas, axis=[0, 1], keepdims=True), [batch_size_per_gpu, num_frames_input, 1])
                        else:
                            alpha_sums = tf.tile(tf.math.reduce_sum(alphas, axis=1, keepdims=True), [1, num_frames_input, 1])
                        tiptilt_sums = tf.slice(alpha_sums, [0, 0, 0], [batch_size_per_gpu, num_frames_input, 2])
                        if nn_mode >= MODE_2:
                            tt_sums = tf.slice(x, [batch_size_per_gpu*num_frames_input*jmax], [batch_size_per_gpu*2])
                            tt_sums = tf.tile(tf.reshape(tt_sums, [batch_size_per_gpu, 1, 2]), [1, num_frames_input, 1])
                            tiptilt_sums = tiptilt_sums + tt_sums
                        tiptilt_means = tiptilt_sums / tf.constant(self.num_frames, dtype="float32")
                        zeros = tf.zeros([batch_size_per_gpu, num_frames_input, jmax - 2])
                        tiptilt_means = tf.concat([tiptilt_means, zeros], axis=2)
                        alphas = alphas - tiptilt_means
                        return tf.reshape(alphas, [batch_size_per_gpu, num_frames_input*jmax])
                    x = tf.reshape(alphas_layer, [batch_size_per_gpu*num_frames_input*jmax])
                    if nn_mode >= MODE_2:
                        x = tf.concat([x, tf.reshape(tt_sums_input, [batch_size_per_gpu*2])], axis=0)
                    alphas_layer = keras.layers.Lambda(zero_avg, name='alphas_layer')(x)
                else:
                    alphas_layer = keras.layers.Lambda(lambda alphas: multiply(alphas, 1.), name='alphas_layer')(alphas_layer)
                    
                #obj_layer = keras.layers.Dense(256)(obj_layer)
                #obj_layer = keras.layers.Dense(128)(obj_layer)
                #obj_layer = keras.layers.Dense(64)(obj_layer)
                #obj_layer = keras.layers.Dense(128)(obj_layer)
                #obj_layer = keras.layers.Dense(256)(obj_layer)
                #obj_layer = keras.layers.Dense(512)(obj_layer)
                #obj_layer = keras.layers.Dense(1152)(obj_layer)
               
                
                a1 = alphas_layer#tf.reshape(alphas_layer, [batch_size_per_gpu, num_frames_input*jmax])
                a2 = tf.reshape(tf.transpose(diversity_input, [0, 3, 1, 2]), [batch_size_per_gpu, 2*nx*nx])
                a3 = tf.concat([a1, a2], axis=1)

                if nn_mode == MODE_1:                    
                    
                    hidden_layer = keras.layers.concatenate([tf.reshape(a3, [batch_size_per_gpu*(num_frames_input*jmax+2*nx*nx)]), tf.reshape(image_input, [batch_size_per_gpu*num_frames_input*2*nx*nx])])
                    #hidden_layer = keras.layers.concatenate([tf.reshape(alphas_layer, [batch_size*jmax*num_frames_input]), tf.reshape(image_input, [batch_size*num_frames_input*2*nx*nx]), tf.reshape(diversity_input, [batch_size*num_frames_input*2*nx*nx])])
                    output = keras.layers.Lambda(self.psf.mfbd_loss)(hidden_layer)
                    #output = keras.layers.Lambda(lambda x: tf.reshape(tf.math.reduce_sum(x), [1]))(output)
                    #output = keras.layers.Flatten()(output)
                    #output = keras.layers.Lambda(lambda x: tf.math.reduce_sum(x))(output)

                    model = keras.models.Model(inputs=[image_input, diversity_input], outputs=output)
                elif nn_mode >= MODE_2:
                                                      
                    hidden_layer = keras.layers.concatenate([tf.reshape(a3, [batch_size_per_gpu*(num_frames_input*jmax + 2*nx*nx)]), 
                                                             tf.reshape(image_input, [batch_size_per_gpu*num_frames_input*2*nx*nx]),
                                                             tf.reshape(DD_DP_PP_input, [batch_size_per_gpu*4*nx*nx])])
                    #hidden_layer = keras.layers.concatenate([tf.reshape(alphas_layer, [batch_size*jmax*num_frames_input]), tf.reshape(image_input, [batch_size*num_frames_input*2*nx*nx]), tf.reshape(diversity_input, [batch_size*num_frames_input*2*nx*nx])])
                    output = keras.layers.Lambda(self.psf.mfbd_loss, name='output_layer')(hidden_layer)
                    
                    if nn_mode == MODE_3:
                        model = keras.models.Model(inputs=[image_input, diversity_input, DD_DP_PP_input, tt_sums_input, alphas_input, image_diff_input], outputs=output)
                    else:
                        model = keras.models.Model(inputs=[image_input, diversity_input, DD_DP_PP_input, tt_sums_input, alphas_input], outputs=output)
    
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
                if nn_mode == MODE_1:
                    loss = y_pred
                    if sum_over_batch:
                        return tf.math.reduce_sum(loss)/(nx*nx)
                    else:
                        return tf.math.reduce_sum(loss, axis=[1, 2])/(nx*nx)
                elif nn_mode >= MODE_2:
                    if sum_over_batch:
                        loss = tf.slice(y_pred, [0, 0, 0, 0], [1, 1, nx, nx])
                        return tf.math.reduce_sum(loss, axis=[1, 2])/(nx*nx)
                    else:
                        loss = tf.slice(y_pred, [0, 0, 0, 0], [batch_size_per_gpu, 1, nx, nx])
                        return tf.math.reduce_sum(loss, axis=[1, 2, 3])/(nx*nx)
                
            #self.model.compile(optimizer='adadelta', loss=mfbd_loss)#'mse')
            self.model.compile(optimizer='adadelta', loss=mfbd_loss)#'mse')

        nn_mode_, params = load_weights(model)
        
        epoch = 0
        epoch_mode_2 = 0
        val_loss = float("inf")
        
        mode_2_index_ = mode_2_index

        if nn_mode_ is not None:
            assert(nn_mode_ == nn_mode) # Model was saved with different mode
            if nn_mode_ == MODE_1:
                n_epochs_1_, n_epochs2_, epoch = params
            elif nn_mode_ >= MODE_2:
                n_epochs_1_, n_epochs_2_, n_epochs_mode_2_, epoch, epoch_mode_2, val_loss, mode_2_index_ = params
        else:
            nn_mode_ = nn_mode

        # Overwrite
        n_epochs_1_ = n_epochs_1
        n_epochs_2_ = n_epochs_2

        self.nn_mode = nn_mode_

        self.n_epochs_1 = n_epochs_1_
        self.n_epochs_2 = n_epochs_2_
        self.epoch = epoch
        if self.nn_mode >= MODE_2:
            # Overwrite
            n_epochs_mode_2_ = n_epochs_mode_2

            self.n_epochs_mode_2 = n_epochs_mode_2_
            self.epoch_mode_2 = epoch_mode_2
            self.val_loss = val_loss
            
            self.mode_2_index = mode_2_index_

    # Inputs should be grouped per object (first axis)
    def deconvolve(self, Ds, alphas, diversity, do_fft=True):
        num_objs = Ds.shape[0]
        #assert(len(alphas) == len(Ds))
        #assert(Ds.shape[3] == 2) # Ds = [num_objs, num_frames, nx, nx, 2]
        num_frames = Ds.shape[1]
        if not train:
            assert(num_frames == n_test_frames)
        self.psf_test.set_num_frames(num_frames)
        self.psf_test.set_batch_size(num_objs)
        with tf.device(gpu_id):
            alphas = tf.constant(alphas, dtype='float32')
            diversity = tf.constant(diversity, dtype='float32')
            Ds = tf.constant(Ds, dtype='float32')
            Ds = tf.reshape(tf.transpose(Ds, [0, 2, 3, 1, 4]), [num_objs, nx, nx, 2*num_frames])

            a1 = tf.reshape(alphas, [num_objs, num_frames*jmax])
            a2 = tf.reshape(diversity, [num_objs, 2*nx*nx])
            a3 = tf.concat([a1, a2], axis=1)

            #x = tf.concat([tf.reshape(a3, [num_frames*jmax+2*nx*nx]), tf.reshape(Ds, [num_frames*2*nx*nx])], axis=0)
            x = tf.concat([tf.reshape(a3, [num_objs*(num_frames*jmax+2*nx*nx)]), tf.reshape(Ds, [num_objs*num_frames*2*nx*nx])], axis=0)
            image_deconv, _ = self.psf_test.deconvolve(x, do_fft=do_fft)
            print("image_deconv", image_deconv.numpy().shape)
            return image_deconv
        
    # Inputs should be grouped per object (first axis)
    def Ds_reconstr(self, Ds, alphas, diversity):
        image_deconv = self.deconvolve(Ds, alphas, diversity, do_fft=False)
        with tf.device(gpu_id):
            image_deconv = tf.reshape(image_deconv, [alphas.shape[0], 1, self.nx, self.nx])
            image_deconv = tf.tile(image_deconv, [1, 2*alphas.shape[1], 1, 1])
            alphas = tf.constant(alphas, dtype='float32')
            diversity = tf.constant(diversity, dtype='float32')
            
            num_frames = alphas.shape[1]
            num_objs = alphas.shape[0]
            
            a1 = tf.reshape(alphas, [num_objs, num_frames*jmax])
            a2 = tf.reshape(diversity, [num_objs, 2*nx*nx])
            a3 = tf.concat([a1, a2], axis=1)
            
            self.psf_test.set_batch_size(num_objs)
            self.psf_test.set_num_frames(num_frames)
            ret_val = self.psf_test.Ds_reconstr2(image_deconv, a3)#tf.reshape(a3, [num_objs*(num_frames*jmax+2*nx*nx)]))
        #print("image_deconv", image_deconv.numpy().shape)
        return ret_val
        

    def set_data(self, Ds, objs, diversity, positions, train_perc=.8):
        assert(Ds.shape[1] >= self.num_frames)
        #assert(self.num_frames <= Ds.shape[1])
        if self.num_objs is None or self.num_objs <= 0:
            self.num_objs = Ds.shape[0]
        self.num_objs = min(self.num_objs, Ds.shape[0])
        #assert(self.num_objs <= Ds.shape[0])
        assert(Ds.shape[2] == 2)
        if objs is None:
            # Just generate dummy array in case we don't have true object data
            objs = np.zeros((Ds.shape[0], Ds.shape[3], Ds.shape[4]))
        if shuffle1:
            i1 = random.randint(0, Ds.shape[0] + 1 - self.num_objs)
            i2 = random.randint(0, Ds.shape[1] + 1 - self.num_frames)
        else:
            i1 = 0
            i2 = 0
        Ds = Ds[i1:i1+self.num_objs, i2:i2+self.num_frames]
        objs = objs[i1:i1+self.num_objs]
        if positions is not None:
            positions = positions[i1:i1+self.num_objs]
        
        self.Ds, self.objs, self.diversities, num_frames, self.obj_ids, self.positions, _s = convert_data(Ds, objs, diversity, positions)
        
        med = np.median(self.Ds, axis=(1, 2), keepdims=True)
        #std = np.std(Ds, axis=(1, 2), keepdims=True)
        self.Ds -= med
        self.Ds = self.hanning.multiply(self.Ds, axis=1)
        self.Ds += med
        ##Ds /= std
        self.Ds /= med
        
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
                    
        if shuffle2 and (not sum_over_batch or batch_size == 1):
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
        
        print("n_train, n_validation", n_train, n_validation)
        
        self.n_train = n_train
        self.n_validation = n_validation

        if sum_over_batch:
            assert((self.n_train // self.num_objs) % batch_size == 0)

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
        

    def group_per_obj(self, Ds, alphas, diversities, obj_ids, DD_DP_PP=None, tt=None):
        unique_obj_ids = np.unique(obj_ids)
        used_obj_ids = dict()
        
        num_frames = Ds.shape[0]//len(unique_obj_ids)

        Ds_per_obj = np.empty((len(unique_obj_ids), num_frames, Ds.shape[1], Ds.shape[2], Ds.shape[3]))
        if alphas is not None:
            alphas_per_obj = np.empty((len(unique_obj_ids), num_frames, alphas.shape[1]))
        else:
            alphas_per_obj = None
        diversities_per_obj = np.empty((len(unique_obj_ids), diversities.shape[1], diversities.shape[2], diversities.shape[3]))
        
        if DD_DP_PP is not None:
            DD_DP_PP_sums_per_obj = np.zeros((len(unique_obj_ids), 4, self.nx, self.nx))
        else:
            DD_DP_PP_sums_per_obj = None

        if tt is not None:
            tt_sums_per_obj = np.zeros((len(unique_obj_ids), 2))
        else:
            tt_sums_per_obj = None

        for i in np.arange(len(Ds)):
            #if sum_over_batch:
            #    obj_id = obj_ids[i*batch_size]
            #else:
            obj_id = obj_ids[i]
            if not obj_id in used_obj_ids:
                used_obj_ids[obj_id] = (len(used_obj_ids), 0)
            obj_index, frame_index = used_obj_ids[obj_id]
            Ds_per_obj[obj_index, frame_index] = Ds[i]
            if alphas is not None:
                alphas_per_obj[obj_index, frame_index] = alphas[i]
            used_obj_ids[obj_id] = (obj_index, frame_index + 1)
            diversities_per_obj[obj_index] = diversities[i]
            if DD_DP_PP is not None:
                DD_DP_PP_sums_per_obj[obj_id] += DD_DP_PP[i]
            if tt is not None:
                tt_sums_per_obj[obj_id] += np.sum(tt[i], axis=0)

        return Ds_per_obj, alphas_per_obj, diversities_per_obj, DD_DP_PP_sums_per_obj, tt_sums_per_obj


    def predict_mode2(self, Ds, diversities, DD_DP_PP, obj_ids, tt_sums, alphas_in, Ds_diff=None):
        output_layer_model = Model(inputs=self.model.input, outputs=self.model.get_layer("output_layer").output)
        
        if nn_mode == MODE_3:
            output = output_layer_model.predict([Ds, diversities, DD_DP_PP, tt_sums, alphas_in, Ds_diff], batch_size=batch_size)
            alphas_layer_model = Model(inputs=self.model.input, outputs=self.model.get_layer("alphas_layer").output)
            alphas = alphas_layer_model.predict([Ds, diversities, DD_DP_PP, tt_sums, alphas_in, Ds_diff], batch_size=batch_size)
        else:
            output = output_layer_model.predict([Ds, diversities, DD_DP_PP, tt_sums, alphas_in], batch_size=batch_size)
            alphas_layer_model = Model(inputs=self.model.input, outputs=self.model.get_layer("alphas_layer").output)
            alphas = alphas_layer_model.predict([Ds, diversities, DD_DP_PP, tt_sums, alphas_in], batch_size=batch_size)
            #alphas = None

        DD_DP_PP_out = output[:, 1:, :, :]
        #DD_DP_PP_sums = dict()
        #DD_DP_PP_counts = dict()
        assert(len(DD_DP_PP_out) == len(Ds))
        tt = np.empty((len(alphas), num_frames_input, 2))
        for frame in np.arange(num_frames_input):
            tt[:, frame] = alphas[:, frame*jmax:frame*jmax+2]
        Ds_per_obj, alphas_per_obj, diversities_per_obj, DD_DP_PP_sums_per_obj, tt_sums_per_obj = self.group_per_obj(Ds, alphas, diversities, obj_ids, DD_DP_PP_out, tt)

        num_frames = alphas_per_obj.shape[1]
        assert(num_frames == self.num_frames)
        if nn_mode == MODE_3:
            num_objs = alphas_per_obj.shape[0]
            
            self.psf_test.set_num_frames(num_frames)
            self.psf_test.set_batch_size(num_objs)

            a1 = tf.reshape(tf.constant(alphas_per_obj, dtype='float32'), [num_objs, num_frames*jmax])
            a2 = tf.reshape(tf.constant(diversities_per_obj, dtype='float32'), [num_objs, 2*nx*nx])
            a3 = tf.concat([a1, a2], axis=1)
            
            Ds_reconstrs_per_obj = self.psf_test.Ds_reconstr(DD_DP_PP_sums_per_obj[:, 1, :, :], DD_DP_PP_sums_per_obj[:, 2, :, :], DD_DP_PP_sums_per_obj[:, 3, :, :], a3)#tf.reshape(a3, [num_objs*(num_frames*jmax+2*nx*nx)]))
            Ds_reconstrs_per_obj = np.reshape(Ds_reconstrs_per_obj, (Ds_reconstrs_per_obj.shape[0], Ds_reconstrs_per_obj.shape[1], Ds_reconstrs_per_obj.shape[2], Ds_reconstrs_per_obj.shape[3]//2, 2))
            Ds_reconstrs_per_obj = np.transpose(Ds_reconstrs_per_obj, (0, 3, 1, 2, 4))

            
        for i in np.arange(len(Ds)):
            if sum_over_batch:
                if i % batch_size == 0:
                    for j in np.arange(i, i+batch_size):
                        assert(obj_ids[j] == obj_ids[i])
                    DD_DP_PP_out_batch_sum = np.sum(DD_DP_PP_out[i:i+batch_size], axis=0)
                    DD_DP_PP[i:i+batch_size] = (DD_DP_PP_sums_per_obj[obj_ids[i]] - DD_DP_PP_out_batch_sum)/batch_size
            else:
                DD_DP_PP[i] = DD_DP_PP_sums_per_obj[obj_ids[i]] - DD_DP_PP_out[i]
            if i % batch_size*num_frames_input == 0:
                for j in np.arange(i, i+batch_size*num_frames_input):
                    assert(obj_ids[j] == obj_ids[i])
                tt_batch_sum = np.sum(tt[i:i+batch_size], axis=(0, 1))
                tt_sums[i:i+batch_size] = tt_sums_per_obj[obj_ids[i]] - tt_batch_sum
            if no_shuffle:#not shuffle0 and not shuffle1 and not shuffle2:
                # Use alphas of previous frame
                for j in np.arange(0, num_alphas_input):
                    if (i + j) % (num_frames/num_frames_input) < num_alphas_input:
                        alphas_in[i, jmax*num_frames_input*j:jmax*num_frames_input*(j+1)] = np.zeros_like(alphas[i])
                    else:
                        alphas_in[i, jmax*num_frames_input*j:jmax*num_frames_input*(j+1)] = alphas[i-num_alphas_input+j]
                        
                #if i % (num_frames/num_frames_input) < 0:
                #    alphas_in[i, :jmax*num_frames_input] = alphas[i]
                #else:
                #    alphas_in[i, :jmax] = alphas[i - 1, jmax*(num_frames_input-1):jmax*num_frames_input]
                #if num_frames_input > 0:
                #    alphas_in[i, jmax:jmax*num_frames_input] = alphas[i, :jmax*(num_frames_input-1)]
                    
                
            if nn_mode == MODE_3:
                if i % Ds_reconstrs_per_obj.shape[1] == 0:
                    Ds_diff[i:i+Ds_reconstrs_per_obj.shape[1]] = Ds[i:i+Ds_reconstrs_per_obj.shape[1]] - Ds_reconstrs_per_obj[obj_ids[i]]#, i % Ds_reconstr.shape[1]]

                    if not train and i == 0:
                        ###########################################################
                        # DEBUG -- REMOVE
                        my_test_plot = plot.plot(nrows=2, ncols=2)
                        my_test_plot.colormap(Ds[i, :, :, 0], [0, 0], show_colorbar=True)
                        my_test_plot.colormap(Ds[i, :, :, 1], [0, 1])
                        my_test_plot.colormap(Ds_reconstrs_per_obj[obj_ids[i]][i, :, :, 0], [1, 0])
                        my_test_plot.colormap(Ds_reconstrs_per_obj[obj_ids[i]][i, :, :, 1], [1, 1])
                        my_test_plot.save(f"{dir_name}/reconstr{i}.png")
                        my_test_plot.close()
                        ###########################################################
            

 
    def train(self):
        jmax = self.jmax
        model = self.model

        #print(self.Ds_train.shape, self.objs_train.shape, self.Ds_validation.shape, self.objs_validation.shape)
        
        output_data_train = np.zeros(self.objs_train.shape[0])
        output_data_validation = np.zeros(self.objs_validation.shape[0])

        
        # TODO: find out why it doesnt work with datasets
        #ds = tf.data.Dataset.from_tensors((self.Ds_train, output_data_train)).batch(batch_size)
        #ds_val = tf.data.Dataset.from_tensors((self.Ds_validation, output_data_validation)).batch(batch_size)
        
        shuffle_epoch = True
        if sum_over_batch and batch_size > 1:
            shuffle_epoch = False
        if self.nn_mode == MODE_1:
            for epoch in np.arange(self.epoch, self.n_epochs_2):
                history = model.fit(x=[self.Ds_train, self.diversities_train], y=output_data_train,
                            epochs=self.n_epochs_1,
                            batch_size=batch_size,
                            shuffle=shuffle_epoch,
                            validation_data=[[self.Ds_validation, self.diversities_validation], output_data_validation],
                            #callbacks=[keras.callbacks.TensorBoard(log_dir='model_log')],
                            verbose=1,
                            steps_per_epoch=None,
                            callbacks=[MyCustomCallback(model)])
                save_weights(model, (self.n_epochs_1, self.n_epochs_2, epoch))
            self.epoch = 0
        elif self.nn_mode >= MODE_2:
            DD_DP_PP = np.zeros((len(self.Ds), 4, nx, nx))
            DD_DP_PP_train = DD_DP_PP[:self.n_train]
            DD_DP_PP_validation = DD_DP_PP[self.n_train:self.n_train+self.n_validation]
            tt_sums = np.zeros((len(self.Ds), 2))
            tt_sums_train = tt_sums[:self.n_train]
            tt_sums_validation = tt_sums[self.n_train:self.n_train+self.n_validation]
            alphas = np.zeros((len(self.Ds), num_alphas_input*num_frames_input*jmax))
            alphas_train = alphas[:self.n_train]
            alphas_validation = alphas[self.n_train:self.n_train+self.n_validation]
            Ds_diff = None
            if nn_mode == MODE_3:
                Ds_per_obj, alphas_per_obj, diversities_per_obj, DD_DP_PP_sums_per_obj, _ = self.group_per_obj(self.Ds, np.zeros((len(self.Ds), jmax)), self.diversities, self.obj_ids, DD_DP_PP)
                #Ds_per_obj, diversity_per_obj = self.group_per_obj(self.Ds, self.diversities, self.obj_ids)
                #reconstr = np.empty((len(self.Ds), nx, nx))
                #reconstr_train = reconstr[:self.n_train]
                #reconstr_validation = reconstr[self.n_train:self.n_train+self.n_validation]
                #reconstrs = dict()
                #for obj_id in np.unique(self.obj_ids):
                #    print("obj_id", obj_id)
                #    sys.stdout.flush()
                #    Ds_ = np.asarray(Ds_per_obj[obj_id])
                #    reconstrs[obj_id] = self.deconvolve(Ds_, np.zeros((len(Ds_), jmax)), diversities_per_obj[obj_id])
                #    reconstrs[obj_id] /= np.median(reconstrs[obj_id])
                Ds_reconstrs_per_obj = self.Ds_reconstr(Ds_per_obj, alphas_per_obj, diversities_per_obj).numpy()
                #Ds_reconstr_per_obj = self.psf_test.Ds_reconstr(DD_DP_PP_sums_per_obj[:, 1, :, :], DD_DP_PP_sums_per_obj[:, 2, :, :], DD_DP_PP_sums_per_obj[:, 3, :, :], alphas_per_obj)
                Ds_reconstrs_per_obj = np.reshape(Ds_reconstrs_per_obj, (Ds_reconstrs_per_obj.shape[0], Ds_reconstrs_per_obj.shape[1], Ds_reconstrs_per_obj.shape[2], Ds_reconstrs_per_obj.shape[3]//2, 2))
                Ds_reconstrs_per_obj = np.transpose(Ds_reconstrs_per_obj, (0, 3, 1, 2, 4))

                Ds_diff = np.empty_like(self.Ds)
                Ds_diff_train = Ds_diff[:self.n_train]
                Ds_diff_validation = Ds_diff[self.n_train:self.n_train+self.n_validation]
                for i in np.arange(len(self.Ds)):
                    if i % Ds_reconstrs_per_obj.shape[1] == 0:
                        print("Data shapes", self.Ds[i:i+Ds_reconstrs_per_obj.shape[1]].shape, Ds_reconstrs_per_obj[self.obj_ids[i]].shape)
                        
                        Ds_reconstr_i = Ds_reconstrs_per_obj[self.obj_ids[i]]
                        med = np.median(Ds_reconstr_i, axis=(1, 2), keepdims=True)
                        Ds_reconstr_i /= med
                        
                        Ds_diff[i:i+Ds_reconstrs_per_obj.shape[1]] = self.Ds[i:i+Ds_reconstrs_per_obj.shape[1]] - Ds_reconstr_i#, i % Ds_reconstr.shape[1]]

                    ###########################################################
                    # DEBUG -- REMOVE
                    #if i % 160 == 0:
                    #    my_test_plot = plot.plot(nrows=1, ncols=2)
                    #    my_test_plot.colormap(self.Ds[i, :, :, 0], [0], show_colorbar=True)
                    #    my_test_plot.colormap(reconstr[i], [1])
                    #    my_test_plot.save(f"{dir_name}/reconstr_a{i}.png")
                    #    my_test_plot.close()
                    ###########################################################
                                        
            self.predict_mode2(self.Ds, self.diversities, DD_DP_PP, self.obj_ids, tt_sums, alphas, Ds_diff)
            print("Index, num epochs, epoch_mode_2:", self.mode_2_index, self.n_epochs_2, self.epoch_mode_2)
            for epoch_mode_2 in np.arange(self.epoch_mode_2, self.n_epochs_mode_2):
                validation_losses = []
                input_data_train = [self.Ds_train, self.diversities_train, DD_DP_PP_train, tt_sums_train, alphas_train]
                input_data_validation = [self.Ds_validation, self.diversities_validation, DD_DP_PP_validation, tt_sums_validation, alphas_validation]
                if nn_mode == MODE_3:
                    input_data_train.append(Ds_diff_train)
                    input_data_validation.append(Ds_diff_validation)
                start_epoch = 0
                end_epoch = 10#1
                #if epoch_mode_2 == self.mode_2_index:
                #    start_epoch = self.epoch
                #    end_epoch = self.n_epochs_2
                    
                for epoch in np.arange(start_epoch, end_epoch):
                    history = model.fit(x=input_data_train, y=output_data_train,
                                epochs=self.n_epochs_1,
                                batch_size=batch_size,
                                shuffle=shuffle_epoch,
                                validation_data=[input_data_validation, output_data_validation],
                                #callbacks=[keras.callbacks.TensorBoard(log_dir='model_log')],
                                verbose=1,
                                steps_per_epoch=None,
                                callbacks=[MyCustomCallback(model)])
                    if self.val_loss > history.history['val_loss'][-1]:
                        self.val_loss = history.history['val_loss'][-1]
                        save_weights(model, (self.n_epochs_1, self.n_epochs_2, self.n_epochs_mode_2, epoch, epoch_mode_2, self.val_loss, self.mode_2_index))
                    else:
                        print("Validation loss increased", self.val_loss, history.history['val_loss'][-1])
                        self.val_loss = float("inf")
                        load_weights(model)
                        break
                    #validation_losses.append(history.history['val_loss'])
                    #if len(validation_losses) >= 10:
                    #    print("Average validation loss: " + str(np.mean(validation_losses[-10:])))
                
                    #    if len(validation_losses) >= 20:
                    #        if np.mean(validation_losses[-10:]) > np.mean(validation_losses[-20:-10]):
                    #            break
                    #        validation_losses = validation_losses[-20:]
                    
                self.predict_mode2(self.Ds, self.diversities, DD_DP_PP, self.obj_ids, tt_sums, alphas, Ds_diff)
                ###########################################################
                # DEBUG -- REMOVE
                #for i in np.arange(len(self.Ds)):
                #    if i % 160 == 0:
                #        my_test_plot = plot.plot(nrows=1, ncols=2)
                #        my_test_plot.colormap(self.Ds[i, :, :, 0], [0], show_colorbar=True)
                #        my_test_plot.colormap(reconstr[i], [1])
                #        my_test_plot.save(f"{dir_name}/reconstr{i}.png")
                #           my_test_plot.close()
                ###########################################################
                    
                self.epoch = 0
                print("epoch_mode_2, n_epochs_mode_2", epoch_mode_2, self.n_epochs_mode_2)
            #print("Exiting")
            #sys.exit()
            self.epoch_mode_2 = 0
            self.mode_2_index += 1
            #if self.mode_2_index >= self.n_epochs_mode_2:
            #    self.mode_2_index = 0
            #if self.n_epochs_2 > 1: 
            #    self.n_epochs_2 //= 2

        
        #######################################################################
        # Plot some of the training data results
        n_test = min(num_objs, 5)

        alphas_layer_model = Model(inputs=model.input, outputs=model.get_layer("alphas_layer").output)
        if self.nn_mode == MODE_1:
            pred_alphas = alphas_layer_model.predict([self.Ds, self.diversities], batch_size=batch_size)
        elif self.nn_mode >= MODE_2:
            #DD_DP_PP = np.zeros((len(self.Ds), 4, nx, nx))
            #tt_sums = np.zeros((len(self.Ds), 2))
            input_data = [self.Ds, self.diversities, DD_DP_PP, tt_sums, alphas]
            if nn_mode == MODE_3:
                #reconstr = np.empty((len(self.Ds), nx, nx))
                #reconstrs = dict()
                #for obj_id in np.unique(self.obj_ids):
                #    Ds_ = np.asarray(Ds_per_obj[obj_id])
                #    reconstrs[obj_id] = self.deconvolve(Ds_, np.zeros((len(Ds_), jmax)), diversity_per_obj[obj_id])
                #for i in np.arange(len(self.Ds)):
                #    reconstr[i] = reconstrs[obj_ids[i]]
                #for epoch in np.arange(n_epochs_mode_2):
                #    self.predict_mode2(self.Ds, self.diversities, DD_DP_PP, self.obj_ids, reconstr)
                input_data.append(Ds_diff)
            pred_alphas = alphas_layer_model.predict(input_data, batch_size=batch_size)
            
        pred_Ds = None
        #pred_Ds = model.predict([self.Ds, self.objs], batch_size=1)
        #predicted_coefs = model.predict(Ds_train[0:n_test])
    
        #self.Ds_reconstr = np.array(self.Ds_train.shape[0], 1, self.Ds_train.shape[2], self.Ds_train.shape[3])
    
        #self.Ds_reconstr = np.array(self.Ds_train.shape[0], 1, self.Ds_train.shape[2], self.Ds_train.shape[3])
        obj_ids_used = []
        objs_reconstr = []
        i = 0
        while len(obj_ids_used) < n_test and i < len(self.objs):
            
            obj = self.objs[i]#np.reshape(self.objs[i], (self.nx, self.nx))
            found = False

            ###################################################################            
            # Just to plot results only for different objects
            for obj_id in obj_ids_used:
                if obj_id == self.obj_ids[i]:
                    found = True
                    break
            if found:
                i += 1
                continue
            ###################################################################            
            obj_ids_used.append(self.obj_ids[i])

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
                obj_reconstr = self.psf_check.deconvolve(DF, alphas=np.reshape(pred_alphas[i], (num_frames_input, jmax)), gamma=gamma, do_fft = True, fft_shift_before = False, ret_all=False, a_est=None, normalize = False, fltr=self.filter)
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
                my_test_plot.colormap(obj, [row, 0], show_colorbar=True)
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

            my_test_plot.save(f"{dir_name}/train{i}.png")
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
    
    def test(self, Ds_, objs, diversity, positions, coords, file_prefix, true_coefs=None):
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
        #std = np.std(Ds, axis=(1, 2), keepdims=True)
        Ds -= med
        Ds = self.hanning.multiply(Ds, axis=1)
        Ds += med
        ##Ds /= std
        Ds /= med
        
        #Ds = np.transpose(np.reshape(Ds_, (num_frames*num_objects, Ds_.shape[2], Ds_.shape[3], Ds_.shape[4])), (0, 2, 3, 1))
        #objs = objs[:num_objects]
        #objs = np.reshape(np.repeat(objs, num_frames, axis=0), (num_frames*objs.shape[0], objs.shape[1], objs.shape[2]))
        
        #objs = np.tile(objs, (num_frames, 1, 1))
        #objs = np.reshape(objs, (len(objs), -1))

        # Shuffle the data
        #random_indices = random.choice(len(Ds), size=len(Ds), replace=False)
        #Ds = Ds[random_indices]
        #objs = objs[random_indices]
        #diversities = diversities[random_indices]f

        #Ds = np.zeros((num_objects, 2*num_frames, Ds_.shape[3], Ds_.shape[4]))
        #for i in np.arange(num_objects):
        #    for j in np.arange(num_frames):
        #        Ds[i, 2*j] = Ds_[j, i, 0]
        #        Ds[i, 2*j+1] = Ds_[j, i, 1]

        start = time.time()    
        #pred_alphas = intermediate_layer_model.predict([Ds, np.zeros_like(objs), np.tile(Ds, [1, 1, 1, 16])], batch_size=1)
        alphas_layer_model = Model(inputs=model.input, outputs=model.get_layer("alphas_layer").output)

        if self.nn_mode == MODE_1:
            pred_alphas = alphas_layer_model.predict([Ds, diversities], batch_size=batch_size)
        elif self.nn_mode >= MODE_2:
            DD_DP_PP = np.zeros((len(Ds), 4, nx, nx))
            tt_sums = np.zeros((len(Ds), 2))
            alphas = np.zeros((len(Ds), num_alphas_input*num_frames_input*jmax))
            input_data = [Ds, diversities, DD_DP_PP, tt_sums, alphas]
            Ds_diff = None
            if nn_mode == MODE_3:
                #group_per_obj(self, Ds, alphas, diversities, obj_ids, DD_DP_PP=None)
                Ds_per_obj, alphas_per_obj, diversities_per_obj, DD_DP_PP_sums_per_obj, _ = self.group_per_obj(Ds, np.zeros((len(Ds), jmax)), diversities, obj_ids, DD_DP_PP)
                #reconstr = np.empty((len(Ds), nx, nx))
                #reconstrs = dict()
                #for obj_id in np.unique(obj_ids):
                #    Ds_ = np.asarray(Ds_per_obj[obj_id])
                #    reconstrs[obj_id] = self.deconvolve(Ds_, np.zeros((len(Ds_), jmax)), diversity_per_obj[obj_id])
                
                Ds_reconstrs_per_obj = self.Ds_reconstr(Ds_per_obj, alphas_per_obj, diversities_per_obj).numpy()
                Ds_reconstrs_per_obj = np.reshape(Ds_reconstrs_per_obj, (Ds_reconstrs_per_obj.shape[0], Ds_reconstrs_per_obj.shape[1], Ds_reconstrs_per_obj.shape[2], Ds_reconstrs_per_obj.shape[3]//2, 2))
                Ds_reconstrs_per_obj = np.transpose(Ds_reconstrs_per_obj, (0, 3, 1, 2, 4))

                Ds_diff = np.empty_like(Ds)
                for i in np.arange(len(Ds)):
                    if i % Ds_reconstrs_per_obj.shape[1] == 0:
                        Ds_reconstr_i = Ds_reconstrs_per_obj[obj_ids[i]]
                        med1 = np.median(Ds_reconstr_i, axis=(1, 2), keepdims=True)
                        Ds_reconstr_i /= med1

                        Ds_diff[i:i+Ds_reconstrs_per_obj.shape[1]] = Ds[i:i+Ds_reconstrs_per_obj.shape[1]] - Ds_reconstr_i#, i % Ds_reconstr.shape[1]]
                              
                        if i == 0:
                            ###########################################################
                            # DEBUG -- REMOVE
                            my_test_plot = plot.plot(nrows=2, ncols=2)
                            my_test_plot.colormap(Ds[i, :, :, 0], [0, 0], show_colorbar=True)
                            my_test_plot.colormap(Ds[i, :, :, 1], [0, 1])
                            my_test_plot.colormap(Ds_reconstr_i[i, :, :, 0], [1, 0])
                            my_test_plot.colormap(Ds_reconstr_i[i, :, :, 1], [1, 1])
                            my_test_plot.save(f"{dir_name}/reconstr_a{i}.png")
                            my_test_plot.close()
                            ###########################################################
                        
                #for i in np.arange(len(Ds)):
                #    reconstr[i] = reconstrs[obj_ids[i]]
                input_data.append(Ds_diff)
            for epoch in np.arange(n_epochs_mode_2):
                self.predict_mode2(Ds, diversities, DD_DP_PP, obj_ids, tt_sums, alphas, Ds_diff)
            #for epoch in np.arange(n_epochs_mode_2):
            #    print("DD_DP_PP", DD_DP_PP[0, 0, 0, 0], DD_DP_PP[0, 1, 0, 0], DD_DP_PP[0, 2, 0, 0], DD_DP_PP[0, 3, 0, 0])
            #    self.predict_mode2(Ds, diversities, DD_DP_PP, obj_ids)
            pred_alphas = alphas_layer_model.predict(input_data, batch_size=batch_size)
            
        #Ds *= std
        Ds *= med
        #Ds += med
        
        end = time.time()
        print("Prediction time: " + str(end - start))

        #obj_reconstr_mean = np.zeros((self.nx-1, self.nx-1))
        #DFs = np.zeros((len(objs), 2, 2*self.nx-1, 2*self.nx-1), dtype='complex') # in Fourier space
        
        obj_ids_test = []
        
        cropped_Ds = []
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
                cropped_Ds.append(Ds[i, :, :, 0][top_left_delta[0]:bottom_right_delta[0], top_left_delta[1]:bottom_right_delta[1]])
                full_shape += cropped_obj.shape
                print("cropped_obj.shape", cropped_obj.shape, top_left_coord)

            # Find all other realizations of the same object
            DFs = []
            Ds_ = []
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
                            Ds_.append(Ds[j, :, :, 2*l:2*l+2])
                            DFs.append(np.array([DF, DF_d]))
                            alphas.append(pred_alphas[j, l*jmax:(l+1)*jmax])
                            print("alphas", j, l, alphas[-1][0])
                            #if n_test_frames is not None and len(alphas) >= n_test_frames:
                            #    break
                    #if len(alphas) >= n_test_frames:
                    #    break
            Ds_ = np.asarray(Ds_)
            DFs = np.asarray(DFs, dtype="complex")
            alphas = np.asarray(alphas)
            
            if smooth_window > 0:
                alphas1 = alphas[:-smooth_window]
                for smooth_i in np.arange(1, smooth_window):
                    alphas1 += alphas[smooth_i:-(smooth_window-smooth_i)]
                alphas1 += alphas[smooth_window:]
                alphas = alphas1/(smooth_window+1)
                Ds_ = Ds_[smooth_window//2:-smooth_window//2]
                
            print("alphas", alphas.shape, Ds_.shape)
            
            #obj_reconstr = psf_check.deconvolve(np.array([[DF, DF_d]]), alphas=np.array([pred_alphas[i]]), gamma=gamma, do_fft = True, fft_shift_before = False, ret_all=False, a_est=None, normalize = False)
            #obj_reconstr = fft.ifftshift(obj_reconstr[0])
            #obj_reconstr_mean += obj_reconstr

            if len(alphas) > 0:
                diversity = np.concatenate((diversities[i, :, :, 0], diversities[i, :, :, 1]))
                #diversity = np.concatenate((diversities[i, :, :, 0][nx//4:nx*3//4,nx//4:nx*3//4], diversities[i, :, :, 1][nx//4:nx*3//4,nx//4:nx*3//4]))
                self.psf_check.coh_trans_func.set_diversity(diversity)
                obj_reconstr = self.deconvolve(Ds_[None,], alphas, diversity).numpy()[0]
                #obj_reconstr = self.psf_check.deconvolve(DFs, alphas=alphas, gamma=gamma, do_fft = True, fft_shift_before = False, 
                #                                         ret_all=False, a_est=None, normalize = False, fltr=self.filter)
                #obj_reconstr = fft.ifftshift(obj_reconstr)

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
            my_test_plot = plot.plot(nrows=n_rows, ncols=2, width=8, height=6)
            row = 0
            if obj_reconstr is not None:
                my_test_plot.colormap(obj, [row, 0], show_colorbar=True)
                my_test_plot.colormap(obj_reconstr, [row, 1])
                row += 1
            my_test_plot.colormap(Ds[i, :, :, 0], [row, 0])
            my_test_plot.colormap(Ds[i, :, :, 1], [row, 1])
            my_test_plot.save(f"{dir_name}/{file_prefix}{i}.png")
            my_test_plot.close()

            if true_coefs is not None:
                true_alphas = true_coefs[obj_ids[i]]
                if smooth_window > 0:
                    true_alphas = true_alphas[smooth_window//2:-smooth_window//2]
                nrows = int(np.sqrt(jmax))
                ncols = int(math.ceil(jmax/nrows))
                my_test_plot = plot.plot(nrows=nrows, ncols=ncols, smart_axis=False)
                row = 0
                col = 0
                #xs = np.arange(modes_nn.shape[0]*modes_nn.shape[1])
                nf = min(alphas.shape[0], true_alphas.shape[0])
                xs = np.arange(nf)
                for coef_index in np.arange(alphas.shape[1]):
                    scale = np.std(alphas[:, coef_index])/np.std(true_alphas[:, coef_index])
                    mean = np.mean(alphas[:, coef_index])
                    my_test_plot.plot(xs, np.reshape(alphas[:nf, coef_index]-mean, -1), [row, col], "r-")
                    my_test_plot.plot(xs, np.reshape(true_alphas[:nf, coef_index]*scale, -1), [row, col], "b--")
                    col += 1
                    if col >= ncols:
                        row += 1
                        col = 0
                my_test_plot.save(f"{dir_name}/alphas{i}.png")
                my_test_plot.close()

        if estimate_full_image:
            max_pos = np.max(positions, axis = 0)
            min_coord = np.min(cropped_coords, axis = 0)
            full_shape[0] = full_shape[0] // (max_pos[1] + 1)
            full_shape[1] = full_shape[1] // (max_pos[0] + 1)
            print("full_shape", full_shape)
            full_obj = np.zeros(full_shape)
            full_reconstr = np.zeros(full_shape)
            full_D = np.zeros(full_shape)
            for i in np.arange(len(cropped_objs)):
                x = cropped_coords[i][0]-min_coord[0]
                y = cropped_coords[i][1]-min_coord[1]
                s = cropped_objs[i].shape
                print(x, y, s)
                full_obj[x:x+s[0],y:y+s[1]] = cropped_objs[i]
                full_reconstr[x:x+s[0],y:y+s[1]] = cropped_reconstrs[i]
                full_D[x:x+s[0],y:y+s[1]] = cropped_Ds[i]
            my_test_plot = plot.plot(nrows=1, ncols=3, size=plot.default_size(len(full_obj), len(full_obj)))
            my_test_plot.colormap(full_obj, [0], show_colorbar=True)
            my_test_plot.colormap(full_reconstr, [1])
            my_test_plot.colormap(full_D, [2])
            
            my_test_plot.set_axis_title([0], "MOMFBD")
            my_test_plot.set_axis_title([1], "Neural network")
            my_test_plot.set_axis_title([2], "Raw frame")
            my_test_plot.save(f"{dir_name}/{file_prefix}.png")
            my_test_plot.close()
            
            max_obj = np.max(full_obj)
            min_obj = np.min(full_obj)
            c1_obj = (max_obj-min_obj)/(max_obj+min_obj)

            max_reconstr = np.max(full_reconstr)
            min_reconstr = np.min(full_reconstr)
            c1_reconstr = (max_reconstr-min_reconstr)/(max_reconstr+min_reconstr)

            print("Contrasts 1", c1_obj, c1_reconstr)
            print("Contrasts 2", np.std(full_obj), np.std(full_reconstr))
            

if train:

    Ds, objs, pupil, modes, diversity, true_coefs, positions, coords = load_data(data_file)

    nx = Ds.shape[3]
    jmax = len(modes)
    jmax_to_use = 4

    if shuffle0:
        random_indices = random.choice(Ds.shape[1], size=Ds.shape[1], replace=False)
        Ds = Ds[:, random_indices]

    #hanning = utils.hanning(nx, 10)
    #med = np.median(Ds, axis=(3, 4), keepdims=True)
    #std = np.std(Ds, axis=(3, 4), keepdims=True)
    #Ds -= med
    #Ds = hanning.multiply(Ds, axis=3)
    #Ds += med
    ##Ds /= std
    #Ds /= med
    
    n_train = int(len(Ds)*.8)
    print("n_train, n_test", n_train, len(Ds) - n_train)
    print("num_frames", Ds.shape[1])
        
    Ds_train = Ds[:n_train]
    num_frames_valid = num_frames_input*batch_size
    Ds_test = Ds[n_train:, :num_frames_valid]
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

    my_test_plot.save(f"{dir_name}/null_deconv.png")
    my_test_plot.close()

    ###########################################################################
    n_test_frames = Ds_test.shape[1] # Necessary to be set (TODO: refactor)
    model = nn_model(jmax, nx, num_frames, num_objs, pupil, modes)

    sys.stdout.flush()

    for rep in np.arange(0, num_reps):
        model.set_data(Ds_train, objs_train, diversity, positions_train)
        print("Rep no: " + str(rep))
    
        #model.psf.set_jmax_used(jmax_to_use)
        model.train()
        
        if jmax_to_use <= jmax//2:
            jmax_to_use *= 2
        else:
            jmax_to_use = jmax
            

        #if rep % 5 == 0:
        #model.test(Ds_test, objs_test, diversity, positions_test, coords_test, "validation")
        
        #if np.mean(model.validation_losses[-10:]) > np.mean(model.validation_losses[-20:-10]):
        #    break
        #model.validation_losses = model.validation_losses[-20:]
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

    
    Ds, objs, pupil, modes, diversity, true_coefs, positions, coords = load_data(data_file)

    nx = Ds.shape[3]
    jmax = len(modes)

    '''
    pa_check = psf.phase_aberration([])#len(modes), start_index=1)
    pa_check.set_terms(np.array([]))#np.zeros((jmax, nx//2, nx//2)))#modes)
    ctf_check = psf.coh_trans_func()
    ctf_check.set_phase_aberr(pa_check)
    ctf_check.set_pupil(pupil)
    #ctf_check.set_diversity(diversity[i, j])
    psf_check = psf.psf(ctf_check, corr_or_fft=False)
    for i in np.arange(100, 110):
        ###############################################################
        # DBG
        obj_tmp = psf_check.critical_sampling(objs[i])
        my_test_plot = plot.plot(nrows=1, ncols=3)
        my_test_plot.colormap(objs[i], [0])
        my_test_plot.colormap(obj_tmp, [1])
        my_test_plot.colormap(objs[i]-obj_tmp, [2])
        my_test_plot.save(f"{dir_name}/critical_sampling{i}.png")
        my_test_plot.close()
        ###############################################################
    print("Critical sampling tetśt")
    '''

    if n_test_objects is None:
        n_test_objects = Ds.shape[0]
    if n_test_frames is None:
        n_test_frames = Ds.shape[1]
    n_test_objects = min(Ds.shape[0], n_test_objects)
    n_test_frames = min(Ds.shape[1], n_test_frames)
    
    print("n_test_objects, n_test_frames", n_test_objects, n_test_frames)
    #if nn_mode == MODE_2 and n_epochs_mode_2 > 0:
    #    assert(n_test_frames == num_frames_mode_2)
    
    max_pos = np.max(positions, axis = 0)

    print(max_pos)
    max_pos = np.round(max_pos*np.sqrt(n_test_objects/len(Ds))).astype(int)
    print(max_pos)
    filtr = np.all(positions < max_pos, axis=1)

    if no_shuffle:
        stride = 1
    else:
        stride = Ds.shape[1] // n_test_frames
    #n_test_frames //= stride
    Ds = Ds[filtr, :stride*n_test_frames:stride]
    objs = objs[filtr]
    positions = positions[filtr]
    coords = coords[filtr]
    true_coefs = true_coefs[filtr, :stride*n_test_frames:stride]
    
    n_test_frames -= smooth_window
    
    #hanning = utils.hanning(nx, 10)
    #med = np.median(Ds, axis=(3, 4), keepdims=True)
    #std = np.std(Ds, axis=(3, 4), keepdims=True)
    #Ds -= med
    #Ds = hanning.multiply(Ds, axis=3)
    #Ds += med
    ##Ds /= std
    #Ds /= med
    
    print("true_coefs", true_coefs.shape)
    #print(positions)
    #print(coords)

    #mean = np.mean(Ds, axis=(3, 4), keepdims=True)
    #std = np.std(Ds, axis=(3, 4), keepdims=True)
    #Ds -= mean
    #Ds /= np.median(Ds, axis=(3, 4), keepdims=True)

    model = nn_model(jmax, nx, n_test_frames, num_objs, pupil, modes)
    
    model.test(Ds, objs, diversity, positions, coords, "test", true_coefs=true_coefs)

#logfile.close()
