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

   
i = 1
dir_name = "."
#if len(sys.argv) > i:
#    dir_name = sys.argv[i]
#i +=1


train = True
if len(sys.argv) > i:
    if sys.argv[i].upper() == "TEST":
        train = False
i +=1

if train:
    data_file = "data_nn"
else:
    data_file = "data_nn_test"

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

activation_fn = "relu"

   
train_perc = .8 
num_reps = 1000

n_epochs_2 = 20
n_epochs_1 = 1

# How many objects to use in training
num_objs = 80#0#None


batch_size = 32
n_channels = 8
    
if dir_name is None:
    dir_name = "results" + time.strftime("%Y%m%d-%H%M%S")
    os.mkdir(dir_name)
    

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

n_gpus = 0#len(gpus)

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
            raise Exception("No data found")
    data_train = loaded['data_train']
    loglik_train = loaded['loglik_train']
    data_test = loaded['data_test']
    loglik_test = loaded['loglik_test']
    return data_train, loglik_train, data_test, loglik_test


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
            return None
    params = pickle.load(open(dir_name + '/params.dat', 'rb'))
    return params

def save_weights(model, params):
    model.save_weights(dir_name + '/weights.tf')
    with open(dir_name + '/params.dat', 'wb') as f:
        pickle.dump(params, f, protocol=4)



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


class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self): 
        super(ScaleLayer, self).__init__() 
        self.scale = tf.Variable(1.)
        
    def call(self, inputs):
        return inputs * self.scale

class nn_model:       
    
    
    def __init__(self, nx, ny, nz, num_input_channels):
        
        self.nx = nx
        self.ny = ny
        self.ny = ny
        self.num_input_channels = num_input_channels

        batch_size_per_gpu = max(1, batch_size//max(1, n_gpus))

        #self.strategy = tf.distribute.MirroredStrategy()
        #with self.strategy.scope():
        with tf.device(gpu_id):

            field_input = keras.layers.Input((nx, ny, nz, num_input_channels), name='field_input')
    
            print("Creating model")
            
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
            

            def conv_layer(x, n_channels, kernel=(3, 3, 2), max_pooling=(2, 2, 1), batch_normalization=True, num_convs=3, activation=activation_fn):
                for i in np.arange(num_convs):
                    x1 = keras.layers.Conv3D(n_channels, (1, 1, 1), activation='linear', padding='same')(x)#(normalized)
                    x2 = keras.layers.Conv3D(n_channels, kernel, activation=activation, padding='same')(x)#(normalized)
                    x = keras.layers.add([x2, x1])#tf.keras.backend.tile(image_input, [1, 1, 1, 16])])
                    if batch_normalization:
                        x = keras.layers.BatchNormalization()(x)
                if max_pooling is not None:
                    x = keras.layers.MaxPool3D(pool_size=max_pooling)(x)
                return x
            
            def seq_block(x):
                x_out = tf.slice(x, [0, 0], [1, 1024])
                x_i = x_out
                for i in np.arange(1, batch_size_per_gpu):
                    x_i1 = tf.concat([tf.slice(x, [i, 0], [1, 1024]), x_i], axis=1)
                    x_i = keras.layers.Dense(1024, activation=activation_fn)(x_i1)
                    x_out = tf.concat([x_out, x_i], axis=0)
                return x_out
                        
            hidden_layer = conv_layer(field_input, kernel=(3, 3, 2), n_channels=n_channels)
            hidden_layer = conv_layer(hidden_layer, kernel=(3, 3, 2), n_channels=2*n_channels)

            hidden_layer = keras.layers.Flatten()(hidden_layer)
            hidden_layer = keras.layers.Dense(36*n_channels, activation=activation_fn)(hidden_layer)

            hidden_layer = keras.layers.Dense(128, activation=activation_fn)(hidden_layer)
            hidden_layer = keras.layers.Dense(32, activation=activation_fn)(hidden_layer)
            loglik_layer = keras.layers.Dense(1, activation="linear", name="loglik_layer")(hidden_layer)
            
            model = keras.models.Model(inputs=[field_input], outputs=loglik_layer)

            
        self.model = model
        
            
        self.model.compile(optimizer='adadelta', loss='mse')

        params = load_weights(model)
        
        epoch = 0
        val_loss = float("inf")
        
        if params is not None:
            n_epochs_1_, n_epochs2_, epoch, val_loss = params


        # Overwrite
        n_epochs_1_ = n_epochs_1
        n_epochs_2_ = n_epochs_2


        self.n_epochs_1 = n_epochs_1_
        self.n_epochs_2 = n_epochs_2_
        self.epoch = epoch
        self.val_loss = val_loss


 
    def train(self, data_train, loglik_train, data_test, loglik_test):
        model = self.model


        shuffle_epoch = True

        for epoch in np.arange(self.epoch, self.n_epochs_2):
            history = model.fit(x=[data_train], y=loglik_train,
                        epochs=self.n_epochs_1,
                        batch_size=batch_size,
                        shuffle=shuffle_epoch,
                        validation_data=[[data_test], loglik_test],
                        #callbacks=[keras.callbacks.TensorBoard(log_dir='model_log')],
                        verbose=1,
                        steps_per_epoch=None,
                        callbacks=[MyCustomCallback(model)])
            if True:#self.val_loss > history.history['val_loss'][-1]:
                self.val_loss = history.history['val_loss'][-1]
                save_weights(model, (self.n_epochs_1, self.n_epochs_2, epoch, self.val_loss))
            else:
                print("Validation loss increased", self.val_loss, history.history['val_loss'][-1])
                #self.val_loss = float("inf")
                load_weights(model)
                break
        self.epoch = 0
        

    def test(self, data_test, loglik_test):
        model = self.model
        

        pred_logliks = model.predict([data_test], batch_size=batch_size)
        
        x = np.arange(len(pred_logliks))
        my_plot = utils.plot()
        my_plot.plot(self, x, pred_logliks, ax_index=None, params="r-")
        my_plot.plot(self, x, loglik_test, ax_index=None, params="b--")
        my_plot.save("nn_results.png")
        my_plot.close()
        
        


if train:

    data_train, loglik_train, data_test, loglik_test = load_data(data_file)
    
    print("data", data_train.shape)

    # Data shape (N, 3, nx, ny, nz)
    # Data[:, 0]: bx
    # Data[:, 1]: by
    # Data[:, 2]: bz
    
    # Transpose 2nd index to last index (we use channels last)
    data_train = np.transpose(data_train, (0, 2, 3, 4, 1))
    
    nx = data_train.shape[1]
    ny = data_train.shape[2]
    nz = data_train.shape[3]
    num_input_channels = data_train.shape[4]

    model = nn_model(nx, ny, nz, num_input_channels)

    for rep in np.arange(0, num_reps):
        model.set_data()
        print("Rep no: " + str(rep))
    
        model.train(data_train, loglik_train, data_test, loglik_test)
        
else:

    
    _, _, data_test, loglik_test = load_data(data_file)
    
    nx = data_test.shape[1]
    ny = data_test.shape[2]
    nz = data_test.shape[3]
    num_input_channels = data_test.shape[4]



    model = nn_model(nx, ny, nz, num_input_channels)
    
    model.test(data_test, loglik_test)
