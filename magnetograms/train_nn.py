import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import numpy.random as random
sys.setrecursionlimit(10000)
import keras
#import tensorflow.keras as keras
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

#import zarr
import pickle
import os.path
import numpy.fft as fft
import matplotlib.pyplot as plt

import time
import nn_model
#import scipy.signal as signal
import tables

   
i = 1
dir_name = "."
if len(sys.argv) > i:
    dir_name = sys.argv[i]
i +=1


train = True
if len(sys.argv) > i:
    if sys.argv[i].upper() == "TEST":
        train = False
i +=1

if train:
    data_file = "data_nn"
else:
    data_file = "data_nn"

if len(sys.argv) > i:
    data_file = sys.argv[i]
i +=1

activation_fn = "relu"

   
#train_perc = .8 
num_reps = 1000

n_epochs_2 = 20
n_epochs_1 = 1

# How many objects to use in training
#num_objs = 80#0#None


batch_size = 32
n_channels = 8
    
if dir_name is None:
    dir_name = "results" + time.strftime("%Y%m%d-%H%M%S")
    os.mkdir(dir_name)
    

sys.path.append('../utils')
sys.path.append('..')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    gpu_id = '/device:GPU:0'
else:
    gpu_id = 'CPU'

if train:
    sys.stdout = open(dir_name + '/log.txt', 'a')
else:
    if gpus:
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
    f = tables.open_file(dir_name + "/" + data_file + ".f5", mode='r')
    data_train = []
    loglik_train = []
    data_test = []
    loglik_test = []
    suffix = 1
    while True:
        try:
            n = f.get_node(f.root, f"data_train{suffix}")
            data_train.extend(n[:])
            n = f.get_node(f.root, f"data_test{suffix}")
            data_test.extend(n[:])
            n = f.get_node(f.root, f"loglik_train{suffix}")
            loglik_train.extend(n[:, 0])
            n = f.get_node(f.root, f"loglik_test{suffix}")
            loglik_test.extend(n[:, 0])
            suffix += 1
        except:
            pass
    print("Num nodes loaded", suffix - 1)

    return np.asarray(data_train), np.asarray(loglik_train), np.asarray(data_test), np.asarray(loglik_test)



if train:

    data_train, loglik_train, data_test, loglik_test = load_data(data_file)#data_test, loglik_test = load_data(data_file)

    # Transpose 2nd index to last index (we use channels last)
    data_train = np.transpose(data_train, (0, 2, 3, 4, 1))
    data_test = np.transpose(data_test, (0, 2, 3, 4, 1))
    
    mean = np.mean(data_train, axis = 0)
    data_train -= mean
    std = np.std(data_train, axis = 0)
    data_train /= std

    #num_train = int(len(data_train)*.8)
    #data_test = data_train[num_train:]
    #data_train = data_train[:num_train]

    #loglik_test = loglik_train[num_train:]
    #loglik_train = loglik_train[:num_train]

    data_test -= mean#np.mean(data_test, axis = 0)
    data_test /= std#np.std(data_test, axis = 0)
    
    
    print("data", data_train.shape)
    loglik_train /= 1e7
    loglik_test /= 1e7
    print(loglik_test.shape)

    # Data shape (N, 3, nx, ny, nz)
    # Data[:, 0]: bx
    # Data[:, 1]: by
    # Data[:, 2]: bz
    
    
    nx = data_train.shape[1]
    ny = data_train.shape[2]
    nz = data_train.shape[3]
    num_input_channels = data_train.shape[4]

    model = nn_model.nn_model(data_file, dir_name, n_gpus, gpu_id)
    if not model.load():
        model.init(nx, ny, nz, num_input_channels, batch_size, activation_fn, n_channels, n_epochs_1, n_epochs_2, mean, std)
    model.create()

    for rep in np.arange(0, num_reps):
        print("Rep no: " + str(rep))
    
        model.train(data_train, loglik_train, data_test, loglik_test)
        
else:

    
    data_train, loglik_train, data_test, loglik_test = load_data(data_file)
    data_train = np.transpose(data_train, (0, 2, 3, 4, 1))

    print(data_train.shape)
    mean = np.mean(data_train, axis = 0)
    data_train -= mean
    std = np.std(data_train, axis = 0)
    data_train /= std

    #num_train = int(len(data_train)*.8)
    #data_test = data_train[num_train:]
    #loglik_test = loglik_train[num_train:]

    #data_test -= mean#np.mean(data_test, axis = 0)
    #data_test /= std#np.std(data_test, axis = 0)
    loglik_test /= 1e7

    data_test = np.transpose(data_test, (0, 2, 3, 4, 1))
    
    nx = data_test.shape[1]
    ny = data_test.shape[2]
    nz = data_test.shape[3]
    num_input_channels = data_test.shape[4]


    model = nn_model.nn_model(data_file, dir_name, n_gpus, gpu_id)
    if not model.load():
        model.init(nx, ny, nz, num_input_channels, batch_size, activation_fn, n_channels, n_epochs_1, n_epochs_2, mean, std)
    model.create()
    
    model.test(data_test, loglik_test)
