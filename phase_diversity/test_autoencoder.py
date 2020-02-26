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

# How many objects to use in training
num_objs = 10#None

n_epochs = 10
num_iters = 10
num_reps = 1000
shuffle = True

MODE_1 = 1 # aberrated images --> object
MODE_2 = 2 # aberrated images --> wavefront coefs + object --> aberrated images
MODE_3 = 3 # aberrated images  --> wavefront coefs + object --> reconstructed object - object
nn_mode = MODE_1

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
    dir_name = "test_autoencoder" + time.strftime("%Y%m%d-%H%M%S")
    os.mkdir(dir_name)
    

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
import utils


def load_data():
    data_file = dir_name + '/learn_object_Ds.dat.npz'
    if os.path.exists(data_file):
        loaded = np.load(data_file)
        #Ds = loaded['a']
        objs = loaded['b']
        return objs

    return None

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



class nn_model:

    
    def __init__(self, nx):
        image_input = keras.layers.Input((nx, nx, 1), name='image_input') # Channels first

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

            image_input_tiled = keras.layers.Lambda(lambda x : tile(x, 128))(image_input)
            #image_input = tf.keras.backend.tile(image_input, [1, 1, 1, 32])#tf.keras.backend.tile(image_input, [1, 1, 1, 16])])
            #image_input_tiled = keras.layers.Input((nx, nx, 32), name='image_input_tiled')
            #object_input  = keras.layers.Reshape((nx*nx))(object_input)
            ###################################################################
            # Autoencoder
            ###################################################################
            hidden_layer0 = keras.layers.Conv2D(128, (7, 7), activation='relu', padding='same')(image_input)#(normalized)
            #hidden_layer0 = keras.layers.Conv2D(32, (64, 64), activation='relu', padding='same')(hidden_layer0)#(normalized)
            #hidden_layer0 = keras.layers.BatchNormalization()(hidden_layer0)

            hidden_layer0a = keras.layers.add([hidden_layer0, image_input_tiled], name='h0')#tf.keras.backend.tile(image_input, [1, 1, 1, 16])])
            #hidden_layer0 = keras.layers.concatenate([hidden_layer0, image_input], name='h0')
            hidden_layer1a = keras.layers.MaxPooling2D()(hidden_layer0a)
            #hidden_layer1 = tf.keras.backend.tile(hidden_layer1, [1, 1, 1, 2])
            hidden_layer1b = keras.layers.Lambda(lambda x : tile(x, 2))(hidden_layer1a)
            
            hidden_layer2 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(hidden_layer1a)#(normalized)
            #hidden_layer2 = keras.layers.Conv2D(32, (32, 32), activation='relu', padding='same')(hidden_layer2)#(normalized)
            #hidden_layer2 = keras.layers.BatchNormalization()(hidden_layer2)
            hidden_layer2a = keras.layers.add([hidden_layer2, hidden_layer1b], name='h2')
            #hidden_layer2 = keras.layers.concatenate([hidden_layer2, hidden_layer1], name='h2')
            
            hidden_layer3a = keras.layers.MaxPooling2D()(hidden_layer2a)
            #hidden_layer3 = tf.keras.backend.tile(hidden_layer3, [1, 1, 1, 2])
            hidden_layer3b = keras.layers.Lambda(lambda x : tile(x, 2))(hidden_layer3a)
            hidden_layer4 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(hidden_layer3a)#(normalized)
            #hidden_layer4 = keras.layers.Conv2D(32, (16, 16), activation='relu', padding='same')(hidden_layer4)#(normalized)
            #hidden_layer4 = keras.layers.BatchNormalization()(hidden_layer4)
            hidden_layer4a = keras.layers.add([hidden_layer4, hidden_layer3b], name='h4')
            #hidden_layer4 = keras.layers.concatenate([hidden_layer4, hidden_layer3], name='h4')
            
            hidden_layer5a = keras.layers.MaxPooling2D()(hidden_layer4a)
            #hidden_layer5b = keras.layers.Lambda(lambda x : tile(x, 2))(hidden_layer5a)
            hidden_layer6 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(hidden_layer5a)#(normalized)
            #hidden_layer6 = keras.layers.Conv2D(32, (8, 8), activation='relu', padding='same')(hidden_layer6)#(normalized)
            #hidden_layer6 = keras.layers.BatchNormalization()(hidden_layer6)
            hidden_layer6a = keras.layers.add([hidden_layer6, hidden_layer5a], name='h6')
            
            hidden_layer7a = keras.layers.MaxPooling2D()(hidden_layer6a)
            hidden_layer7 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(hidden_layer7a)#(normalized)
            #hidden_layer7 = keras.layers.BatchNormalization()(hidden_layer7)
            #hidden_layer7 = keras.layers.add([hidden_layer7, hidden_layer7a], name='h8')
            hidden_layer8 = keras.layers.MaxPooling2D()(hidden_layer7)


            #hidden_layer7 = keras.layers.Lambda(lambda x : tile(x, 2))(hidden_layer7)
            #hidden_layer8 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(hidden_layer7)#(normalized)
            ##hidden_layer6 = keras.layers.Conv2D(32, (8, 8), activation='relu', padding='same')(hidden_layer6)#(normalized)
            ##hidden_layer6 = keras.layers.BatchNormalization()(hidden_layer6)
            #hidden_layer8 = keras.layers.add([hidden_layer8, hidden_layer7], name='h8')
            
            
            #hidden_layer8 = keras.layers.MaxPooling2D()(hidden_layer7)

            hidden_layer = keras.layers.Flatten()(hidden_layer8)
            hidden_layer = keras.layers.Dense(4608, activation='relu')(hidden_layer)
            #hidden_layer = keras.layers.Dense(1000, activation='relu')(hidden_layer)
            #hidden_layer = keras.layers.Dense(1000, activation='relu')(hidden_layer)
            
            obj_layer = keras.layers.Flatten()(hidden_layer)
            #obj_layer = keras.layers.Dense(1024, activation='relu')(obj_layer)
            #obj_layer = keras.layers.Dense(512, activation='relu')(obj_layer)
            #obj_layer = keras.layers.Dense(1024, activation='relu')(obj_layer)
            #obj_layer = keras.layers.Dense(2304, activation='relu')(obj_layer)
            obj_layer = keras.layers.Dense(4608, activation='relu')(obj_layer)
            obj_layer1 = keras.layers.Reshape((3, 3, 512))(obj_layer)
            obj_layer = keras.layers.Conv2D(512, (1, 1), padding='same', activation='relu')(obj_layer1)#(normalized)
            obj_layer = keras.layers.add([obj_layer, obj_layer1])

            obj_layer1 = keras.layers.UpSampling2D((2, 2))(obj_layer)
            #obj_layer1 = keras.layers.Reshape((6, 6, 256))(obj_layer)
            obj_layer = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(obj_layer1)#(normalized)
            obj_layer = keras.layers.add([obj_layer, obj_layer1])

            obj_layer1 = keras.layers.UpSampling2D((2, 2))(obj_layer)
            obj_layer1 = keras.layers.add([obj_layer1, hidden_layer6a])
            obj_layer = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(obj_layer1)#(normalized)
            #obj_layer = keras.layers.add([obj_layer, obj_layer1])
            #obj_layer = keras.layers.add([obj_layer, tf.slice(obj_layer1, [0, 0, 0, 0], [1, nx//8, nx//8, 256])])
            obj_layer = keras.layers.add([obj_layer, obj_layer1])
            
            obj_layer = keras.layers.UpSampling2D((2, 2))(obj_layer)
            obj_layer1 = keras.layers.Lambda(lambda x : resize(x))(obj_layer)
            obj_layer1 = keras.layers.add([obj_layer1, hidden_layer4a])
            obj_layer = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(obj_layer1)#(normalized)
            #obj_layer = keras.layers.add([obj_layer, untile(obj_layer1, 32)])
            obj_layer = keras.layers.add([obj_layer, tf.slice(obj_layer1, [0, 0, 0, 0], [1, nx//4, nx//4, 256])])
            
            obj_layer1 = keras.layers.UpSampling2D((2, 2))(obj_layer)
            obj_layer1 = keras.layers.add([obj_layer1, hidden_layer2a])
            obj_layer = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(obj_layer1)#(normalized)
            #obj_layer = keras.layers.add([obj_layer, untile(obj_layer1, 16)])
            #obj_layer = keras.layers.add([obj_layer, tf.slice(obj_layer1, [0, 0, 0, 0], [1, nx//2, nx//2, 256])])
            obj_layer = keras.layers.add([obj_layer, obj_layer1])
            
            obj_layer1 = keras.layers.UpSampling2D((2, 2))(obj_layer)
            #image_input_tiled2 = keras.layers.Lambda(lambda x : tile(x, 16))(image_input)
            #obj_layer1 = keras.layers.add([obj_layer1, image_input_tiled2])
            #obj_layer1 = keras.layers.add([obj_layer1, tf.slice(image_input, [0, 0, 0, 0], [1, nx, nx, 1])])
            #obj_layer1 = keras.layers.add([obj_layer1, tf.reshape(object_input, [1, nx, nx, 1])])
            obj_layer = keras.layers.Conv2D(128, (7, 7), padding='same', activation='relu')(obj_layer1)#(normalized)
            #obj_layer1 = keras.layers.add([obj_layer, hidden_layer0a])
            obj_layer = keras.layers.Conv2D(1, (7, 7), padding='same', activation='relu')(obj_layer)#(normalized)
            #obj_layer = keras.layers.add([obj_layer, tf.math.reduce_sum(obj_layer1, axis=3, keepdims=True)])
            #obj_layer = keras.layers.add([obj_layer, tf.slice(obj_layer1, [0, 0, 0, 0], [1, nx, nx, 1])])
            obj_layer = keras.layers.BatchNormalization()(obj_layer)
            output = keras.layers.Reshape((nx, nx), name='obj_layer')(obj_layer)
            
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
        self.model.compile(optimizer='adadelta', loss='mse')
        self.nx = nx
        self.validation_losses = []
        self.nn_mode = nn_mode_


    def set_data(self, objs, train_perc=.75, validation_perc=.2):
        self.objs = objs
                    
        # Shuffle the data
        random_indices = random.choice(len(self.objs), size=len(self.objs), replace=False)
        self.objs = self.objs[random_indices]
        
        n_train = int(math.ceil(len(self.objs)*train_perc))
        n_validation = int(math.ceil(len(self.objs)*validation_perc))

        self.objs_train = self.objs[:n_train] 
        self.objs_validation = self.objs[n_train:n_train+n_validation]
        self.objs_test = self.objs[n_train+n_validation:]
        
        self.objs_train_input = np.reshape(self.objs_train, (self.objs_train.shape[0], self.objs_train.shape[1], self.objs_train.shape[2], 1))
        self.objs_validation_input = np.reshape(self.objs_validation, (self.objs_validation.shape[0], self.objs_validation.shape[1], self.objs_validation.shape[2], 1))
        self.objs_test_input = np.reshape(self.objs_test, (self.objs_test.shape[0], self.objs_test.shape[1], self.objs_test.shape[2], 1))
        
       

    def train(self):
        model = self.model

        for epoch in np.arange(n_epochs):
            history = model.fit(self.objs_train_input, self.objs_train,
                        epochs=1,
                        batch_size=1,
                        shuffle=True,
                        validation_data=(self.objs_validation_input, self.objs_validation),
                        #callbacks=[keras.callbacks.TensorBoard(log_dir='model_log')],
                        verbose=1)
            
            save_weights(model)
        self.validation_losses.append(history.history['val_loss'])
        print("Average validation loss: " + str(np.mean(self.validation_losses[-10:])))
    

    def test(self):
        model = self.model

        #######################################################################
        # Plot some of the training data results
        pred_objs = model.predict(self.objs_test_input, batch_size=1)
    
        for i in np.arange(len(self.objs_test)):
            

            my_test_plot = plot.plot(nrows=1, ncols=3)
            #my_test_plot.colormap(np.reshape(self.objs[i], (self.nx+1, self.nx+1)), [0])
            #my_test_plot.colormap(np.reshape(pred_objs[i], (self.nx+1, self.nx+1)), [1])
            my_test_plot.colormap(self.objs_test[i], [0, 0], show_colorbar=True, colorbar_prec=2)
            my_test_plot.colormap(pred_objs[i], [0, 1])
            my_test_plot.colormap(np.abs(self.objs_test[i] - pred_objs[i]), [0, 2])

            my_test_plot.save(dir_name + "/train_results" + str(i) + ".png")
            my_test_plot.close()

 
        #######################################################################
                    


objs = load_data()


nx = objs.shape[1]

my_test_plot = plot.plot()
my_test_plot.colormap(objs[0])
my_test_plot.save(dir_name + "/obj.png")
my_test_plot.close()

model = nn_model(nx)

model.set_data(objs)

if train:
    for rep in np.arange(0, num_reps):
        print("Rep no: " + str(rep))
    
        model.train()
        model.test()

        model.set_data(objs)
            
    
        #if np.mean(model.validation_losses[-10:]) > np.mean(model.validation_losses[-20:-10]):
        #    break
        model.validation_losses = model.validation_losses[-20:]
else:
    model.test()

#logfile.close()