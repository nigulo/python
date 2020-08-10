import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
#import numpy.random as random
sys.setrecursionlimit(10000)
import keras
#import tensorflow.keras as keras
keras.backend.set_image_data_format('channels_last')
#from keras import backend as K
import tensorflow as tf
#import tensorflow.signal as tf_signal
#from tensorflow.keras.models import Model

#tf.compat.v1.disable_eager_execution()
#from multiprocessing import Process

#import math

import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

#import zarr
import pickle
import os.path

sys.path.append('../utils')
import plot


def load_weights(model, dir_name):
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

def save_weights(model, params, dir_name):
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
    
    
    def __init__(self, nx, ny, nz, num_input_channels, dir_name, batch_size, n_gpus, gpu_id, activation_fn, n_channels, n_epochs_1=None, n_epochs_2=None):
        
        self.nx = nx
        self.ny = ny
        self.ny = ny
        self.num_input_channels = num_input_channels
        self.dir_name = dir_name
        self.batch_size = batch_size

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
        
        optimizer = keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
            
        self.model.compile(optimizer=optimizer, loss='mse')#'adadelta', loss='mse')

        params = load_weights(model, self.dir_name)
        
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
                        batch_size=self.batch_size,
                        shuffle=shuffle_epoch,
                        validation_data=[[data_test], loglik_test],
                        #callbacks=[keras.callbacks.TensorBoard(log_dir='model_log')],
                        verbose=1,
                        steps_per_epoch=None,
                        callbacks=[MyCustomCallback(model)])
            if True:#self.val_loss > history.history['val_loss'][-1]:
                self.val_loss = history.history['val_loss'][-1]
                save_weights(model, (self.n_epochs_1, self.n_epochs_2, epoch, self.val_loss), self.dir_name)
            else:
                print("Validation loss increased", self.val_loss, history.history['val_loss'][-1])
                #self.val_loss = float("inf")
                load_weights(model, self.dir_name)
                break
        self.epoch = 0
        

    def test(self, data_test, loglik_test):
        model = self.model
        

        pred_logliks = model.predict([data_test], batch_size=self.batch_size)[:,0]
        
        print(pred_logliks.shape, loglik_test.shape)
        x = np.arange(len(pred_logliks))
        my_plot = plot.plot(nrows=1, ncols=3, smart_axis=False)
        my_plot.plot(x=x, y=pred_logliks, ax_index=0, params="k-")
        my_plot.plot(x=x, y=loglik_test, ax_index=1, params="k-")
        my_plot.plot(x=x, y=np.abs(pred_logliks-loglik_test), ax_index=2, params="k-")
        #my_plot.plot(x=x, y=loglik_test, ax_index=None, params="b--")
        my_plot.save("nn_results.png")
        my_plot.close()
        
        
