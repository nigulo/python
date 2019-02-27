import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import numpy.random as random
import keras

import psf
import utils

import matplotlib.pyplot as plt
import matplotlib.colorbar as cb
from matplotlib import cm
import pickle
import os.path

import time

nx = 320
ny = 320

zoomin_start = nx/2 - 32
zoomin_end = ny/2 + 32

otf_or_psf = True

psf_vals_nx = 64
psf_vals_ny = 64


aperture_func = lambda u: utils.aperture_circ(u, 0.2, 100.0)

n_coefs = 50
n_data = 200
max_huge_set_size = 1000
if os.path.isfile("config.txt"):
    with open("config.txt", "r") as f:
        lines = f.readlines()
        n_coefs1 = int(lines[0])
        n_data1 = int(lines[1])
        max_huge_set_size1 = int(lines[2])
        otf_or_psf1 = False
        if len(lines) > 3:
            otf_or_psf1 = bool(lines[3])
        assert(n_coefs == n_coefs1 and max_huge_set_size == max_huge_set_size1 and otf_or_psf1 == otf_or_psf)
        print("Using old configuraton")
    read_data = True
else:
    with open('config.txt', 'w') as f:
        f.write(str(n_coefs) + "\n")
        f.write(str(n_data) + "\n")
        f.write(str(max_huge_set_size) + "\n")
        f.write(str(otf_or_psf) + "\n")
    read_data = False

assert(n_data <= max_huge_set_size and (max_huge_set_size % n_data) == 0)
n_train = int(n_data*0.75)

num_sets = 10000n_coefs
n_epochs = 100

def reverse_colourmap(cmap, name = 'my_cmap_r'):
     return mpl.colors.LinearSegmentedColormap(name, cm.revcmap(cmap._segmentdata))

my_cmap = reverse_colourmap(plt.get_cmap('binary'))#plt.get_cmap('winter')

def create_model():
    
    print("Creating model")
    coefs = keras.layers.Input((n_coefs,), name='coefs')

    if otf_or_psf:
        #psf = keras.layers.Input(IMAGE_SHAPE, name='psf')
        
        start_nx = 8#(((psf_vals_nx + 1)/2 + 1)/2 + 1)/2 + 2
        start_ny = 8#(((psf_vals_ny + 1)/2 + 1)/2 + 1)/2 + 2
        
        hidden = keras.layers.Dense(start_nx*start_ny, activation='relu')(coefs)
        hidden = keras.layers.Dense(start_nx*start_ny, activation='relu')(coefs)
        hidden = keras.layers.Dense(start_nx*4*start_ny*4, activation='relu')(coefs)
        hidden = keras.layers.Dense(start_nx*4*start_ny*4, activation='relu')(hidden)
        hidden = keras.layers.Dense(start_nx*4*start_ny*4, activation='relu')(hidden)
        hidden = keras.layers.Dense(2*start_nx*4*start_ny*4, activation='relu')(hidden)
        hidden = keras.layers.Dense(2*start_nx*4*start_ny*4, activation='relu')(hidden)
        hidden = keras.layers.Dense(2*start_nx*4*start_ny*4, activation='relu')(hidden)
        hidden = keras.layers.Dense(2*start_nx*8*start_ny*8, activation='relu')(hidden)
        hidden = keras.layers.Dense(2*start_nx*8*start_ny*8, activation='relu')(hidden)
        #hidden = keras.layers.Dense(start_nx*start_ny*16, activation='relu')(hidden)
        #hidden = keras.layers.core.Flatten()(hidden)
    
        output = keras.layers.Reshape((2, start_nx*8, start_ny*8, 1))(hidden)
    
    
    else:
    
        start_nx = 8#(((psf_vals_nx + 1)/2 + 1)/2 + 1)/2 + 2
        start_ny = 8#(((psf_vals_ny + 1)/2 + 1)/2 + 1)/2 + 2
        
        hidden = keras.layers.Dense(start_nx*4*start_ny*4, activation='relu')(coefs)
        hidden = keras.layers.Dense(start_nx*4*start_ny*4, activation='relu')(hidden)
        hidden = keras.layers.Dense(start_nx*4*start_ny*4, activation='relu')(hidden)
        hidden = keras.layers.Dense(start_nx*4*start_ny*4, activation='relu')(hidden)
        hidden = keras.layers.Dense(start_nx*4*start_ny*4, activation='relu')(hidden)
        hidden = keras.layers.Dense(start_nx*4*start_ny*4, activation='relu')(hidden)
        hidden = keras.layers.Dense(start_nx*16*start_ny*16, activation=None)(hidden)
        hidden = keras.layers.Dense(start_nx*8*start_ny*8, activation='relu')(hidden)
        #hidden = keras.layers.Dense(start_nx*start_ny*16, activation='relu')(hidden)
        #hidden = keras.layers.core.Flatten()(hidden)
    
        hidden = keras.layers.Reshape((start_nx*8, start_ny*8, 1))(hidden)
    
        #hidden = keras.layers.Conv2D(20, (3, 3), activation='relu', padding='same')(hidden)
        #hidden = keras.layers.UpSampling2D((2, 2))(hidden)
        #hidden = keras.layers.Conv2D(20, (3, 3), activation='relu', padding='same')(hidden)
    
        #hidden = keras.layers.Conv2D(20, (3, 3), subsample=(2, 2), activation='relu', padding='same')(hidden)
    
        output = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(hidden)
    

    model = keras.models.Model(input=coefs, output=output)
    optimizer = keras.optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False)
    #optimizer = keras.optimizers.RMSprop(lr=0.0025, rho=0.95, epsilon=0.01)
    #model.compile(optimizer, loss='mean_absolute_error')
    #model.compile(optimizer='adadelta', loss='binary_crossentropy')
    #model.compile(optimizer=optimizer, loss='binary_crossentropy')
    model.compile(optimizer='adadelta', loss='mean_absolute_error')
    return model


validation_losses = []

def plot_data(psf_vals, predicted_psfs, n_test, fig_name):
    
    extent=[0., 1., 0., 1.]
    plot_aspect=(extent[1]-extent[0])/(extent[3]-extent[2])#*2/3 

    if otf_or_psf:
        fig, axes = plt.subplots(nrows=n_test, ncols=4)
        fig.set_size_inches(10, 2.5*n_test)
        psf_vals = np.reshape(psf_vals[:n_test], (n_test, 2, psf_vals_nx, psf_vals_ny))
        predicted_psfs = np.reshape(predicted_psfs, (n_test, 2, psf_vals_nx, psf_vals_ny))
    else:
        fig, axes = plt.subplots(nrows=n_test, ncols=2)
        fig.set_size_inches(20, 2.5*n_test)
        psf_vals = np.reshape(psf_vals[:n_test], (n_test, psf_vals_nx, psf_vals_ny))
        predicted_psfs = np.reshape(predicted_psfs, (n_test, psf_vals_nx, psf_vals_ny))
    
    for i in np.arange(0, n_test):
    
        if otf_or_psf:
            ax = axes[i][0]
            ax.imshow(psf_vals[i,0].T,extent=extent,cmap=my_cmap,origin='lower')
            ax.set_aspect(aspect=plot_aspect)
    
            ax = axes[i][1]
            ax.imshow(predicted_psfs[i,0].T,extent=extent,cmap=my_cmap,origin='lower')
            ax.set_aspect(aspect=plot_aspect)

            ax = axes[i][2]
            ax.imshow(psf_vals[i,1].T, extent=extent,cmap=my_cmap,origin='lower')
            ax.set_aspect(aspect=plot_aspect)
    
            ax = axes[i][3]
            ax.imshow(predicted_psfs[i,1].T,extent=extent,cmap=my_cmap,origin='lower')
            ax.set_aspect(aspect=plot_aspect)
        else:
            ax = axes[i][0]
            ax.imshow(psf_vals[i].T,extent=extent,cmap=my_cmap,origin='lower')
            ax.set_aspect(aspect=plot_aspect)
    
            ax = axes[i][1]
            ax.imshow(predicted_psfs[i].T,extent=extent,cmap=my_cmap,origin='lower')
            ax.set_aspect(aspect=plot_aspect)

    fig.savefig(fig_name)
    plt.close(fig)

def train(model, coefs_train, coefs_test, psf_train, psf_test):
    #print("Training")
    if np.shape(coefs_test)[0] > 0:
        history = model.fit(coefs_train, psf_train,
                    epochs=n_epochs,
                    batch_size=10,
                    shuffle=True,
                    validation_data=(coefs_test, psf_test),
                    #callbacks=[keras.callbacks.TensorBoard(log_dir='model_log')],
                    verbose=1)
        
        validation_losses.append(history.history['val_loss'])
        print("Average validation loss: " + str(np.mean(validation_losses[-10:])))

    else:
        history = model.fit(coefs_train, psf_train,
                    epochs=n_epochs,
                    batch_size=10,
                    shuffle=True,
                    #callbacks=[keras.callbacks.TensorBoard(log_dir='model_log')],
                    verbose=1)

    ############################
    # Plot training data results
    n_test = 5
    predicted_psfs = model.predict(coefs_train[0:n_test])
    
    
    plot_data(psf_train, predicted_psfs, n_test, 'psf_train.png')
    
    return model

def test(model):
    
        n_test_data = 5
        
        #print("Generating test data")
        coefs, psf_vals = gen_data(n_test_data)
        #coefs = np.random.normal(size=(n_test_data, n_coefs))
        #psf_vals = np.zeros((n_test_data, nx, ny))
        #for i in np.arange(0, n_test_data):
        #    pa = psf.phase_aberration(zip(np.arange(4, n_coefs + 4), coefs[i]))
        #    ctf = psf.coh_trans_func(lambda u: 1.0, pa, lambda u: 0.0)
        #    psf_vals[i] = psf.psf(ctf, ValueError: cannot reshape array of size 81920 into shape (10,64,64,1)
    
        start = time.time()    
        predicted_psfs = model.predict(coefs)
        end = time.time()
        #print("Prediction time" + (end - start))

        
        plot_data(psf_vals, predicted_psfs, n_test_data, 'psf_test.png')
        

def gen_data(n_data, normalize=True, log=False):
    #coefs = np.zeros((n_data, n_coefs))
    coefs = np.random.normal(size=(n_data, n_coefs))*10
    if otf_or_psf:
        psf_vals = np.zeros((n_data, 2, psf_vals_nx, psf_vals_ny))
    else:
        psf_vals = np.zeros((n_data, psf_vals_nx, psf_vals_ny))
        
    start = time.time()
    for i in np.arange(0, n_data):
        #i_max = np.argmax(np.abs(coefs[i]))
        #pa = psf.phase_aberration([(4 + i_max, coefs[i_max])])
        pa = psf.phase_aberration(zip(np.arange(4, n_coefs + 4), coefs[i]))
        ctf = psf.coh_trans_func(aperture_func, pa, lambda u: 0.0)
        
        if otf_or_psf:
            psf_vals[i] = psf.psf(ctf, nx, ny).get_otf_vals()[:,zoomin_start:zoomin_end,zoomin_start:zoomin_end]
        else:
            psf_vals[i] = psf.psf(ctf, nx, ny).get_incoh_vals()[zoomin_start:zoomin_end,zoomin_start:zoomin_end]
            
        #psf_vals[i] = utils.trunc(psf_vals[i], 1e-2)
        
        if normalize:
            if otf_or_psf:
                min_val = np.reshape(np.min(psf_vals[i], axis=(1,2)), (2,1,1))
                max_val = np.reshape(np.max(psf_vals[i], axis=(1,2)), (2,1,1))
            else:    
                min_val = np.min(psf_vals[i])
                max_val = np.max(psf_vals[i])
            
            psf_vals[i,:] = (psf_vals[i,:] - min_val)/(max_val - min_val)
        if log:
            psf_vals[i] += 1.0
            psf_vals[i] = np.log(psf_vals[i])
            if otf_or_psf:
                min_val = np.reshape(np.min(psf_vals[i], axis=(1,2)), (2,1,1))
                max_val = np.reshape(np.max(psf_vals[i], axis=(1,2)), (2,1,1))
            else:    
                min_val = np.min(psf_vals[i])
                max_val = np.max(psf_vals[i])
            psf_vals[i] = (psf_vals[i] - min_val)/(max_val - min_val)
        print(str(float(i)*100/n_data) + "%")
    end = time.time()
    #print("Generation time: " + str(end - start))
    return coefs, psf_vals


def load_data(huge_set_num):
    data_file = 'data' + str(huge_set_num) + '.pkl'
    if load_data and os.path.isfile(data_file):
        return pickle.load(open(data_file, 'rb'))
    else:
        return None

def save_data(data, huge_set_num):
    with open('data' + str(huge_set_num) + '.pkl', 'wb') as f:
        pickle.dump(data, f)



model = create_model()

huge_set_size = min(max_huge_set_size, n_data*num_sets)
num_small_sets = huge_set_size / n_data
for huge_set_num in np.arange(0, n_data*num_sets/huge_set_size):

    data = load_data(huge_set_num)
    if data is None:
        print("Generating training data: " +  str(huge_set_num))
        huge_coefs, huge_psf_vals = gen_data(huge_set_size)
        save_data((huge_coefs, huge_psf_vals), huge_set_num)
    else:
        huge_coefs, huge_psf_vals = data
    
    num_reps = 3
    for rep in np.arange(0, num_reps):
        print("Rep no: " + str(rep))
        for set_num in np.arange(0, num_small_sets):
            coefs = huge_coefs[set_num*n_data:(set_num+1)*n_data]
            psf_vals = huge_psf_vals[set_num*n_data:(set_num+1)*n_data]
            
            if otf_or_psf:
                psf_vals = np.reshape(psf_vals, (len(psf_vals), 2, psf_vals_nx, psf_vals_ny, 1))
            else:
                psf_vals = np.reshape(psf_vals, (len(psf_vals), psf_vals_nx, psf_vals_ny, 1))
            coefs_train = coefs[:n_train] 
            coefs_test = coefs[n_train:]
            psf_train = psf_vals[:n_train]
            psf_test = psf_vals[n_train:]
            
            if rep == num_reps-1:
                # In the laast iteration train on the full set
                train(model, coefs, np.array([]), psf_vals, np.array([]))
            else:
                train(model, coefs_train, coefs_test, psf_train, psf_test)
        
            test(model)
