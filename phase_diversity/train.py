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

import time

nx = 320
ny = 320

zoomin_start = nx/2 - 32
zoomin_end = ny/2 + 32

psf_vals_nx = 64
psf_vals_ny = 64


aperture_func = lambda u: psf.aperture_circ(u, 0.2, 100.0)

n_coefs = 50
n_data = 100
max_huge_set_size = 1000
assert(n_data <= max_huge_set_size and (max_huge_set_size % n_data) == 0)
n_train = int(n_data*0.75)

num_sets = 100

n_epochs = 100

def reverse_colourmap(cmap, name = 'my_cmap_r'):
     return mpl.colors.LinearSegmentedColormap(name, cm.revcmap(cmap._segmentdata))

my_cmap = reverse_colourmap(plt.get_cmap('binary'))#plt.get_cmap('winter')

def create_model():
    
    print("Creating model")
    
    coefs = keras.layers.Input((n_coefs,), name='coefs')
    #psf = keras.layers.Input(IMAGE_SHAPE, name='psf')
    
    start_nx = 8#(((psf_vals_nx + 1)/2 + 1)/2 + 1)/2 + 2
    start_ny = 8#(((psf_vals_ny + 1)/2 + 1)/2 + 1)/2 + 2
    
    hidden = keras.layers.Dense(start_nx*start_ny, activation='relu')(coefs)
    #hidden = keras.layers.Reshape((start_nx, start_ny, 1))(hidden)
    #hidden = keras.layers.Conv2D(20, (3, 3), activation='relu', padding='same')(hidden)
    #hidden = keras.layers.core.Flatten()(hidden)

    hidden = keras.layers.Dense(start_nx*start_ny*2, activation='relu')(hidden)
    hidden = keras.layers.Dense(start_nx*start_ny*3, activation='relu')(hidden)
    #hidden = keras.layers.Reshape((start_nx*2, start_ny*2, 1))(hidden)
    #hidden = keras.layers.Conv2D(20, (3, 3), activation='relu', padding='same')(hidden)
    #hidden = keras.layers.core.Flatten()(hidden)

    hidden = keras.layers.Dense(start_nx*start_ny*4, activation='relu')(hidden)
    hidden = keras.layers.Dense(start_nx*start_ny*6, activation='relu')(hidden)
    #hidden = keras.layers.Reshape((start_nx*4, start_ny*4, 1))(hidden)
    #hidden = keras.layers.Conv2D(20, (3, 3), activation='relu', padding='same')(hidden)
    #hidden = keras.layers.core.Flatten()(hidden)

    hidden = keras.layers.Dense(start_nx*start_ny*8, activation='relu')(hidden)
    hidden = keras.layers.Dense(start_nx*start_ny*12, activation='relu')(hidden)
    hidden = keras.layers.Dense(start_nx*start_ny*16, activation='relu')(hidden)
    #hidden = keras.layers.Dense(start_nx*start_ny*32, activation='relu')(hidden)
    #hidden = keras.layers.Dense(start_nx*start_ny*64, activation='sigmoid')(hidden)
    hidden = keras.layers.Reshape((start_nx*4, start_ny*4, 1))(hidden)
    hidden = keras.layers.Conv2D(20, (3, 3), activation='relu', padding='same')(hidden)
    hidden = keras.layers.UpSampling2D((2, 2))(hidden)
    hidden = keras.layers.Conv2D(20, (3, 3), activation='relu', padding='same')(hidden)
    #hidden = keras.layers.Conv2D(20, (3, 3), activation='relu', padding='same')(hidden)
    #hidden = keras.layers.UpSampling2D((2, 2))(hidden)
    #hidden = keras.layers.Conv2D(20, (3, 3), activation='relu', padding='same')(hidden)
    #hidden = keras.layers.Conv2D(20, (3, 3), activation='relu', padding='same')(hidden)
    #hidden = keras.layers.UpSampling2D((2, 2))(hidden)
    #hidden = keras.layers.Conv2D(20, (3, 3), activation='relu')(hidden)
    #hidden = keras.layers.Conv2D(20, (3, 3), activation='relu')(hidden)
    #hidden = keras.layers.UpSampling2D((2, 2))(hidden)
    #hidden = keras.layers.Conv2D(40, (5, 5), activation='relu')(hidden)
    #hidden = keras.layers.Conv2D(40, (5, 5), activation='relu')(hidden)
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
    predicted_psfs = np.reshape(predicted_psfs, (n_test, psf_vals_nx, psf_vals_ny))
    
    extent=[0., 1., 0., 1.]
    plot_aspect=(extent[1]-extent[0])/(extent[3]-extent[2])#*2/3 

    fig, axes = plt.subplots(nrows=n_test, ncols=2)
    fig.set_size_inches(6, 3*n_test)
    
    for i in np.arange(0, n_test):
    
        ax = axes[i][0]
        #ax.imshow(np.log(psf_vals[i].T),extent=extent,cmap=my_cmap,origin='lower')
        ax.imshow(np.reshape(psf_train[i], (psf_vals_nx, psf_vals_ny)).T,extent=extent,cmap=my_cmap,origin='lower')
        #ax1.set_title(r'Factor graph')
        #ax1.set_ylabel(r'$f$')
        #start, end = ax32.get_xlim()
        #ax1.xaxis.set_ticks(np.arange(5, end, 4.9999999))
        #ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
        #ax1.xaxis.labelpad = -1
        #ax1.set_xlabel(r'$l_{\rm coh}$')#,fontsize=20)
        ax.set_aspect(aspect=plot_aspect)

        ax = axes[i][1]
        #ax.imshow(np.log(predicted_psfs[i].T),extent=extent,cmap=my_cmap,origin='lower')
        ax.imshow(predicted_psfs[i].T,extent=extent,cmap=my_cmap,origin='lower')
        #ax1.set_title(r'Factor graph')
        #ax1.set_ylabel(r'$f$')
        #start, end = ax32.get_xlim()
        #ax1.xaxis.set_ticks(np.arange(5, end, 4.9999999))
        #ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
        #ax1.xaxis.labelpad = -1
        #ax1.set_xlabel(r'$l_{\rm coh}$')#,fontsize=20)
        ax.set_aspect(aspect=plot_aspect)

    fig.savefig('psf_train.png')
    plt.close(fig)
    
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
        #    psf_vals[i] = psf.psf(ctf, nx, ny).get_incoh_vals()
    
        start = time.time()    
        predicted_psfs = model.predict(coefs)
        end = time.time()
        #print("Prediction time" + (end - start))
        predicted_psfs = np.reshape(predicted_psfs, (n_test_data, psf_vals_nx, psf_vals_ny))
        
        extent=[0., 1., 0., 1.]
        plot_aspect=(extent[1]-extent[0])/(extent[3]-extent[2])#*2/3 
        
        fig, axes = plt.subplots(nrows=n_test_data, ncols=2)
        fig.set_size_inches(6, 3*n_test_data)
        
        for i in np.arange(0, n_test_data):
        
            ax = axes[i][0]
            #ax.imshow(np.log(psf_vals[i].T),extent=extent,cmap=my_cmap,origin='lower')
            ax.imshow(psf_vals[i].T,extent=extent,cmap=my_cmap,origin='lower')
            #ax1.set_title(r'Factor graph')
            #ax1.set_ylabel(r'$f$')
            #start, end = ax32.get_xlim()
            #ax1.xaxis.set_ticks(np.arange(5, end, 4.9999999))
            #ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
            #ax1.xaxis.labelpad = -1
            #ax1.set_xlabel(r'$l_{\rm coh}$')#,fontsize=20)
            ax.set_aspect(aspect=plot_aspect)

            ax = axes[i][1]
            #ax.imshow(np.log(predicted_psfs[i].T),extent=extent,cmap=my_cmap,origin='lower')
            ax.imshow(predicted_psfs[i].T,extent=extent,cmap=my_cmap,origin='lower')
            #ax1.set_title(r'Factor graph')
            #ax1.set_ylabel(r'$f$')
            #start, end = ax32.get_xlim()
            #ax1.xaxis.set_ticks(np.arange(5, end, 4.9999999))
            #ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
            #ax1.xaxis.labelpad = -1
            #ax1.set_xlabel(r'$l_{\rm coh}$')#,fontsize=20)
            ax.set_aspect(aspect=plot_aspect)

        fig.savefig('psf_test.png')
        plt.close(fig)

def gen_data(n_data, normalize=True, log=False):
    #coefs = np.zeros((n_data, n_coefs))
    coefs = np.random.normal(size=(n_data, n_coefs))*10
    psf_vals = np.zeros((n_data, psf_vals_nx, psf_vals_ny))
    start = time.time()
    for i in np.arange(0, n_data):
        #i_max = np.argmax(np.abs(coefs[i]))
        #pa = psf.phase_aberration([(4 + i_max, coefs[i_max])])
        pa = psf.phase_aberration(zip(np.arange(4, n_coefs + 4), coefs[i]))
        ctf = psf.coh_trans_func(aperture_func, pa, lambda u: 0.0)
        psf_vals[i] = psf.psf(ctf, nx, ny).get_incoh_vals()[zoomin_start:zoomin_end,zoomin_start:zoomin_end]
        #psf_vals[i] = utils.trunc(psf_vals[i], 1e-2)
        if normalize:
            min_val = np.min(psf_vals[i])
            max_val = np.max(psf_vals[i])
            psf_vals[i] = (psf_vals[i] - min_val)/(max_val - min_val)
        if log:
            psf_vals[i] += 1.0
            psf_vals[i] = np.log(psf_vals[i])
            min_val = np.min(psf_vals[i])
            max_val = np.max(psf_vals[i])
            psf_vals[i] = (psf_vals[i] - min_val)/(max_val - min_val)
        print(str(float(i)*100/n_data) + "%")
    end = time.time()
    #print("Generation time: " + str(end - start))
    return coefs, psf_vals


model = create_model()

huge_set_size = min(max_huge_set_size, n_data*num_sets)
num_small_sets = huge_set_size / n_data
for huge_set_num in np.arange(0, n_data*num_sets/huge_set_size):

    print("Generating training data: " +  str(huge_set_num))
    huge_coefs, huge_psf_vals = gen_data(huge_set_size)
    
    num_reps = 10
    for rep in np.arange(0, num_reps):
        for set_num in np.arange(0, num_small_sets):
            coefs = huge_coefs[set_num*n_data:(set_num+1)*n_data]
            psf_vals = huge_psf_vals[set_num*n_data:(set_num+1)*n_data]
            
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
