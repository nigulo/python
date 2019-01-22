import numpy as np
import numpy.random as random
import keras

import psf

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colorbar as cb
from matplotlib import cm

nx = 320
ny = 320

zoomin_start = nx/2 - 33
zoomin_end = ny/2 + 33

psf_vals_nx = 66
psf_vals_ny = 66


aperture_func = lambda u: psf.aperture_circ(u, 0.2, 100.0)

n_coefs = 10
n_data = 3
n_train = int(n_data*0.75)

n_epochs = 500

def reverse_colourmap(cmap, name = 'my_cmap_r'):
     return mpl.colors.LinearSegmentedColormap(name, cm.revcmap(cmap._segmentdata))

def create_model():
    
    coefs = keras.layers.Input((n_coefs,), name='coefs')
    #psf = keras.layers.Input(IMAGE_SHAPE, name='psf')
    
    start_nx = 10#(((psf_vals_nx + 1)/2 + 1)/2 + 1)/2 + 2
    start_ny = 10#(((psf_vals_ny + 1)/2 + 1)/2 + 1)/2 + 2
    
    hidden1 = keras.layers.Dense(start_nx*start_ny, activation='relu')(coefs)
    hidden2 = keras.layers.Reshape((start_nx, start_ny, 1))(hidden1)
    hidden3 = keras.layers.Conv2D(18, (4, 4), activation='relu', padding='same')(hidden2)
    hidden4 = keras.layers.UpSampling2D((2, 2))(hidden3)
    hidden5 = keras.layers.Conv2D(8, (6, 6), activation='relu', padding='same')(hidden4)
    hidden6 = keras.layers.UpSampling2D((2, 2))(hidden5)
    hidden7 = keras.layers.Conv2D(4, (8, 8), activation='relu')(hidden6)
    hidden8 = keras.layers.UpSampling2D((2, 2))(hidden7)
    output = keras.layers.Conv2D(1, (10, 10), activation='sigmoid', padding='same')(hidden8)

    model = keras.models.Model(input=coefs, output=output)
    optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    #model.compile(optimizer, loss='mse')
    #model.compile(optimizer='adadelta', loss='binary_crossentropy')
    model.compile(optimizer='adadelta', loss='mse')
    return model


def train(coefs_train, coefs_test, psf_train, psf_test):
    print("Training")
    model = create_model()
    model.fit(coefs_train, psf_train,
                epochs=n_epochs,
                batch_size=128,
                shuffle=True,
                validation_data=(coefs_test, psf_test),
                callbacks=[keras.callbacks.TensorBoard(log_dir='model_log')])
    return model

def test(model):
    
        n_test_data = 10
        
        print("Generating test data")
        coefs, psf_vals = gen_data(n_test_data)
        #coefs = np.random.normal(size=(n_test_data, n_coefs))
        #psf_vals = np.zeros((n_test_data, nx, ny))
        #for i in np.arange(0, n_test_data):
        #    pa = psf.phase_aberration(zip(np.arange(4, n_coefs + 4), coefs[i]))
        #    ctf = psf.coh_trans_func(lambda u: 1.0, pa, lambda u: 0.0)
        #    psf_vals[i] = psf.psf(ctf, nx, ny).get_incoh_vals()
    
        predicted_psfs = model.predict(coefs)
        predicted_psfs = np.reshape(predicted_psfs, (n_test_data, psf_vals_nx, psf_vals_ny))
        
        extent=[0., 1., 0., 1.]
        plot_aspect=(extent[1]-extent[0])/(extent[3]-extent[2])#*2/3 
        
        #my_cmap = reverse_colourmap(plt.get_cmap('gnuplot'))
        my_cmap = reverse_colourmap(plt.get_cmap('binary'))#plt.get_cmap('winter')
        
        
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

        fig.savefig('psf_nn.png')
        plt.close(fig)

def gen_data(n_data, normalize=True, log=False):
    #coefs = np.zeros((n_data, n_coefs))
    coefs = np.random.normal(size=(n_data, n_coefs))
    psf_vals = np.zeros((n_data, psf_vals_nx, psf_vals_ny))
    for i in np.arange(0, n_data):
        pa = psf.phase_aberration(zip(np.arange(4, n_coefs + 4), coefs[i]))
        ctf = psf.coh_trans_func(aperture_func, pa, lambda u: 0.0)
        psf_vals[i] = psf.psf(ctf, nx, ny).get_incoh_vals()[zoomin_start:zoomin_end,zoomin_start:zoomin_end]
        if log:
            psf_vals[i] = np.log(psf_vals[i])
        if normalize:
            min_val = np.min(psf_vals[i])
            max_val = np.max(psf_vals[i])
            psf_vals[i] = (psf_vals[i] - min_val)/(max_val - min_val)
    return coefs, psf_vals


print("Generating training data")
coefs, psf_vals = gen_data(n_data)
psf_vals = np.reshape(psf_vals, (len(psf_vals), psf_vals_nx, psf_vals_ny, 1))
coefs_train = coefs[:n_train] 
coefs_test = coefs[n_train:]
psf_train = psf_vals[:n_train]
psf_test = psf_vals[n_train:]

model = train(coefs_train, coefs_test, psf_train, psf_test)
test(model)
