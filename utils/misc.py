import numpy as np
import scipy.ndimage
import os
import pickle

def normalize(ds, axis = None):
    min_ds = np.min(ds, axis=axis, keepdims=True)
    max_ds = np.max(ds, axis=axis, keepdims=True)
    assert(np.all(max_ds != min_ds))
    ds_norm = (ds - min_ds)/(max_ds - min_ds)
    return ds_norm

def center(ds, axis = None):
    mean_ds = np.mean(ds, axis=axis)
    return ds - mean_ds

def sample_image(image, factor):
    if len(image.shape) > 2:
        image_out = []
        for i in np.arange(len(image)):
            image_out.append(sample_image(image[i], factor))
        return np.asarray(image_out)
    if image.dtype == 'complex':
        real = scipy.ndimage.zoom(image.real, factor, output=None, order=3, mode='constant', cval=0.0, prefilter=True)
        imag = scipy.ndimage.zoom(image.imag, factor, output=None, order=3, mode='constant', cval=0.0, prefilter=True)
        return real + imag*1.j
    else:
        return scipy.ndimage.zoom(image, factor, output=None, order=3, mode='constant', cval=0.0, prefilter=True)


def load(filename):
    if filename is not None:
        data_file = filename
        if os.path.isfile(data_file):
            return pickle.load(open(data_file, 'rb'))
    return None

def save(filename, state):
    with open(filename, 'wb') as f:
        pickle.dump(state, f, protocol=4)

def trunc(ds, perc):
    p1 = np.percentile(ds, perc*100)
    p2 = np.percentile(ds, (1.0-perc)*100)
    #print(perc*100, (1.0-perc)*100, p1, p2)
    ds_out = np.array(ds)
    if perc < 0.5:
        p1a = p1
        p1 = p2
        p2 = p1a
    for i in np.arange(0, np.shape(ds_out)[0]):
        for j in np.arange(np.shape(ds_out)[1]):
            if ds_out[i, j] > p1:
                ds_out[i, j] = p1
            elif ds_out[i, j] < p2:
                ds_out[i, j] = p2
    return ds_out
