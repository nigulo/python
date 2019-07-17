import numpy as np
import scipy.ndimage

def normalize(ds, axis = None):
    min_ds = np.min(ds, axis=axis)
    max_ds = np.max(ds, axis=axis)
    assert(max_ds != min_ds)
    ds_norm = (ds - min_ds)/(max_ds - min_ds)
    return ds_norm

def center(ds, axis = None):
    mean_ds = np.mean(ds, axis=axis)
    return ds - mean_ds

def sample_image(image, factor):
    if image.dtype == 'complex':
        real = scipy.ndimage.zoom(image.real, factor, output=None, order=3, mode='constant', cval=0.0, prefilter=True)
        imag = scipy.ndimage.zoom(image.imag, factor, output=None, order=3, mode='constant', cval=0.0, prefilter=True)
        return real + imag*1.j
    else:
        return scipy.ndimage.zoom(image, factor, output=None, order=3, mode='constant', cval=0.0, prefilter=True)

