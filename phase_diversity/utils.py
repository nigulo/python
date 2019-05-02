import numpy as np
import scipy.special as special
#import scipy.misc
import scipy.ndimage

def cart_to_polar(xs):
    scalar = False
    if len(np.shape(xs)) == 1:
        scalar = True
        xs = np.array([xs])
    rhos = np.sqrt(np.sum(xs**2, axis=xs.ndim-1))
    phis = np.arctan2(xs[...,1], xs[...,0])
    ret_val = np.stack((rhos, phis), axis = xs.ndim-1)
    if scalar:
        ret_val = ret_val[0]
    return ret_val

def polar_to_cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

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

def aperture_circ(xs, r_scale_fact=1., coef=5.0):
    assert(r_scale_fact > 0 and r_scale_fact <= 1.)
    scalar = False
    if len(np.shape(xs)) == 1:
        assert(False) # Not supported anymore
        scalar = True
        xs = np.array([xs])
    r = np.max(xs)*r_scale_fact

    if coef > 0.0:
        ret_val = 0.5+0.5*special.erf(coef*(r-np.sqrt(np.sum(xs**2, axis=xs.ndim-1))))
    else:
        ret_val = np.zeros(np.shape(xs)[0])
        indices = np.where(np.sum(xs**2, axis=xs.ndim-1) <= r*r)[0]
        ret_val[indices] = 1.0
    if scalar:
        ret_val = ret_val[0]
    return ret_val
    
def upsample(image):
    #return scipy.misc.imresize(image, (image.shape[0]*2-1, image.shape[1]*2-1))
    zoom_perc = (float(image.shape[0])*2.-1.)/image.shape[0]
    return scipy.ndimage.zoom(image, zoom_perc, output=None, order=3, mode='constant', cval=0.0, prefilter=True)

def downsample(image):
    #return scipy.misc.imresize(image, (image.shape[0]*2-1, image.shape[1]*2-1))
    zoom_perc = (float(image.shape[0])+1.)/2./image.shape[0]
    if image.dtype == 'complex':
        real = scipy.ndimage.zoom(image.real, zoom_perc, output=None, order=3, mode='constant', cval=0.0, prefilter=True)
        imag = scipy.ndimage.zoom(image.imag, zoom_perc, output=None, order=3, mode='constant', cval=0.0, prefilter=True)
        return real + imag*1.j
    else:
        return scipy.ndimage.zoom(image, zoom_perc, output=None, order=3, mode='constant', cval=0.0, prefilter=True)
