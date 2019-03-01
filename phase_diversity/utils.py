import numpy as np
import scipy.special as special
#import scipy.misc
import scipy.ndimage

def cart_to_polar(us):
    scalar = False
    if len(np.shape(us)) == 1:
        scalar = True
        us = np.array([us])
    rhos = np.sqrt(np.sum(us**2, axis=us.ndim-1))
    phis = np.arctan2(us[...,1], us[...,0])
    ret_val = np.stack((rhos, phis), axis = us.ndim-1)
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

def aperture_circ(us, r=1.0, coef=5.0):
    scalar = False
    if len(np.shape(us)) == 1:
        scalar = True
        us = np.array([us])
    if coef > 0.0:
        ret_val = 0.5+0.5*special.erf(coef*(r-np.sqrt(np.sum(us**2, axis=us.ndim-1))))
    else:
        ret_val = np.zeros(np.shape(us)[0])
        indices = np.where(np.sum(us**2, axis=us.ndim-1) <= r*r)[0]
        ret_val[indices] = 1.0
    if scalar:
        ret_val = ret_val[0]
    return ret_val
    
def resize(image):
    #return scipy.misc.imresize(image, (image.shape[0]*2-1, image.shape[1]*2-1))
    zoom_perc = (float(image.shape[0])*2.-1.)/image.shape[0]
    return scipy.ndimage.zoom(image, zoom_perc, output=None, order=3, mode='constant', cval=0.0, prefilter=True)

