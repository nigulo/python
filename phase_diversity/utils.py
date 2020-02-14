import numpy as np
import scipy.special as special
#import scipy.misc
import scipy.ndimage
import numpy.fft as fft
import os
import matplotlib.pyplot as plt
from astropy.io import fits
import misc

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

def aperture_circ(xs, r_scale_fact=1., coef=5.0, radius = None):
    assert(r_scale_fact > 0 and r_scale_fact <= 1.)
    scalar = False
    if len(np.shape(xs)) == 1:
        assert(False) # Not supported anymore
        scalar = True
        xs = np.array([xs])
    if radius is not None:
        r = radius
    else:
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


def get_coords(nx, arcsec_per_px, diameter, wavelength):      
    rad2deg=180.0/np.pi
    scale_angle2CCD=arcsec_per_px/(rad2deg*3600.0)
    diff_limit = wavelength*1.e-8/diameter
    q_number=diff_limit/(scale_angle2CCD)
    rc=1./q_number # telescope_d in pupil space
    #print("diff_limit, scale_angle2CCD, rc", diff_limit, scale_angle2CCD, rc)
     
    #x_limit = nx*rc
    x_limit = 1./nx/rc
    
    #coh_vals = np.zeros((nx, ny))
    xs = np.linspace(-x_limit, x_limit, nx)
    #print("PSF x_limit", xs[0], xs[-1])
    return np.dstack(np.meshgrid(xs, xs)[::-1]), rc, x_limit


def normalize_(Ds, Ps):
    Ds = fft.ifftshift(fft.ifft2(Ds), axes=(-2, -1)).real
    norm = fft.ifftshift(fft.ifft2(Ps), axes=(-2, -1)).real
    norm = np.sum(norm, axis=(2, 3)).repeat(Ds.shape[2]*Ds.shape[3]).reshape((Ds.shape[0], Ds.shape[1], Ds.shape[2], Ds.shape[3]))
    Ds *= norm
    return fft.fft2(fft.fftshift(Ds, axes=(-2, -1)))

    #Ds = fft.ifft2(Ds).real
    #norm = fft.ifftshift(fft.ifft2(Ps)).real
    #norm = np.sum(norm, axis=(2, 3)).repeat(Ds.shape[2]*Ds.shape[3]).reshape((Ds.shape[0], Ds.shape[1], Ds.shape[2], Ds.shape[3]))
    #Ds *= norm
    #1return fft.fft2(Ds)

'''
    When calling from psf, set fft_shift_before = True, 
    when from psf_basis, set fft_shift_before = False
'''
def deconvolve_(Ds, Ps, gamma, do_fft = True, fft_shift_before = False, ret_all=False, tip_tilt = None, a_est=None):
    regularizer_eps = 1e-10
    assert(gamma == 1.0) # Because in likelihood we didn't involve gamma
    D = Ds[:, 0, :, :]
    D_d = Ds[:, 1, :, :]
    
    P = Ps[:, 0, :, :]
    P_d = Ps[:, 1, :, :]

    P_conj = P.conjugate()
    P_d_conj = P_d.conjugate()

    F_image = np.sum(D * P_conj + gamma * D_d * P_d_conj + regularizer_eps, axis=0)
    den = np.sum(P*P_conj + gamma * P_d * P_d_conj + regularizer_eps, axis=0)
    F_image /= den

    if fft_shift_before:
        F_image = fft.ifftshift(F_image, axes=(-2, -1))

    '''
    ###########################################################################
    # REMOVE THIS BLOCK
    D1 = D[0]
    D1_d = D_d[0]
    P1 = P[0]
    P1_d = P_d[0]

    P1_conj = P1.conjugate()
    P1_d_conj = P1_d.conjugate()

    F1_image = D1 * P1_conj + gamma * D1_d * P1_d_conj + regularizer_eps
    den1 = P1*P1_conj + gamma * P1_d * P1_d_conj + regularizer_eps
    F1_image /= den1
    
    if fft_shift_before:
        D1 = fft.ifftshift(D1)
        D1_d = fft.ifftshift(D1_d)
        F1_image = fft.ifftshift(F1_image)
        P1 = fft.ifft2(fft.ifftshift(P1)).real
        P1_d = fft.ifft2(fft.ifftshift(P1_d)).real
    else:
        P1 = fft.ifftshift(fft.ifft2(P1)).real
        P1_d = fft.ifftshift(fft.ifft2(P1_d)).real
    D1 = fft.ifft2(D1).real
    D1_d = fft.ifft2(D1_d).real
    import sys
    sys.path.append('../utils')
    import plot
    import datetime
    my_plot = plot.plot(nrows=1, ncols=5)
    my_plot.colormap(D1, [0])
    my_plot.colormap(D1_d, [1])
    my_plot.colormap(fft.ifft2(F1_image).real, [2])
    my_plot.colormap(np.log(P1), [3])
    my_plot.colormap(np.log(P1_d), [4])
    my_plot.save("psf_basis_deconvolve" + str(datetime.datetime.now().timestamp()) + ".png")
    # REMOVE THIS BLOCK
    ###########################################################################
    '''
    
    #np.savetxt("F.txt", F_image, fmt='%f')
    
    if not do_fft and not ret_all:
        return F_image

    if tip_tilt is not None and a_est is not None:
        #Ps = np.ones_like(Ps)
        #image, image_F, Ps = tip_tilt.deconvolve(D, Ps, a_est)
        image, image_F, Ps = tip_tilt.deconvolve(F_image, Ps, a_est)
    else:
        image = fft.ifft2(F_image).real
        if not fft_shift_before:
            image = fft.ifftshift(image, axes=(-2, -1))
        #image = fft.ifft2(F_image).real
        
       
    if ret_all:
        return image, F_image, Ps
    else:
        return image

'''
    is_planet specifies whether image contains a disk shaped object.
    image_size is the number of pixels to sample from the whole image (both in x and y).
        If the image is smaller than image_size then the whole image is returned.
    tile specifies if the sampling is repeated in x and y directions or not. If
        tile is False then only image (0:image_size, 0:image_size) is returned.
    
'''
def read_images(dir="images", image_file=None, is_planet = False, image_size = None, tile=False, scale=1.0):
    assert(not is_planet or (is_planet and not tile))
    images = []
    images_d = []
    nx = 0
    nx_orig = 0
    for root, dirs, files in os.walk(dir):
        for file in files:
            if image_file is not None and file[:len(image_file)] != image_file:
                continue
            if file[-5:] == '.fits':
                hdul = fits.open(dir + "/" + file)
                image = hdul[0].data
                hdul.close()
            else:
                image = plt.imread(dir + "/" + file)
                if len(image.shape) == 3:
                    image = image[:, :, 0]
                    
                #image = plt.imread(dir + "/" + file)
            if scale != 1.:
                image = misc.sample_image(image, scale)
            start_coord = 0
            if image_size is None:
                image_size = min(image.shape[0], image.shape[1])
            else:
                image_size = min(image.shape[0], image.shape[1], image_size)
            while start_coord + image_size <= image.shape[0] and start_coord + image_size <= image.shape[1]:
                if is_planet:# Try to detect center
                    row_mean = np.mean(image, axis = 0)
                    col_mean = np.mean(image, axis = 1)
                    
                    max_row = np.argmax(row_mean)
                    max_col = np.argmax(col_mean)
                    
                    #start_index_max = max(0, min(image.shape[0], image.shape[1]) - nx_orig)
                    start_index_x = max_col - image_size//2#np.random.randint(0, start_index_max)
                    start_index_y = max_row - image_size//2#np.random.randint(0, start_index_max)
                    sub_image = image[start_index_x:start_index_x + image_size,start_index_y:start_index_y + image_size]
                else:
                    sub_image = image[start_coord:start_coord + image_size, start_coord:start_coord+ image_size]
                
                
                nx_orig = np.shape(sub_image)[0]
                #sub_image = upsample(sub_image)
                nx = np.shape(sub_image)[0]
                assert(nx == np.shape(sub_image)[1])
                if len(images) > 0:
                    assert(nx == np.shape(images[-1])[0])
                
                if '_d.' not in file:
                    images.append(sub_image)
                else:
                    images_d.append(sub_image)
                
                if not tile:
                    break
                start_coord += image_size
    assert(len(images_d) == 0 or len(images_d) == len(images))
    print("Num images", len(images), images[0].shape, nx, nx_orig)
    return images, images_d, nx, nx_orig