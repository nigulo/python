import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import scipy.special as special
import scipy.signal as signal
import numpy.fft as fft
import matplotlib.pyplot as plt
from matplotlib import cm
import utils

import matplotlib.pyplot as plt
import psf_basis

def reverse_colourmap(cmap, name = 'my_cmap_r'):
     return mpl.colors.LinearSegmentedColormap(name, cm.revcmap(cmap._segmentdata))

my_cmap = reverse_colourmap(plt.get_cmap('binary'))#plt.get_cmap('winter')

image = plt.imread('granulation.png')

image = image[0:10,0:10]

nx = np.shape(image)[0]
ny = np.shape(image)[1]

assert(nx == ny)

fimage = fft.fft2(image)
#vals = fft.ifft2(vals)
#vals = fft.ifft2(vals).real
#fimage = np.roll(np.roll(fimage, int(nx/2), axis=0), int(ny/2), axis=1)
    
print(np.shape(image))

jmax = 5
arcsec_per_px = 0.055
diameter = 50.0
wavelength = 5250.0
F_D = 1.0



psf = psf_basis.psf_basis(jmax = jmax, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength, nx = nx, F_D = F_D)

for trial in np.arange(0, 10):
    betas = np.random.normal(size=jmax) + np.random.normal(size=jmax)*1.j
    psf.set_betas(betas)
    
    psf.create_basis(do_fft=True, do_defocus=True)
    
    fmeasurement = np.zeros((nx, ny), dtype='complex')
    for j in np.arange(0, jmax):
        for k in np.arange(0, j):
            FX, FY = psf.getFXFY(j, k)
            
            fmeasurement += fimage*(FX + FY)
            
    print(fmeasurement)

    fmeasurement_d = np.zeros((nx, ny), dtype='complex')
    for j in np.arange(0, jmax):
        for k in np.arange(0, j):
            FX, FY = psf.getFXFY(j, k, defocus=True)
            
            fmeasurement_d += fimage*(FX + FY)

    print(fmeasurement_d)

    for (fft_image, label) in [(fmeasurement, ""), (fmeasurement_d, "_d")]:

        measurement = fft.ifft2(fft_image).real
        measurement = np.roll(np.roll(measurement, int(nx/2), axis=0), int(ny/2), axis=1)
        
        extent=[0., 1., 0., 1.]
        plot_aspect=(extent[1]-extent[0])/(extent[3]-extent[2])#*2/3 
        
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(6, 3)
        ax1.imshow(image[::-1],extent=extent,cmap=my_cmap,origin='lower')
        ax1.set_aspect(aspect=plot_aspect)
        
        ax2.imshow(measurement[::-1],extent=extent,cmap=my_cmap,origin='lower')
        ax2.set_aspect(aspect=plot_aspect)
        
        fig.savefig("measurement" + str(trial) + label + ".png")
        plt.close(fig)
