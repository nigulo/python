import numpy as np
#import opticspy
#import opticspy.test.test_surface
#import opticspy.aperture

import psf
import utils
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colorbar as cb
from matplotlib import cm

#Z = opticspy.zernike.Coefficient(Z11=1) 
#Z.zernikesurface()

nx = 100

extent=[0., 1., 0., 1.]
plot_aspect=(extent[1]-extent[0])/(extent[3]-extent[2])#*2/3 

def reverse_colourmap(cmap, name = 'my_cmap_r'):
     return mpl.colors.LinearSegmentedColormap(name, cm.revcmap(cmap._segmentdata))

#my_cmap = reverse_colourmap(plt.get_cmap('gnuplot'))
my_cmap = plt.get_cmap('winter')


num_rows = 5
num_cols = 5
num_tests = num_rows * num_cols

aperture_func = lambda xs: utils.aperture_circ(xs, 1.0, 15.0)

fig, ax = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(6, 6)

vals = np.zeros((nx, nx))
for x in np.arange(0, nx):
    for y in np.arange(0, nx):
        x1 = 2.0*(float(x) - nx/2) / nx
        y1 = 2.0*(float(y) - nx/2) / nx
        vals[x, y] = aperture_func(np.array([x1, y1]))
ax.imshow(vals.T,extent=extent,cmap=reverse_colourmap(plt.get_cmap('binary')),origin='lower', vmin=np.min(vals), vmax=np.max(vals))
ax.set_aspect(aspect=plot_aspect)

fig.savefig('aperture.png')
plt.close(fig)


fig1, (axes1) = plt.subplots(nrows=num_rows, ncols=num_cols)
fig1.set_size_inches(30, 30)

fig2, (axes2) = plt.subplots(nrows=num_rows, ncols=num_cols)
fig2.set_size_inches(30, 30)

axes1 = axes1.flatten()
axes2 = axes2.flatten()

arcsec_per_px = 0.0055
diameter = 20.0
wavelength = 5250.0

for index in np.arange(0, num_tests):
    if index == 0:
        pa = psf.phase_aberration([], start_index = 0)
    else:
        pa = psf.phase_aberration([0.]*(index-1) + [10.], start_index = 0)#[(index, 10.0)])
    vals = np.zeros((nx, nx))
    for x in np.arange(0, nx):
        for y in np.arange(0, nx):
            #x1 = np.sqrt(2)*(float(x) - nx/2) / nx
            #y1 = np.sqrt(2)*(float(y) - ny/2) / ny
            x1 = 2.0*(float(x) - nx/2) / nx
            y1 = 2.0*(float(y) - nx/2) / nx
            if x1**2+y1**2 <= 1.0:
                pa.calc_terms(np.array([[x1, y1]]))
                vals[x, y] = pa()
            else:
                vals[x, y] = 0.0

    ax = axes1[index]
    ax.imshow(vals.T,extent=extent,cmap=my_cmap,origin='lower', vmin=np.min(vals), vmax=np.max(vals))
    #ax1.set_title(r'Factor graph')
    #ax1.set_ylabel(r'$f$')
    #start, end = ax32.get_xlim()
    #ax1.xaxis.set_ticks(np.arange(5, end, 4.9999999))
    #ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
    #ax1.xaxis.labelpad = -1
    #ax1.set_xlabel(r'$l_{\rm coh}$')#,fontsize=20)
    ax.set_aspect(aspect=plot_aspect)
    ax.set_adjustable('box-forced')


    ctf = psf.coh_trans_func(aperture_func, pa, lambda u: 0.0)#np.sum(u**2))
    #ctf = psf.coh_trans_func(lambda u: 1.0, pa, lambda u: 0.0)#np.sum(u**2))

    fig3, (ax31, ax32) = plt.subplots(nrows=2, ncols=1)
    fig3.set_size_inches(4, 6)

    psf_vals = psf.psf(ctf, nx, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength).calc()
    ax31.hist(psf_vals.flatten(), bins=100)
    print(np.min(psf_vals), np.max(psf_vals), np.mean(psf_vals))
    psf_vals = utils.trunc(psf_vals, 1e-3)
    ax32.hist(psf_vals.flatten(), bins=100)
    print(np.min(psf_vals), np.max(psf_vals), np.mean(psf_vals))

    fig3.savefig('hist' + str(index) + '.png')
    plt.close(fig3)
    
    psf_vals = psf_vals[int(0.4*nx*2):int(0.6*nx*2),int(0.4*nx*2):int(0.6*nx*2)]
    #psf_vals = np.log(psf_vals)
    ax = axes2[index]
    ax.imshow(psf_vals,extent=extent,cmap=reverse_colourmap(plt.get_cmap('binary')),origin='lower', vmin=np.min(psf_vals), vmax=np.max(psf_vals))
    #ax1.set_title(r'Factor graph')
    #ax1.set_ylabel(r'$f$')
    #start, end = ax32.get_xlim()
    #ax1.xaxis.set_ticks(np.arange(5, end, 4.9999999))
    #ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
    #ax1.xaxis.labelpad = -1
    #ax1.set_xlabel(r'$l_{\rm coh}$')#,fontsize=20)
    ax.set_aspect(aspect=plot_aspect)
    ax.set_adjustable('box-forced')

    index += 1

    fig1.savefig('zernike.png')
    fig2.savefig('psf.png')

plt.close(fig1)
plt.close(fig2)

