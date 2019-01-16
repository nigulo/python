import numpy as np
import opticspy
import opticspy.test.test_surface
import opticspy.aperture

import psf
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colorbar as cb
from matplotlib import cm

#Z = opticspy.zernike.Coefficient(Z11=1) 
#Z.zernikesurface()

nx = 100
ny = 100

extent=[0., 1., 0., 1.]
plot_aspect=(extent[1]-extent[0])/(extent[3]-extent[2])#*2/3 

def reverse_colourmap(cmap, name = 'my_cmap_r'):
     return mpl.colors.LinearSegmentedColormap(name, cm.revcmap(cmap._segmentdata))

#my_cmap = reverse_colourmap(plt.get_cmap('gnuplot'))
my_cmap = plt.get_cmap('winter')


num_rows = 5
num_cols = 5
num_tests = num_rows * num_cols
fig1, (axes1) = plt.subplots(nrows=num_rows, ncols=num_cols)
fig1.set_size_inches(30, 30)

fig2, (axes2) = plt.subplots(nrows=num_rows, ncols=num_cols)
fig2.set_size_inches(30, 30)

axes1 = axes1.flatten()
axes2 = axes2.flatten()

index = 1
        
for index in np.arange(0, num_tests):
    pa = psf.phase_aberration([(index + 1, 1.0)])
    vals = np.zeros((nx, ny))
    for x in np.arange(0, nx):
        for y in np.arange(0, ny):
            #x1 = np.sqrt(2)*(float(x) - nx/2) / nx
            #y1 = np.sqrt(2)*(float(y) - ny/2) / ny
            x1 = 2.0*(float(x) - nx/2) / nx
            y1 = 2.0*(float(y) - ny/2) / ny
            if x1**2+y1**2 <= 1:
                vals[x, y] = pa.get_value([x1, y1])
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


    ctf = psf.coh_trans_func(lambda u: 1.0, pa, lambda u: 0.0)

    psf_vals = psf.psf(ctf, nx, ny).get_incoh_vals()
    
    psf_vals = np.log(psf_vals)
    ax = axes2[index]
    ax.imshow(psf_vals.T,extent=extent,cmap=my_cmap)#,origin='lower', vmin=np.min(psf_vals), vmax=np.max(psf_vals))
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
plt.close(fig1)

fig2.savefig('psf.png')
plt.close(fig2)