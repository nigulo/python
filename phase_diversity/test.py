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

fig, (axes) = plt.subplots(nrows=5, ncols=5)
fig.set_size_inches(30, 30)

index = 1
for rows in axes:
    for ax in rows:
        pa = psf.phase_aberration([(index, 1.0)])
        vals = np.zeros((nx, ny))
        for x in np.arange(0, nx):
            for y in np.arange(0, ny):
                x1 = 2*(float(x) - nx/2) / nx
                y1 = 2*(float(y) - ny/2) / ny
                if x1**2+y1**2 <= 1:
                    vals[x, y] = pa.get_value([x1, y1])
                else:
                    vals[x, y] = 0.0
        index += 1
    
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


fig.savefig('test.png')
plt.close(fig)
