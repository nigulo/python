import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import psf_basis
import zernike

def reverse_colourmap(cmap, name = 'my_cmap_r'):
     return mpl.colors.LinearSegmentedColormap(name, cm.revcmap(cmap._segmentdata))


def main():
    my_cmap = reverse_colourmap(plt.get_cmap('binary'))#plt.get_cmap('winter')
    
    
    zoom_factor = 1.0
    jmax = 6
    arcsec_per_px = 0.0155
    diameter = 100.0
    wavelength = 5250.0
    nx = 100
    defocus = 2.*np.pi
    
    psf = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength, defocus = defocus)
    psf.create_basis(do_fft=False, do_defocus=True)
    
    for defocus in [False, True]:
        
        # Init figures ############################################################
        num_cells = jmax*int(jmax/2) + jmax
        ncols = jmax
        nrows = int(num_cells / ncols)
        if ncols * nrows < num_cells:
            ncols += 1
            
        assert(ncols * nrows >= num_cells)
        fig_x, axes_x = plt.subplots(nrows=nrows, ncols=ncols)
        fig_x.set_size_inches(ncols*3, nrows*3)
        
        fig_y, axes_y = plt.subplots(nrows=nrows, ncols=ncols)
        fig_y.set_size_inches(ncols*3, nrows*3)

        extent=[0., 1., 0., 1.]
        plot_aspect=(extent[1]-extent[0])/(extent[3]-extent[2])#*2/3 
        ###########################################################################
    
        # Generate all the basis functions
        row = 0
        col = 0
        for j in np.arange(1, jmax + 1):
            n, m = zernike.get_nm(j)
    
            for k in np.arange(1, j + 1):
                n_p, m_p = zernike.get_nm(k)
                print(n, m, n_p, m_p)
    
                X, Y = psf.get_XY(j, k, defocus=defocus)
                
                # Do the plotting #################################################
    
                ax_x = axes_x[row][col]
                ax_y = axes_y[row][col]
                col += 1
                if col >= ncols:
                    col = 0
                    row += 1
                
                title_x = r'$X^{'+ str(m) + r',' + str(m_p) + r'}_{' + str(n) + ',' + str(n_p) + r'}$'
                title_y = r'$Y^{'+ str(m) + r',' + str(m_p) + r'}_{' + str(n) + ',' + str(n_p) + r'}$'
                ax_x.text(0.95, 0.01, title_x,
                        verticalalignment='bottom', horizontalalignment='right',
                        transform=ax_x.transAxes,
                        color='yellow', fontsize=10)
        
                ax_y.text(0.95, 0.01, title_y,
                        verticalalignment='bottom', horizontalalignment='right',
                        transform=ax_y.transAxes,
                        color='yellow', fontsize=10)
    
                center = nx/2
                zoom_radius = nx*zoom_factor/2
                left = int(center-zoom_radius)
                right = int(center+zoom_radius)
                
                #X = utils.trunc(X, 1e-2)

                ax_x.imshow(X[::-1][left:right,left:right].real,extent=extent,cmap=my_cmap,origin='lower')
                ax_x.set_aspect(aspect=plot_aspect)
    
                ax_y.imshow(Y[::-1][left:right,left:right].real,extent=extent,cmap=my_cmap,origin='lower')
                ax_y.set_aspect(aspect=plot_aspect)
                
                
    
                ###################################################################
    
                if defocus:
                    fig_x.savefig("psf_defocus_x.png")
                    fig_y.savefig("psf_defocus_y.png")
                else:
                    fig_x.savefig("psf_x.png")
                    fig_y.savefig("psf_y.png")
    
    
        plt.close(fig_x)
        plt.close(fig_y)

if __name__ == "__main__":
    main()