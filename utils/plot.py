import matplotlib as mpl
mpl.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colorbar as cb
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.ticker import LogFormatterMathtext, FormatStrFormatter
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import numpy as np

def reverse_colourmap(cmap, name = 'my_cmap_r'):
     return mpl.colors.LinearSegmentedColormap(name, cm.revcmap(cmap._segmentdata))

class plot:

    def __init__(self, nrows=1, ncols=1, width=4.3, height = 3., extent=[0., 1., 0., 1.], title=None):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
        fig.set_size_inches(width*ncols, height*nrows)
        fig.suptitle(title, fontsize=16)
        self.fig = fig
        self.axes = axes
        self.extent = extent
        self.my_cmap = reverse_colourmap(plt.get_cmap('binary'))#plt.get_cmap('winter')
        self.title = None
        self.colorbars = dict()

    def set_color_map(self, color_map, reverse=True):
        if reverse:
            self.my_cmap = reverse_colourmap(plt.get_cmap(color_map))
        else:
            self.my_cmap = plt.get_cmap(color_map)

    '''
        Set the current axis title
    '''
    def set_title(self, title):
        self.title = title


    def get_ax(self, ax_index = []):
        if len(ax_index) == 2:
            ax = self.axes[ax_index[0]][ax_index[1]]
        elif len(ax_index) == 1:
            ax = self.axes[ax_index[0]]
        else:
            ax = self.axes
        return ax
    
    def plot(self, x, y, ax_index = [], params="k-"):
        ax = self.get_ax(ax_index)
        ax.plot(x, y, params)
        self.post_processing(ax)

        
    def colormap(self, dat, ax_index = [], vmin=None, vmax=None, colorbar = False, colorbar_prec=None):
        ax = self.get_ax(ax_index)

        if self.extent is None:
            left, right = ax.get_xlim()
            bottom, top = ax.get_ylim()
            self.extent = [left, right, bottom, top]
            plot_aspect=(self.extent[1]-self.extent[0])/(self.extent[3]-self.extent[2])#*2/3 
            ax.set_aspect(aspect=plot_aspect)


        im = ax.imshow(dat[::-1],extent=self.extent,cmap=self.my_cmap,origin='lower', vmin=vmin, vmax=vmax)

        if colorbar and not ax in self.colorbars:
            if colorbar_prec is None:
                z_max = np.max(dat)
                z_min = np.min(dat)
                if z_max > z_min:
                    scale = np.log10(z_max-z_min)
                    colorbar_prec = int(np.floor(scale))
                else:
                    colorbar_prec = 1
                if colorbar_prec < 0:
                    l_f = FormatStrFormatter('%1.' + str(abs(colorbar_prec)) +'f')
                else:
                    l_f = FormatStrFormatter('%' + str(colorbar_prec) +'.f')
            else:
                l_f = FormatStrFormatter('%' + str(colorbar_prec) +'f')

            pos = ax.get_position().get_points()
            x0 = pos[0, 0]
            y0 = pos[0, 1]
            x1 = pos[1, 0]
            y1 = pos[1, 1]
            width = x1 - x0
            
            cbar_ax = self.fig.add_axes([x1, y0, width/20, y1-y0])
            self.colorbars[ax] = cbar_ax
            self.fig.colorbar(im, cax=cbar_ax, format=l_f)#, label=r'Label')
        self.post_processing(ax)

    def contour(self, x, y, z, ax_index = [], levels=None):
        ax = self.get_ax(ax_index)
        ax.contour(x, y, z, levels=levels)
        self.post_processing(ax)
        
    def post_processing(self, ax):
        if self.title is not None:
            ax.set_title(self.title)
        
        

    def vectors(self, x1s, x2s, y1s, y2s, ax_index = [], units='width', scale=None, color = 'k', key = '', key_pos = 'E'):
        ax = self.get_ax(ax_index)
        q = ax.quiver(x1s, x2s, y1s, y2s, units=units, scale = scale, color = color)
        if key is not None and key != '':
            ax.quiverkey(q, X=0.3, Y=1.1, U=10,
                 label=key, labelpos='E')


    def set_axis(self, ax_index = None, limits = [[], []], labels = None):
        if ax_index is None:
            ax = None
        elif len(ax_index) == 2:
            ax = self.axes[ax_index[0]][ax_index[1]]
        elif len(ax_index) == 1:
            ax = self.axes[ax_index[0]]
        else:
            ax = self.axes
        for ax1 in self.axes.flatten():
            if ax is None or ax == ax1:
                if len(limits[0]) == 0:
                    ax1.get_xaxis().set_visible(False)
                if len(limits[1]) == 0:
                    ax1.get_yaxis().set_visible(False)
                    
    def hist(self, data, ax_index = [], bins = 20):
        ax = self.get_ax(ax_index)
        hist, bin_edges = np.histogram(data, bins = bins)
        ax.bar(bin_edges[:-1], hist, width=(bin_edges[-1]-bin_edges[0])/bins)


    def save(self, file_name):
        self.fig.savefig(file_name)


    def close(self):
        plt.close(self.fig)
