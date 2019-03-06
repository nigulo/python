import matplotlib as mpl
mpl.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colorbar as cb
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.ticker import LogFormatterMathtext, FormatStrFormatter
import matplotlib.colors as colors
import matplotlib.ticker as ticker

def reverse_colourmap(cmap, name = 'my_cmap_r'):
     return mpl.colors.LinearSegmentedColormap(name, cm.revcmap(cmap._segmentdata))

class plot_map:

    def __init__(self, nrows=1, ncols=1, width=4.3, height = 3., extent=[0., 1., 0., 1.]):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
        fig.set_size_inches(width*ncols, height*nrows)
        self.fig = fig
        self.axes = axes
        self.extent = extent
        self.plot_aspect=(extent[1]-extent[0])/(extent[3]-extent[2])#*2/3 
        self.my_cmap = reverse_colourmap(plt.get_cmap('binary'))#plt.get_cmap('winter')

    def plot(self, dat, ax_index = [], vmin=None, vmax=None, colorbar = True):
        if len(ax_index) == 2:
            ax = self.axes[ax_index[0]][ax_index[1]]
        elif len(ax_index) == 1:
            ax = self.axes[ax_index[0]]
        else:
            ax = self.axes
        im = ax.imshow(dat[::-1],extent=self.extent,cmap=self.my_cmap,origin='lower', vmin=vmin, vmax=vmax)

        ax.set_aspect(aspect=self.plot_aspect)
        
        if colorbar:
            l_f = FormatStrFormatter('%1.2f')

            pos = ax.get_position().get_points()
            x0 = pos[0, 0]
            y0 = pos[0, 1]
            x1 = pos[1, 0]
            y1 = pos[1, 1]
            width = x1 - x0
            
            cbar_ax = self.fig.add_axes([x1, y0, width/20, y1-y0])
            self.fig.colorbar(im, cax=cbar_ax, format=l_f)#, label=r'Label')


    def save(self, file_name):
        self.fig.savefig(file_name)


    def close(self):
        plt.close(self.fig)