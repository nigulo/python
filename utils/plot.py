import matplotlib as mpl
mpl.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colorbar as cb
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.ticker import LogFormatterMathtext, FormatStrFormatter
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import pyhdust.triangle as triangle
import numpy as np

def reverse_colourmap(cmap, name = 'my_cmap_r'):
     return mpl.colors.LinearSegmentedColormap(name, cm.revcmap(cmap._segmentdata))


def create_cmap(cmap_name, reverse=True):
    if reverse:
        return reverse_colourmap(plt.get_cmap(cmap_name))
    else:
        return plt.get_cmap(cmap_name)

def default_size(nx, ny):
    return [nx/50, ny/50]
    
class plot:
    
    def __init__(self, nrows=1, ncols=1, width=3.33, height=2.0, size=None, extent=[0., 1., 0., 1.], title=None, smart_axis="xy"):
        if size is not None:
            width = size[0]
            height = size[1]
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
        fig.set_size_inches(width*ncols, height*nrows)
        if title is not None:
            fig.suptitle(title, fontsize=16)
        self.fig = fig
        self.axes = axes
        self.extent = extent
        self.title = None
        self.colorbars = dict()
        self.ims = dict()
        self.show_colorbar = False
        self.axis_title_font_size = 4*max(width, height)
        self.axis_label_font_size = 3*max(width, height)
        self.axis_units_font_size = 2.5*max(width, height)
        self.nrows = nrows
        self.ncols = ncols
        self.width = width
        self.height = height
        self.used_axis = set()
        
        if type(smart_axis) == str :
            self.smart_axis = smart_axis
        else:
            self.smart_axis = ""


    def set_axis_title_font_size(self, axis_title_font_size):
        self.axis_title_font_size = axis_title_font_size

    def set_axis_label_font_size(self, axis_label_font_size):
        self.axis_label_font_size = axis_label_font_size

    def set_axis_utits_font_size(self, axis_units_font_size):
        self.axis_units_font_size = axis_units_font_size


    def rescale_axis_title_font_size(self, scale):
        self.axis_title_font_size *= scale
    def rescale_axis_label_font_size(self, scale):
        self.axis_label_font_size *= scale

    def rescale_axis_utits_font_size(self, scale):
        self.axis_units_font_size *= scale
        
    '''
        cmap - "bwr", "binary", "winter", "Greys", etc...
    '''    
    def set_default_cmap(self, cmap_name, reverse=True):
        if cmap_name is not None:
            self.default_cmap = create_cmap(cmap_name, reverse)

    def get_default_cmap(self):
        if not hasattr(self, "default_cmap"):
            return None
        else:
            return self.default_cmap


    '''
        Set the current axis title
    '''
    def set_title(self, title):
        self.title = title


    '''
        Meant for internal usage mainly
    '''
    def get_ax(self, ax_index = None):
        if ax_index is None:
            return self.axes
            
        if isinstance(ax_index, (list, tuple, np.ndarray)):
            if len(ax_index) == 2:
                if isinstance(self.axes, (list, tuple, np.ndarray)):
                    if isinstance(self.axes[ax_index[0]], (list, tuple, np.ndarray)):
                        return self.axes[ax_index[0]][ax_index[1]]
                    else:
                        if (ax_index[0] == 0):
                            return self.axes[ax_index[1]]
                        else:
                            assert(ax_index[1] == 0)
                            return self.axes[ax_index[0]]
                else:
                    assert(ax_index[0] == 0 and ax_index[1] == 0)
                    return self.axes
                    
            elif len(ax_index) == 1:
                if isinstance(self.axes, (list, tuple, np.ndarray)):
                    return self.axes[ax_index[0]]
                else:
                    assert(ax_index[0] == 0)
                    return self.axes
            else:
                return self.axes
        else:
            if isinstance(self.axes, (list, tuple, np.ndarray)):
                return self.axes[ax_index]
            else:
                assert(ax_index == 0)
                return self.axis
                
    
    '''
        Plot 2d data
    '''          
    def plot(self, x, y, ax_index=None, params="k-"):
        ax = self.get_ax(ax_index)
        ax.plot(x, y, params)
        self.post_processing(ax)
        
    def set_log(self, ax_index=None, params="y"):
        ax = self.get_ax(ax_index)
        if "x" in params:
            ax.set_xscale("log")
        if "y" in params:
            ax.set_yscale("log")
    
    def legend(self, ax_index=None, legends=[], loc='upper right'):
        ax = self.get_ax(ax_index)
        ax.legend(legends, loc=loc)
        #ax.legend(legends, numpoints = 1,
        #                scatterpoints=1,
        #                loc=loc, ncol=1,
        #                fontsize=10, labelspacing=0.7)
        
    
    
    def set_default_colorbar(self, z_min=0., z_max=1., colorbar_prec=None):
        if colorbar_prec is None:
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
        self.default_colorbar = l_f
        
    def get_defaut_colorbar(self):
        if not hasattr(self, "default_colorbar"):
            return None
        else:
            return self.default_colorbar
        
    def set_colorbar(self, ax_index = None, show_colorbar = False, colorbar_prec=None):
        ax = self.get_ax(ax_index)
        
        if not show_colorbar:
            if ax in self.colorbars:
                #cbar_ax, _ = self.colorbars(ax)
                #cbar_ax.remove()
                del self.colorbars[ax]
        else:
            #if colorbar_prec is None:
            #    l_f = self.default_colorbar
            #else:
            #    l_f = FormatStrFormatter('%' + str(colorbar_prec) +'f')
                    
            #pos = ax.get_position().get_points()
            #x0 = pos[0, 0]
            #y0 = pos[0, 1]
            #x1 = pos[1, 0]
            #y1 = pos[1, 1]
            #width = x1 - x0
            
            #cbar_ax = self.fig.add_axes([x1, y0, width/20, y1-y0])
            #self.colorbars[ax] = cbar_ax
            self.colorbars[ax] = colorbar_prec
            #self.fig.colorbar(self.ims[ax], cax=cbar_ax, format=l_f)#, label=r'Label')
            
    def process_colorbar(self, ax_index):
        ax = self.get_ax(ax_index)
        
        if ax in self.colorbars:
            colorbar_prec = self.colorbars[ax]
        
            if colorbar_prec is None:
                l_f = self.default_colorbar
            else:
                l_f = FormatStrFormatter(f'%{colorbar_prec}f')
                    
            #pos = ax.get_position(original=True).get_points()
            #x0 = pos[0, 0]
            #y0 = pos[0, 1]
            #x1 = pos[1, 0]
            #y1 = pos[1, 1]
            #width = x1 - x0
                
            divider = make_axes_locatable(ax)
            cbar_ax = divider.append_axes("right", size="5%", pad=0.0)
                
            #cbar_ax = self.fig.add_axes([x1, y0 + xlabel_height, width/20, y1-y0])
            cbar_ax.tick_params(labelsize=self.axis_units_font_size)
            self.fig.colorbar(self.ims[ax], cax=cbar_ax, format=l_f)#, label=r'Label')
    
    '''
        Plot colormap
    '''          
    def colormap(self, dat, ax_index=None, vmin=None, vmax=None, show_colorbar=None, colorbar_prec=None, cmap_name=None, reverse_cmap=True):
        ax = self.get_ax(ax_index)
        if self.get_defaut_colorbar() is None:
            z_min = np.min(dat)
            z_max = np.max(dat)
            self.set_default_colorbar(z_min, z_max, colorbar_prec)
            
        if show_colorbar is None:
            show_colorbar = self.show_colorbar
        else:
            self.show_colorbar = show_colorbar
        
        self.set_default_cmap(cmap_name)
        if self.get_default_cmap is None:
            self.set_default_cmap("binary")
        if cmap_name is None:
            cmap = self.get_default_cmap()
        else:
            cmap = create_cmap(cmap_name, reverse_cmap)

        if self.extent is None:
            left, right = ax.get_xlim()
            bottom, top = ax.get_ylim()
            self.extent = [left, right, bottom, top]
            plot_aspect=(self.extent[1]-self.extent[0])/(self.extent[3]-self.extent[2])#*2/3 
            ax.set_aspect(aspect=plot_aspect)


        im = ax.imshow(dat[::-1], extent=self.extent, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        self.ims[ax] = im

        self.set_colorbar(ax_index, show_colorbar, colorbar_prec)
        self.post_processing(ax)

    def contour(self, x, y, z, ax_index = [], levels=None):
        ax = self.get_ax(ax_index)
        ax.contour(x, y, z, levels=levels)
        self.post_processing(ax)
        
    def post_processing(self, ax):
        if self.title is not None:
            ax.set_title(self.title, fontsize=self.axis_title_font_size)
        ax.tick_params(axis='x', labelsize=self.axis_units_font_size)
        ax.tick_params(axis='y', labelsize=self.axis_units_font_size)
        self.used_axis.add(ax)
        
    def vectors(self, x1s, x2s, y1s, y2s, ax_index = [], units='width', scale=None, color = 'k', key = '', key_pos = 'E'):
        ax = self.get_ax(ax_index)
        q = ax.quiver(x1s, x2s, y1s, y2s, units=units, scale = scale, color = color)
        if key is not None and key != '':
            ax.quiverkey(q, X=0.3, Y=1.1, U=10,
                 label=key, labelpos='E')

    def get_axis_limits(self, ax_index = None):
        ax = self.get_ax(ax_index)
        return ax.get_xlim(), ax.get_ylim()

    def get_axis_labels(self, ax_index = None):
        ax = self.get_ax(ax_index)
        return ax.get_xlabel(), ax.get_ylabel()

    def toggle_axis(self, ax_index = None, on=False):
        if ax_index is None:
            ax = None
        else:
            ax = self.get_ax(ax_index)
        if isinstance(self.axes, (list, tuple, np.ndarray)):
            axes = self.axes.flatten()
        else:
            axes = [self.axes]
        for ax1 in axes:
            if ax is None or ax == ax1:
                if isinstance(on, (list, tuple, np.ndarray)):
                    if on[0]:
                        ax1.get_xaxis().set_visible(True)
                    else:
                        ax1.get_xaxis().set_visible(False)
                    if on[1]:
                        ax1.get_yaxis().set_visible(True)
                    else:
                        ax1.get_yaxis().set_visible(False)
                else:
                    if on:
                        ax1.get_xaxis().set_visible(True)
                        ax1.get_yaxis().set_visible(True)
                    else:
                        ax1.get_xaxis().set_visible(False)
                        ax1.get_yaxis().set_visible(False)
                        

        
    def set_axis_limits(self, ax_index = None, limits = None):
        if ax_index is None:
            ax = None
        else:
            ax = self.get_ax(ax_index)
        if isinstance(self.axes, (list, tuple, np.ndarray)):
            axes = self.axes.flatten()
        else:
            axes = [self.axes]
        for ax1 in axes:
            if ax is None or ax == ax1:
                if limits is None:
                    ax1.set_xlim(left=None, right=None)
                    ax1.set_ylim(bottom=None, top=None)
                else:
                    if limits[0] is None:
                        ax1.set_xlim(left=None, right=None)
                    else:
                        ax1.set_xlim(left=limits[0][0], right=limits[0][1])
                    if limits[1] is None:
                        ax1.set_ylim(bottom=None, top=None)
                    else:
                        ax1.set_ylim(bottom=limits[1][0], top=limits[1][1])

    def set_axis_labels(self, ax_index = None, labels = None):
        if ax_index is None:
            ax = None
        else:
            ax = self.get_ax(ax_index)
        if isinstance(self.axes, (list, tuple, np.ndarray)):
            axes = self.axes.flatten()
        else:
            axes = [self.axes]
        for ax1 in axes:
            if ax is None or ax == ax1:
                if labels is None:
                    ax1.set_xlabel(None)
                    ax1.set_ylabel(None)
                else:                    
                    if isinstance(labels, (list, tuple, np.ndarray)):
                        ax1.set_xlabel(labels[0], fontsize=self.axis_label_font_size)
                        ax1.set_ylabel(labels[1], fontsize=self.axis_label_font_size)
                    else:
                        raise "Identical text for both axis? Really?"

    def get_axis_tick_labels(self, ax_index = None):
        ax = self.get_ax(ax_index)

        def map_fn(label):
            return label.get_text()
        return map(map_fn, ax.get_xticklabels()), map(map_fn, ax.get_yticklabels())

    def set_axis_tick_labels(self, ax_index = None, axis = "xy", labels = None):
        if ax_index is None:
            ax = None
        else:
            ax = self.get_ax(ax_index)
        if isinstance(self.axes, (list, tuple, np.ndarray)):
            axes = self.axes.flatten()
        else:
            axes = [self.axes]
        for ax1 in axes:
            if ax is None or ax == ax1:
                if "x" in axis:
                    if labels is None:
                        ax1.set_xticklabels([])
                    elif isinstance(labels, (list, tuple, np.ndarray)):
                        if axis[0] == "x":
                            ax1.set_xticklabels(labels[0], fontsize=self.axis_units_font_size)
                        else:
                            ax1.set_xticklabels(labels[1], fontsize=self.axis_units_font_size)
                    else:
                        ax1.set_xticklabels(labels, fontsize=self.axis_units_font_size)
                if "y" in axis:
                    if labels is None:
                        ax1.set_yticklabels([])
                    elif isinstance(labels, (list, tuple, np.ndarray)):
                        if axis[0] == "y":
                            ax1.set_yticklabels(labels[0], fontsize=self.axis_units_font_size)
                        else:
                            ax1.set_yticklabels(labels[1], fontsize=self.axis_units_font_size)
                    else:
                        ax1.set_yticklabels(labels, fontsize=self.axis_units_font_size)

    def set_axis_ticks(self, ax_index = None, ticks = None):
        if ax_index is None:
            ax = None
        else:
            ax = self.get_ax(ax_index)
        if isinstance(self.axes, (list, tuple, np.ndarray)):
            axes = self.axes.flatten()
        else:
            axes = [self.axes]
        for ax1 in axes:
            if ax is None or ax == ax1:
                if ticks is None:
                    ax1.set_xticks([])
                    ax1.set_yticks([])
                else:                    
                    if isinstance(ticks, (list, tuple, np.ndarray)):
                        ax1.set_xticks(ticks[0])
                        ax1.set_yticks(ticks[1])
                    else:
                        raise "Identical text for both axis? Really?"

    def set_axis_title(self, ax_index = None, title = None):
        if ax_index is None:
            ax = None
        else:
            ax = self.get_ax(ax_index)
        if isinstance(self.axes, (list, tuple, np.ndarray)):
            axes = self.axes.flatten()
        else:
            axes = [self.axes]
        for ax1 in axes:
            if ax is None or ax == ax1:               
                ax1.set_title(title, fontsize=self.axis_title_font_size)
    
    '''
        Plot histogram
    '''          
    def hist(self, data, ax_index = [], bins = 20):
        ax = self.get_ax(ax_index)
        hist, bin_edges = np.histogram(data, bins = bins)
        ax.bar(bin_edges[:-1], hist, width=(bin_edges[-1]-bin_edges[0])/bins)


    def fill(self, x, y_lower, y_upper, ax_index = [], color="lightsalmon", alpha=0.8):
        ax = self.get_ax(ax_index)
        ax.fill_between(x, y_lower, y_upper, alpha=alpha, facecolor=color, interpolate=True)
        

    '''
        Plot pair-wise 2d marginal distributions for the given set of parameter samples
        samples - KxN matrix, where K is the number of parameters and N is the number of samples
        labels - array of size K defining the labels for the parameters
    '''
    def plot_marginals(self, samples, labels):
        fig, ax = plt.subplots(nrows=samples.shape[1], ncols=samples.shape[1])
        fig.set_size_inches(6, 6)
        triangle.corner(samples[:,:], labels=labels, fig=self.fig)
        

    def save(self, file_name):
        first_in_row = dict()
        last_in_column = dict()
        # Do some post-processing
        for row in np.arange(self.nrows):
            for col in np.arange(self.ncols-1, -1, -1):
                ax_index = [row, col]
                self.process_colorbar(ax_index)
                ax = self.get_ax(ax_index)
                
                # Remove empty subplots
                if ax not in self.used_axis:
                    ax.axis('off')
                else:
                    first_in_row[row] = ax
                    last_in_column[col] = ax
        for row in np.arange(self.nrows):
            for col in np.arange(self.ncols):
                ax_index = [row, col]
                ax = self.get_ax(ax_index)
            
                # Smart axis handling
                if last_in_column[col] == ax:
                    if first_in_row[row] == ax:
                        #self.toggle_axis(ax_index = ax_index, on=True)
                        pass
                    else:
                        if "y" in self.smart_axis:
                            #self.toggle_axis(ax_index = ax_index, on=[True, False])
                            self.set_axis_tick_labels(ax_index=ax_index, axis="y", labels=None)
                else:
                    if first_in_row[row] == ax:
                        if "x" in self.smart_axis:
                            #self.toggle_axis(ax_index = ax_index, on=[False, True])
                            self.set_axis_tick_labels(ax_index=ax_index, axis="x", labels=None)
                    else:
                        if "x" in self.smart_axis:
                            if "y" in self.smart_axis:
                                #self.toggle_axis(ax_index = ax_index, on=False)
                                self.set_axis_tick_labels(ax_index=ax_index, labels=None)
                            else:
                                #self.toggle_axis(ax_index = ax_index, on=[False, True])
                                self.set_axis_tick_labels(ax_index=ax_index, axis="x", labels=None)
                        elif "y" in self.smart_axis:
                            #self.toggle_axis(ax_index = ax_index, on=[True, False])
                            self.set_axis_tick_labels(ax_index=ax_index, axis="y", labels=None)
            
        self.fig.savefig(file_name)
        


    def close(self):
        plt.close(self.fig)
