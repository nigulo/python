
import sys
import os
sys.path.append('..')
sys.path.append('../utils')
sys.path.append('../method_comp')

import numpy as np
import scipy
from scipy import stats
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colorbar as cb
from matplotlib import cm
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.ticker import LogFormatterMathtext, FormatStrFormatter
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from collections import OrderedDict as od
import numpy.linalg as la
import kalman_utils as ku
import method_comp_utils as mcu
import MountWilson.mw_utils as mw_utils
import plot

def find_cycle(star, dat):
    
    offset = 1979.3452

    t = dat[:,0]
    y = dat[:,1]
    
    t /= 365.25
    t += offset
    
    duration = t[-1] - t[0]
    t -= duration/2
    
    orig_var = np.var(y)
    noise_var = mw_utils.get_seasonal_noise_var(t, y, per_point=True, num_years=1.0)
    
    # just for removing the duplicates
    t, y, noise_var = mw_utils.downsample(t, y, noise_var)#, min_time_diff=30.0/365.25, average=True)

    n = len(t)

    fig = plot.plot(size=plot.default_size(500, 300))
    fig.plot(t, y, 'b+')
    fig.plot([min(t), max(t)], [np.mean(y), np.mean(y)], 'k:')
    fig.plot([min(t), max(t)], np.mean(y)+[np.sqrt(orig_var), np.sqrt(orig_var)], 'k--')
    fig.plot(t, np.mean(y)+np.sqrt(noise_var), 'k-')
    fig.save(f'{star}_data.png')
    
    noise_var = np.mean(noise_var)
    sig_var = orig_var - noise_var
    assert(sig_var > 0)

    #cov_type = "periodic"
    cov_type = "quasiperiodic"
    
    kalman_utils = ku.kalman_utils(t, y, num_iterations=3)
    #kalman_utils.add_component("periodic", [np.zeros(1), np.zeros(1), np.zeros(1)], {"j_max":2})
    kalman_utils.add_component("quasiperiodic", [np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)], {"j_max":1})
    kalman_utils.add_component("white_noise", [np.zeros(1)])
    
    num_freqs = 100
    num_cohs = 10
    
    if cov_type == "periodic":
        num_cohs = 1
    
    #print freq, length_scale
    
    fg_spec_color = np.zeros((num_cohs, num_freqs, 3))
    kalman_spec_color = np.zeros((num_cohs, num_freqs, 3))
    coh_ind = 0
    
    t_cohs = np.linspace(2.0, duration, num_cohs)
    
    max_fg = None
    max_coh_fg = 0
    max_freq_fg = 0
    
    max_loglik_kalman = None
    max_coh_kalman = 0
    max_freq_kalman = 0
    
    o = np.ones(len(t))
    y2 = y * y
    
    for t_coh in t_cohs:
        fig, (ax1) = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(6, 3)
    
        fg_spec = np.zeros(num_freqs)
        kalman_spec = np.zeros(num_freqs)
        f_ind = 0
        fs = np.linspace(.8, 2., num_freqs)
        for f in fs:
            if cov_type == "periodic":
                k = mcu.calc_cov_p(t, f, sig_var)
                sel_fn = mcu.calc_sel_fn_p(t, f, sig_var)
            else:
                k = mcu.calc_cov_qp(t, f, t_coh, sig_var)
                sel_fn = mcu.calc_sel_fn_qp(t, f, t_coh, sig_var)
                
            k += np.diag(np.ones(n) * noise_var)
            g_1, g_2, g_log = mcu.calc_g(t, sig_var, noise_var, k)
            fg = np.dot(o, np.dot(g_1, y2)) - np.dot(y, np.dot(g_2, y)) +0.5*g_log
            fg_spec[f_ind] = fg
            t_test = np.linspace(min(t), max(t), 2)
            
            loglik_kalman = mcu.calc_kalman(kalman_utils, t, y, sig_var, noise_var, t_coh, f, plot=f_ind==len(fs)/2-1, coh_ind=coh_ind, f_ind=f_ind)
            kalman_spec[f_ind] = -loglik_kalman
            
            if max_fg is None or -fg > max_fg:
                max_fg = -fg
                max_coh_fg = t_coh
                max_freq_fg = f
            if max_loglik_kalman is None or loglik_kalman > max_loglik_kalman:
                max_loglik_kalman = loglik_kalman
                max_coh_kalman = t_coh
                max_freq_kalman = f
            
            f_ind += 1
    
        fg_spec_color[coh_ind, :, 0] = np.repeat(t_coh, len(fs))
        fg_spec_color[coh_ind, :, 1] = fs
        fg_spec_color[coh_ind, :, 2] = fg_spec
    
        kalman_spec_color[coh_ind, :, 0] = np.repeat(t_coh, len(fs))
        kalman_spec_color[coh_ind, :, 1] = fs
        kalman_spec_color[coh_ind, :, 2] = kalman_spec
        
        fg_spec = (fg_spec - min(fg_spec)) / (max(fg_spec) - min(fg_spec))
        kalman_spec = (kalman_spec - min(kalman_spec)) / (max(kalman_spec) - min(kalman_spec))
    
        opt_freq_fg = fs[np.argmin(fg_spec)]
        opt_freq_kalman = fs[np.argmin(kalman_spec)]
    
        ax1.plot(fs, fg_spec, 'r-')
        ax1.plot(fs, kalman_spec, 'y-')
        min_y = 0.0#min(min(d2_spec), min(gp_spec))
        max_y = 1.0#max(max(d2_spec), max(gp_spec))
        if cov_type == "periodic":
            ax1.set_title(r"SNR: " + str(round(sig_var/noise_var, 1)))
            fig.savefig(f'{star}_spec.png')
            print(opt_freq_fg)
        else:
            ax1.set_title(r'$t_{coh}$ estimate: ' + str(round(t_coh, 2)) + ", SNR: " + str(round(sig_var/noise_var, 1)))
            fig.savefig(f'{star}_spec{coh_ind}.png')
            print(t_coh, ";", opt_freq_fg, opt_freq_kalman)
        plt.close(fig)
    
        coh_ind += 1
    
    if num_cohs > 1:
        fg_min = min(np.concatenate(fg_spec_color[:, :, 2]))
        fg_max = max(np.concatenate(fg_spec_color[:, :, 2]))
        kalman_min = min(np.concatenate(kalman_spec_color[:, :, 2]))
        kalman_max = max(np.concatenate(kalman_spec_color[:, :, 2]))
        
        fg_spec_color[:, :, 2] = (fg_spec_color[:, :, 2] - fg_min) / (fg_max - fg_min)
        kalman_spec_color[:, :, 2] = (kalman_spec_color[:, :, 2] - kalman_min) / (kalman_max - kalman_min)
        
        fg_min = min(np.concatenate(fg_spec_color[:, :, 2]))
        fg_max = max(np.concatenate(fg_spec_color[:, :, 2]))
        kalman_min = min(np.concatenate(kalman_spec_color[:, :, 2]))
        kalman_max = max(np.concatenate(kalman_spec_color[:, :, 2]))
            
        extent=[fg_spec_color[:, :, 0].min(), fg_spec_color[:, :, 0].max(), fg_spec_color[:, :, 1].min(), fg_spec_color[:, :, 1].max()]
        plot_aspect=(extent[1]-extent[0])/(extent[3]-extent[2])*2/3 
        
        def reverse_colourmap(cmap, name = 'my_cmap_r'):
             return mpl.colors.LinearSegmentedColormap(name, cm.revcmap(cmap._segmentdata))
        
        my_cmap = reverse_colourmap(plt.get_cmap('gnuplot'))
        
        f, ((ax2, ax4)) = plt.subplots(2, 1, sharex='col', sharey='row')
        #f.tight_layout()
        f.set_size_inches(6, 8)
            
        ax2.imshow(fg_spec_color[:,:,2].T,extent=extent,cmap=my_cmap,origin='lower', vmin=fg_min, vmax=fg_max)
        ax2.set_title(r'Factor graph')
        ax2.set_ylabel(r'$f$')
        #start, end = ax32.get_xlim()
        #ax32.xaxis.set_ticks(np.arange(5, end, 4.9999999))
        #ax32.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
        #ax32.xaxis.labelpad = -1
        #ax2.set_xlabel(r'$l_{\rm coh}$')#,fontsize=20)
        ax2.set_aspect(aspect=plot_aspect)
        ax2.set_adjustable('box')
        ax2.scatter([max_coh_fg], [max_freq_fg], c='b', s=20)
            
        im4 = ax4.imshow(kalman_spec_color[:,:,2].T,extent=extent,cmap=my_cmap,origin='lower', vmin=kalman_min, vmax=kalman_max)
        ax4.set_title(r'Kalman filter')
        ax4.set_ylabel(r'$f$')
        #start, end = ax33.get_xlim()
        #ax33.xaxis.set_ticks(np.arange(5, end, 4.9999999))
        #ax33.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
        #ax33.xaxis.labelpad = -1
        ax4.set_xlabel(r'$\ell$')#,fontsize=20)
        ax4.set_aspect(aspect=plot_aspect)
        ax4.set_adjustable('box')
        ax4.scatter([max_coh_kalman], [max_freq_kalman], c='b', s=20)
    
        ax2.set_xlim((extent[0], extent[1]))
        ax2.set_ylim((extent[2], extent[3]))
        ax4.set_xlim((extent[0], extent[1]))
        ax4.set_ylim((extent[2], extent[3]))
        
        l_f = FormatStrFormatter('%1.2f')
        f.subplots_adjust(left=0.05, right=0.91, wspace=0.05)
        
        cbar_ax = f.add_axes([0.9, 0.20, 0.05, 0.60])
        f.colorbar(im4, cax=cbar_ax, format=l_f, label=r'Statistic')
        
        plt.savefig(f'{star}_spec_3d.pdf')
        plt.close(fig)

if (__name__ == '__main__'):

    data_dir = "data"
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    star = None
    if len(sys.argv) > 2:
        star = sys.argv[2]
        while star[0] == '0': # remove leading zeros
            star = star[1:]
        
    skiprows = 1
    data_found = False
    for root, dirs, dir_files in os.walk(data_dir):
        for file in dir_files:
            if file[-4:] == ".dat":
                file_star = file[:-4]
                file_star = file_star.upper()
                if (file_star[-3:] == '.CL'):
                    file_star = file_star[0:-3]
                if (file_star[0:2] == 'HD'):
                    file_star = file_star[2:]
                while file_star[0] == '0': # remove leading zeros
                    file_star = file_star[1:]
                if star is None or star == file_star:
                    dat = np.loadtxt(data_dir+"/"+file, usecols=(0,1), skiprows=skiprows)
                    find_cycle(file_star, dat)
