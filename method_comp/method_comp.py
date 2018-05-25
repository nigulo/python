# -*- coding: utf-8 -*-
"""
Test of equivalence of general D^2 statistic in original and vector formulation
@author: nigul
"""

import sys
sys.path.append('../kalman/')
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
import GPR_QP
import kalman_utils as ku
import method_comp_utils as mcu

#cov_type = "periodic"
cov_type = "quasiperiodic"

n = 50
time_range = 200
t = np.random.uniform(0.0, time_range, n)
var = 1.0
sig_var = np.random.uniform(0.91, 0.91)
noise_var = var - sig_var
t = np.sort(t)
t -= np.mean(t)

#p = time_range/12.54321#
p = np.random.uniform(time_range/20, time_range/5)
freq = 1.0/p
mean = 0.0

if cov_type == "periodic":
    length_scale = 1e10*p
    k = mcu.calc_cov_p(t, freq, sig_var) + np.diag(np.ones(n) * noise_var)
else:
    length_scale = np.random.uniform(p, 4.0*p)
    k = mcu.calc_cov_qp(t, freq, length_scale, sig_var) + np.diag(np.ones(n) * noise_var)
    
l = la.cholesky(k)
s = np.random.normal(0.0, 1.0, n)

y = np.repeat(mean, n) + np.dot(l, s)
y += mean


kalman_utils = ku.kalman_utils(t, y, num_iterations=3)
#kalman_utils.add_component("periodic", [np.zeros(1), np.zeros(1), np.zeros(1)], {"j_max":2})
kalman_utils.add_component("quasiperiodic", [np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)], {"j_max":1})
kalman_utils.add_component("white_noise", [np.zeros(1)])

num_freqs = 10
num_cohs = 10

if cov_type == "periodic":
    num_cohs = 1
    
fig, (ax1) = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(6, 3)
ax1.plot(t, y, 'b+')
fig.savefig('data.png')
plt.close(fig)

#print freq, length_scale

d2_spec_color = np.zeros((num_cohs, num_freqs, 3))
fg_spec_color = np.zeros((num_cohs, num_freqs, 3))
full_gp_spec_color = np.zeros((num_cohs, num_freqs, 3))
kalman_spec_color = np.zeros((num_cohs, num_freqs, 3))
coh_ind = 0

t_cohs = np.linspace(0.1, length_scale*2, num_cohs)

max_loglik_full_gp = None
max_coh_full_gp = 0
max_freq_full_gp = 0

max_fg = None
max_coh_fg = 0
max_freq_fg = 0

max_d2 = None
max_coh_d2 = 0
max_freq_d2 = 0

max_loglik_kalman = None
max_coh_kalman = 0
max_freq_kalman = 0

o = np.ones(len(t))
y2 = y * y

for t_coh in t_cohs:
    fig, (ax1) = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(6, 3)

    d2_spec = np.zeros(num_freqs)
    fg_spec = np.zeros(num_freqs)
    full_gp_spec = np.zeros(num_freqs)
    kalman_spec = np.zeros(num_freqs)
    f_ind = 0
    fs = np.linspace(0.01, 2.0*freq, num_freqs)
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
        d2 = mcu.calc_d2(t, y, sel_fn, normalize=True)
        d2_spec[f_ind] = d2
        fg_spec[f_ind] = fg
        gpr_gp = GPR_QP.GPR_QP(sig_var=sig_var, length_scale=t_coh, freq=f, noise_var=noise_var, rot_freq=0.0, rot_amplitude=0.0, trend_var=0.0, c=0.0)
        t_test = np.linspace(min(t), max(t), 2)
        gpr_gp.init(t, y)
        (_, _, loglik) = gpr_gp.fit(t_test)
        full_gp_spec[f_ind] = -loglik
        
        loglik_kalman = mcu.calc_kalman(kalman_utils, t, y, sig_var, noise_var, t_coh, f, plot=f_ind==len(fs)/2-1, coh_ind=coh_ind, f_ind=f_ind)
        kalman_spec[f_ind] = -loglik_kalman
        
        if max_loglik_full_gp is None or loglik > max_loglik_full_gp:
            max_loglik_full_gp = loglik
            max_coh_full_gp = t_coh
            max_freq_full_gp = f
        if max_fg is None or -fg > max_fg:
            max_fg = -fg
            max_coh_fg = t_coh
            max_freq_fg = f
        if max_d2 is None or -d2 > max_d2:
            max_d2 = -d2
            max_coh_d2 = t_coh
            max_freq_d2 = f
        if max_loglik_kalman is None or loglik_kalman > max_loglik_kalman:
            max_loglik_kalman = loglik_kalman
            max_coh_kalman = t_coh
            max_freq_kalman = f
        
        f_ind += 1

    d2_spec_color[coh_ind, :, 0] = np.repeat(t_coh, len(fs))
    d2_spec_color[coh_ind, :, 1] = fs
    d2_spec_color[coh_ind, :, 2] = d2_spec

    fg_spec_color[coh_ind, :, 0] = np.repeat(t_coh, len(fs))
    fg_spec_color[coh_ind, :, 1] = fs
    fg_spec_color[coh_ind, :, 2] = fg_spec

    full_gp_spec_color[coh_ind, :, 0] = np.repeat(t_coh, len(fs))
    full_gp_spec_color[coh_ind, :, 1] = fs
    full_gp_spec_color[coh_ind, :, 2] = full_gp_spec

    kalman_spec_color[coh_ind, :, 0] = np.repeat(t_coh, len(fs))
    kalman_spec_color[coh_ind, :, 1] = fs
    kalman_spec_color[coh_ind, :, 2] = kalman_spec
    
    d2_spec = (d2_spec - min(d2_spec)) / (max(d2_spec) - min(d2_spec))
    fg_spec = (fg_spec - min(fg_spec)) / (max(fg_spec) - min(fg_spec))
    full_gp_spec = (full_gp_spec - min(full_gp_spec)) / (max(full_gp_spec) - min(full_gp_spec))
    kalman_spec = (kalman_spec - min(kalman_spec)) / (max(kalman_spec) - min(kalman_spec))

    opt_freq_d2 = fs[np.argmin(d2_spec)]
    opt_freq_fg = fs[np.argmin(fg_spec)]
    opt_freq_full_gp = fs[np.argmin(full_gp_spec)]
    opt_freq_kalman = fs[np.argmin(kalman_spec)]

    ax1.plot(fs, d2_spec, 'b-')
    ax1.plot(fs, fg_spec, 'r-')
    ax1.plot(fs, full_gp_spec, 'g-')
    ax1.plot(fs, kalman_spec, 'y-')
    min_y = 0.0#min(min(d2_spec), min(gp_spec))
    max_y = 1.0#max(max(d2_spec), max(gp_spec))
    ax1.plot([freq, freq], [min_y, max_y], 'k--')
    if cov_type == "periodic":
        ax1.set_title(r"SNR: " + str(round(sig_var/noise_var, 1)))
        fig.savefig('spec.png')
        print freq, opt_freq_d2, opt_freq_fg, opt_freq_full_gp
    else:
        ax1.set_title(r'$t_{coh}$ estimate: ' + str(round(t_coh, 2)) + ", truth: " + str(round(length_scale, 2)) + ", SNR: " + str(round(sig_var/noise_var, 1)))
        fig.savefig('spec' + str(coh_ind) + '.png')
        print length_scale, t_coh, ";", freq, opt_freq_d2, opt_freq_fg, opt_freq_full_gp, opt_freq_kalman
    plt.close(fig)

    coh_ind += 1

if num_cohs > 1:
    d2_min = min(np.concatenate(d2_spec_color[:, :, 2]))
    d2_max = max(np.concatenate(d2_spec_color[:, :, 2]))
    fg_min = min(np.concatenate(fg_spec_color[:, :, 2]))
    fg_max = max(np.concatenate(fg_spec_color[:, :, 2]))
    full_gp_min = min(np.concatenate(full_gp_spec_color[:, :, 2]))
    full_gp_max = max(np.concatenate(full_gp_spec_color[:, :, 2]))
    kalman_min = min(np.concatenate(kalman_spec_color[:, :, 2]))
    kalman_max = max(np.concatenate(kalman_spec_color[:, :, 2]))
    
    d2_spec_color[:, :, 2] = (d2_spec_color[:, :, 2] - d2_min) / (d2_max - d2_min)
    fg_spec_color[:, :, 2] = (fg_spec_color[:, :, 2] - fg_min) / (fg_max - fg_min)
    full_gp_spec_color[:, :, 2] = (full_gp_spec_color[:, :, 2] - full_gp_min) / (full_gp_max - full_gp_min)
    kalman_spec_color[:, :, 2] = (kalman_spec_color[:, :, 2] - kalman_min) / (kalman_max - kalman_min)
    
    d2_min = min(np.concatenate(d2_spec_color[:, :, 2]))
    d2_max = max(np.concatenate(d2_spec_color[:, :, 2]))
    fg_min = min(np.concatenate(fg_spec_color[:, :, 2]))
    fg_max = max(np.concatenate(fg_spec_color[:, :, 2]))
    full_gp_min = min(np.concatenate(full_gp_spec_color[:, :, 2]))
    full_gp_max = max(np.concatenate(full_gp_spec_color[:, :, 2]))
    kalman_min = min(np.concatenate(kalman_spec_color[:, :, 2]))
    kalman_max = max(np.concatenate(kalman_spec_color[:, :, 2]))
    
    #cmin=min(min(np.concatenate(d2_spec_color[:,:,2])), 
    #         min(np.concatenate(gp_spec_color[:,:,2])),
    #         min(np.concatenate(full_gp_spec_color[:,:,2])),
    #         min(np.concatenate(kalman_spec_color[:,:,2]))
    #         )
    #cmax=max(max(np.concatenate(d2_spec_color[:,:,2])), 
    #         max(np.concatenate(gp_spec_color[:,:,2])),
    #        max(np.concatenate(full_gp_spec_color[:,:,2])),
    #        max(np.concatenate(kalman_spec_color[:,:,2]))
    #         )
    
    extent=[d2_spec_color[:, :, 0].min(), d2_spec_color[:, :, 0].max(), d2_spec_color[:, :, 1].min(), d2_spec_color[:, :, 1].max()]
    plot_aspect=(extent[1]-extent[0])/(extent[3]-extent[2])*2/3 
    
    def reverse_colourmap(cmap, name = 'my_cmap_r'):
         return mpl.colors.LinearSegmentedColormap(name, cm.revcmap(cmap._segmentdata))
    
    my_cmap = reverse_colourmap(plt.get_cmap('gnuplot'))
    
    f, ((ax1, ax2, ax3, ax4)) = plt.subplots(4, 1, sharex='col', sharey='row')
    #f.tight_layout()
    f.set_size_inches(4, 8)
    
    ax1.imshow(d2_spec_color[:,:,2].T,extent=extent,cmap=my_cmap,origin='lower', vmin=d2_min, vmax=d2_max)
    ax1.set_title(r'$D^2$')
    #ax31.set_yticklabels(["{:10.1f}".format(1/t) for t in ax31.get_yticks()])
    #ax31.yaxis.labelpad = -16
    ax1.set_ylabel(r'$f$')
    #start, end = ax31.get_xlim()
    #ax31.xaxis.set_ticks(np.arange(5, end, 4.9999999))
    #ax31.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
    #ax31.xaxis.labelpad = -1
    #ax1.set_xlabel(r'$l_{\rm coh}$')#,fontsize=20)
    ax1.set_aspect(aspect=plot_aspect)
    ax1.set_adjustable('box-forced')
    ax1.scatter([length_scale], [freq], c='r', s=20)
    ax1.scatter([max_coh_d2], [max_freq_d2], c='b', s=20)
    
    ax2.imshow(fg_spec_color[:,:,2].T,extent=extent,cmap=my_cmap,origin='lower', vmin=fg_min, vmax=fg_max)
    ax2.set_title(r'Factor graph')
    ax2.set_ylabel(r'$f$')
    #start, end = ax32.get_xlim()
    #ax32.xaxis.set_ticks(np.arange(5, end, 4.9999999))
    #ax32.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
    #ax32.xaxis.labelpad = -1
    #ax2.set_xlabel(r'$l_{\rm coh}$')#,fontsize=20)
    ax2.set_aspect(aspect=plot_aspect)
    ax2.set_adjustable('box-forced')
    ax2.scatter([length_scale], [freq], c='r', s=20)
    ax2.scatter([max_coh_fg], [max_freq_fg], c='b', s=20)
    
    ax3.imshow(full_gp_spec_color[:,:,2].T,extent=extent,cmap=my_cmap,origin='lower', vmin=full_gp_min, vmax=full_gp_max)
    ax3.set_title(r'GP')
    ax3.set_ylabel(r'$f$')
    #start, end = ax33.get_xlim()
    #ax33.xaxis.set_ticks(np.arange(5, end, 4.9999999))
    #ax33.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
    #ax33.xaxis.labelpad = -1
    #ax3.set_xlabel(r'$l_{\rm coh}$')#,fontsize=20)
    ax3.set_aspect(aspect=plot_aspect)
    ax3.set_adjustable('box-forced')
    ax3.scatter([length_scale], [freq], c='r', s=20)
    ax3.scatter([max_coh_full_gp], [max_freq_full_gp], c='b', s=20)
    
    im4 = ax4.imshow(kalman_spec_color[:,:,2].T,extent=extent,cmap=my_cmap,origin='lower', vmin=kalman_min, vmax=kalman_max)
    ax4.set_title(r'Kalman filter')
    ax4.set_ylabel(r'$f$')
    #start, end = ax33.get_xlim()
    #ax33.xaxis.set_ticks(np.arange(5, end, 4.9999999))
    #ax33.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
    #ax33.xaxis.labelpad = -1
    ax4.set_xlabel(r'$\ell$')#,fontsize=20)
    ax4.set_aspect(aspect=plot_aspect)
    ax4.set_adjustable('box-forced')
    ax4.scatter([length_scale], [freq], c='r', s=20)
    ax4.scatter([max_coh_kalman], [max_freq_kalman], c='b', s=20)

    ax1.set_xlim((extent[0], extent[1]))
    ax1.set_ylim((extent[2], extent[3]))
    ax2.set_xlim((extent[0], extent[1]))
    ax2.set_ylim((extent[2], extent[3]))
    ax3.set_xlim((extent[0], extent[1]))
    ax3.set_ylim((extent[2], extent[3]))
    ax4.set_xlim((extent[0], extent[1]))
    ax4.set_ylim((extent[2], extent[3]))
    
    l_f = FormatStrFormatter('%1.2f')
    f.subplots_adjust(left=0.05, right=0.91, wspace=0.05)
    
    cbar_ax = f.add_axes([0.8, 0.20, 0.05, 0.60])
    f.colorbar(im4, cax=cbar_ax, format=l_f, label=r'Statistic')
    
    plt.savefig('spec_3d.pdf')
    plt.close()
