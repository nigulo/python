# -*- coding: utf-8 -*-
"""
Test of equivalence of general D^2 statistic in original and vector formulation
@author: nigul
"""

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

#cov_type = "periodic"
cov_type = "quasiperiodic"

def calc_cov_qp(t, f, length_scale, sig_var):
    k = np.zeros((len(t), len(t)))
    inv_l2 = 1.0/length_scale/length_scale
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            k[i, j] = sig_var*np.exp(-inv_l2*(t[i]-t[j])**2)*(np.cos(2 * np.pi*f*(t[i] - t[j])))
            k[j, i] = k[i, j]
    return k

def calc_cov_p(t, f, sig_var):
    k = np.zeros((len(t), len(t)))
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            k[i, j] = sig_var*(np.cos(2 * np.pi*f*(t[i] - t[j])))
            k[j, i] = k[i, j]
    return k


def calc_sel_fn_qp(t, f, length_scale, sig_var):
    k = np.zeros((len(t), len(t)))
    inv_l2 = 1.0/length_scale/length_scale
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            k[i, j] = np.exp(-np.log(2)*inv_l2*(t[i]-t[j])**2)*(1.0 + 2.0*(np.cos(2 * np.pi*f*(t[i] - t[j]))))
            k[j, i] = k[i, j]
    return k

def calc_sel_fn_p(t, f, sig_var):
    k = np.zeros((len(t), len(t)))
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            k[i, j] = 1.0 + 2.0*(np.cos(2 * np.pi*f*(t[i] - t[j])))
            k[j, i] = k[i, j]
    return k

n = 50
time_range = 200
t = np.random.uniform(0.0, time_range, n)
var = 1.0
sig_var = np.random.uniform(0.2, 0.9)
noise_var = var - sig_var
mean = 0.5

#p = time_range/12.54321#
p = np.random.uniform(time_range/200, time_range/5)
freq = 1.0/p
mean = 0.0

if cov_type == "periodic":
    length_scale = 1e10*p
    k = calc_cov_p(t, freq, sig_var) + np.diag(np.ones(n) * noise_var)
else:
    length_scale = np.random.uniform(p/2.0, 4.0*p)
    k = calc_cov_qp(t, freq, length_scale, sig_var) + np.diag(np.ones(n) * noise_var)
    
l = la.cholesky(k)
s = np.random.normal(0, 1, n)

y = np.repeat(mean, n) + np.dot(l, s)
y += mean

def calc_d2(k, normalize):
    d2 = 0.0
    norm = 0.0
    i = 0
    for ti in t:
        j = 0
        for tj in t:
            g12 = sig_var + k[i, j]
            d2 += g12 * (y[i] - y[j]) ** 2
            norm += g12
            j += 1
        i += 1
    if normalize:
        return d2 / norm / 2.0
    else:
        return d2
        

def calc_gp(k):
    d2 = 0.0
    norm = 0.0
    i = 0
    for ti in t:
        j = 0
        for tj in t:
            kij = k[i, j]
            if i == j:
                kij = 0.0
            #assert(sig_var + noise_var > k[i, j])
            g12 = 1.0 / ((sig_var + noise_var)**2 - kij**2)
            d2 += -g12 * (sig_var + noise_var) * y[i]**2
            #d2 += g12 * (sig_var + noise_var) * y[j]**2
            d2 += +g12 * k[i, j] * y[i] * y[j]
            d2 += -0.5*np.log((sig_var + noise_var)**2 - kij**2)
            #d2 += g12 * (y[i] - y[j]) ** 2
            norm += g12
            j += 1
        i += 1
    return -d2# / norm / 2.0

num_freqs = 100
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
gp_spec_color = np.zeros((num_cohs, num_freqs, 3))
full_gp_spec_color = np.zeros((num_cohs, num_freqs, 3))
coh_ind = 0

t_cohs = np.linspace(length_scale/4.0, length_scale*4.0, num_cohs)

max_loglik_full_gp = None
max_coh_full_gp = 0
max_freq_full_gp = 0
max_gp = None
max_coh_gp = 0
max_freq_gp = 0
max_d2 = None
max_coh_d2 = 0
max_freq_d2 = 0
for t_coh in t_cohs:
    fig, (ax1) = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(6, 3)

    d2_spec = np.zeros(num_freqs)
    d2a_spec = np.zeros(num_freqs)
    gp_spec = np.zeros(num_freqs)
    full_gp_spec = np.zeros(num_freqs)
    f_ind = 0
    fs = np.linspace(0.01, 2.0*freq, num_freqs)
    for f in fs:
        if cov_type == "periodic":
            k = calc_cov_p(t, f, sig_var)
            sel_fn = calc_sel_fn_p(t, f, sig_var)
        else:
            k = calc_cov_qp(t, f, t_coh, sig_var)
            sel_fn = calc_sel_fn_qp(t, f, t_coh, sig_var)
        gp = calc_gp(k)
        k += np.diag(np.ones(n) * noise_var)
        d2 = calc_d2(sel_fn, normalize=True)
        d2a = calc_d2(k, normalize=False)
        d2_spec[f_ind] = d2
        d2a_spec[f_ind] = d2a
        gp_spec[f_ind] = gp
        gpr_gp = GPR_QP.GPR_QP(sig_var=sig_var, length_scale=t_coh, freq=f, noise_var=noise_var, rot_freq=0.0, rot_amplitude=0.0, trend_var=0.0, c=0.0)
        t_test = np.linspace(min(t), max(t), 2)
        gpr_gp.init(t, y)
        (_, _, loglik) = gpr_gp.fit(t_test)
        full_gp_spec[f_ind] = -loglik
        if max_loglik_full_gp is None or loglik > max_loglik_full_gp:
            max_loglik_full_gp = loglik
            max_coh_full_gp = t_coh
            max_freq_full_gp = f
        if max_gp is None or -gp > max_gp:
            max_gp = -gp
            max_coh_gp = t_coh
            max_freq_gp = f
        if max_d2 is None or -d2 > max_d2:
            max_d2 = -d2
            max_coh_d2 = t_coh
            max_freq_d2 = f
        
        f_ind += 1

    d2_spec_color[coh_ind, :, 0] = np.repeat(t_coh, len(fs))
    d2_spec_color[coh_ind, :, 1] = fs
    d2_spec_color[coh_ind, :, 2] = d2_spec

    gp_spec_color[coh_ind, :, 0] = np.repeat(t_coh, len(fs))
    gp_spec_color[coh_ind, :, 1] = fs
    gp_spec_color[coh_ind, :, 2] = gp_spec

    full_gp_spec_color[coh_ind, :, 0] = np.repeat(t_coh, len(fs))
    full_gp_spec_color[coh_ind, :, 1] = fs
    full_gp_spec_color[coh_ind, :, 2] = full_gp_spec

    d2_spec = (d2_spec - min(d2_spec)) / (max(d2_spec) - min(d2_spec))
    d2a_spec = (d2a_spec - min(d2a_spec)) / (max(d2a_spec) - min(d2a_spec))
    gp_spec = (gp_spec - min(gp_spec)) / (max(gp_spec) - min(gp_spec))
    full_gp_spec = (full_gp_spec - min(full_gp_spec)) / (max(full_gp_spec) - min(full_gp_spec))

    opt_freq_d2 = fs[np.argmin(d2_spec)]
    opt_freq_d2a = fs[np.argmin(d2a_spec)]
    opt_freq_gp = fs[np.argmin(gp_spec)]
    opt_freq_full_gp = fs[np.argmin(full_gp_spec)]

    ax1.plot(fs, d2_spec, 'b-')
    ax1.plot(fs, d2a_spec, 'b--')
    ax1.plot(fs, gp_spec, 'r-')
    ax1.plot(fs, full_gp_spec, 'g-')
    min_y = 0.0#min(min(d2_spec), min(gp_spec))
    max_y = 1.0#max(max(d2_spec), max(gp_spec))
    ax1.plot([freq, freq], [min_y, max_y], 'k--')
    if cov_type == "periodic":
        ax1.set_title(r"SNR: " + str(round(sig_var/noise_var, 1)))
        fig.savefig('spec.png')
        print freq, opt_freq_d2, opt_freq_gp, opt_freq_full_gp
    else:
        ax1.set_title(r'$t_{coh}$ estimate: ' + str(round(t_coh, 2)) + ", truth: " + str(round(length_scale, 2)) + ", SNR: " + str(round(sig_var/noise_var, 1)))
        fig.savefig('spec' + str(coh_ind) + '.png')
        print length_scale, t_coh, ";", freq, opt_freq_d2, opt_freq_gp, opt_freq_full_gp
    plt.close(fig)

    coh_ind += 1

if num_cohs > 1:
    d2_min = min(np.concatenate(d2_spec_color[:, :, 2]))
    d2_max = max(np.concatenate(d2_spec_color[:, :, 2]))
    gp_min = min(np.concatenate(gp_spec_color[:, :, 2]))
    gp_max = max(np.concatenate(gp_spec_color[:, :, 2]))
    full_gp_min = min(np.concatenate(full_gp_spec_color[:, :, 2]))
    full_gp_max = max(np.concatenate(full_gp_spec_color[:, :, 2]))
    
    d2_spec_color[:, :, 2] = (d2_spec_color[:, :, 2] - d2_min) / (d2_max - d2_min)
    gp_spec_color[:, :, 2] = (gp_spec_color[:, :, 2] - gp_min) / (gp_max - gp_min)
    full_gp_spec_color[:, :, 2] = (full_gp_spec_color[:, :, 2] - full_gp_min) / (full_gp_max - full_gp_min)
    
    
    cmin=min(min(np.concatenate(d2_spec_color[:,:,2])), 
             min(np.concatenate(gp_spec_color[:,:,2])),
             min(np.concatenate(full_gp_spec_color[:,:,2]))
             )
    cmax=max(max(np.concatenate(d2_spec_color[:,:,2])), 
             max(np.concatenate(gp_spec_color[:,:,2])),
            max(np.concatenate(full_gp_spec_color[:,:,2]))
             )
    
    extent=[d2_spec_color[:, :, 0].min(), d2_spec_color[:, :, 0].max(), d2_spec_color[:, :, 1].min(), d2_spec_color[:, :, 1].max()]
    plot_aspect=(extent[1]-extent[0])/(extent[3]-extent[2])*2/3 
    
    def reverse_colourmap(cmap, name = 'my_cmap_r'):
         return mpl.colors.LinearSegmentedColormap(name, cm.revcmap(cmap._segmentdata))
    
    my_cmap = reverse_colourmap(plt.get_cmap('gnuplot'))
    
    f, ((ax31, ax32, ax33)) = plt.subplots(1, 3, sharex='col', sharey='row')
    f.tight_layout()
    f.set_size_inches(10, 2.5)
    
    ax31.imshow(d2_spec_color[:,:,2].T,extent=extent,cmap=my_cmap,origin='lower', vmin=cmin, vmax=cmax)
    ax31.set_title(r'$D^2$')
    #ax31.set_yticklabels(["{:10.1f}".format(1/t) for t in ax31.get_yticks()])
    #ax31.yaxis.labelpad = -16
    ax31.set_ylabel(r'$f')
    #start, end = ax31.get_xlim()
    #ax31.xaxis.set_ticks(np.arange(5, end, 4.9999999))
    #ax31.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
    #ax31.xaxis.labelpad = -1
    ax31.set_xlabel(r'$l_{\rm coh}$')#,fontsize=20)
    ax31.set_aspect(aspect=plot_aspect)
    ax31.set_adjustable('box-forced')
    ax31.scatter([length_scale], [freq], c='r', s=20)
    ax31.scatter([max_coh_d2], [max_freq_d2], c='b', s=20)
    
    ax32.imshow(gp_spec_color[:,:,2].T,extent=extent,cmap=my_cmap,origin='lower', vmin=cmin, vmax=cmax)
    ax32.set_title(r'GP')
    #start, end = ax32.get_xlim()
    #ax32.xaxis.set_ticks(np.arange(5, end, 4.9999999))
    #ax32.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
    #ax32.xaxis.labelpad = -1
    ax32.set_xlabel(r'$l_{\rm coh}$')#,fontsize=20)
    ax32.set_aspect(aspect=plot_aspect)
    ax32.set_adjustable('box-forced')
    ax32.scatter([length_scale], [freq], c='r', s=20)
    ax32.scatter([max_coh_gp], [max_freq_gp], c='b', s=20)
    
    im33 = ax33.imshow(full_gp_spec_color[:,:,2].T,extent=extent,cmap=my_cmap,origin='lower', vmin=cmin, vmax=cmax)
    ax33.set_title(r'Full GP')
    #start, end = ax33.get_xlim()
    #ax33.xaxis.set_ticks(np.arange(5, end, 4.9999999))
    #ax33.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
    #ax33.xaxis.labelpad = -1
    ax33.set_xlabel(r'$l_{\rm coh}$')#,fontsize=20)
    ax33.set_aspect(aspect=plot_aspect)
    ax33.set_adjustable('box-forced')
    ax33.scatter([length_scale], [freq], c='r', s=20)
    ax33.scatter([max_coh_full_gp], [max_freq_full_gp], c='b', s=20)
    
    
    l_f = FormatStrFormatter('%1.2f')
    f.subplots_adjust(left=0.05, right=0.91, wspace=0.05)
    
    cbar_ax32 = f.add_axes([0.925, 0.14, 0.02, 0.74])
    f.colorbar(im33, cax=cbar_ax32, format=l_f, label=r'$D^2$')
    
    plt.savefig('spec_3d.png')
    plt.close()