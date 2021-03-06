# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 20:20:51 2017

@author: nigul
"""

import sys
sys.path.append('../utils')

import numpy as np
import matplotlib.pyplot as plt
import stat

axis_label_fs = 15
panel_label_fs = 15

dat_orig = np.loadtxt("results_uniform.txt")

#0 is the inde, but not unique

sig_var = dat_orig[:, 2]
sig_vars_plot = np.sort(np.unique(sig_var))

freq_errs_gp = np.ones(len(sig_vars_plot))
freq_errs_fg = np.ones(len(sig_vars_plot))
freq_errs_d2 = np.ones(len(sig_vars_plot))
freq_errs_kalman = np.ones(len(sig_vars_plot))
freq_errs_se_gp = np.ones(len(sig_vars_plot))
freq_errs_se_fg = np.ones(len(sig_vars_plot))
freq_errs_se_d2 = np.ones(len(sig_vars_plot))
freq_errs_se_kalman = np.ones(len(sig_vars_plot))

ell_errs_gp = np.ones(len(sig_vars_plot))
ell_errs_fg = np.ones(len(sig_vars_plot))
ell_errs_d2 = np.ones(len(sig_vars_plot))
ell_errs_kalman = np.ones(len(sig_vars_plot))
ell_errs_se_gp = np.ones(len(sig_vars_plot))
ell_errs_se_fg = np.ones(len(sig_vars_plot))
ell_errs_se_d2 = np.ones(len(sig_vars_plot))
ell_errs_se_kalman = np.ones(len(sig_vars_plot))

#quality41 = abs(f_orig-f_ls_orig)/f_orig        
#quality42 = abs(f_orig-f_gp_orig)/f_orig
#indices41 = np.where(quality41 < 1)
#indices42 = np.where(quality42 < 1)

i = 0
total_d2_filtered_out = 0.0
total = 0.0

for sig_var_plot in sig_vars_plot:
    print sig_var_plot
    dat = dat_orig
    sig_var = dat[:, 2]
    indices = np.where(sig_var == sig_var_plot)[0]
    assert len(indices) > 0
    dat = dat_orig[indices]
    #sig_var = dat[:, 2]
    #indices = np.where(sig_var > last_sig_var_plot)[0]
    #print len(indices)
    #assert len(indices) > 0
    #dat = dat[indices]
    
    n = dat[:, 1]
    sig_var = dat[:, 2]
    freq = dat[:, 3]
    freq_gp = dat[:, 4]
    freq_fg = dat[:, 5]
    freq_d2 = dat[:, 6]
    freq_kalman = dat[:, 7]
    length_scale = dat[:, 8]
    length_scale_gp = dat[:, 9]
    length_scale_fg = dat[:, 10]
    length_scale_d2 = dat[:, 11]
    length_scale_kalman = dat[:, 12]

    (freq_err, freq_err_se) = mean_with_se(abs(freq-freq_gp)/freq)
    freq_errs_gp[i] = freq_err
    freq_errs_se_gp[i] = freq_err_se

    (freq_err, freq_err_se) = mean_with_se(abs(freq-freq_fg)/freq)
    freq_errs_fg[i] = freq_err
    freq_errs_se_fg[i] = freq_err_se

    indices_d2 = np.where(length_scale_d2 > 0.1)[0]
    print "D2 filtered out", len(indices_d2)
    total_d2_filtered_out += len(indices_d2)
    total += len(sig_var)
    (freq_err, freq_err_se) = mean_with_se(abs(freq[indices_d2]-freq_d2[indices_d2])/freq[indices_d2])
    freq_errs_d2[i] = freq_err
    freq_errs_se_d2[i] = freq_err_se

    (freq_err, freq_err_se) = mean_with_se(abs(freq-freq_kalman)/freq)
    freq_errs_kalman[i] = freq_err
    freq_errs_se_kalman[i] = freq_err_se

    ###########################################################################
 
    (ell_err, ell_err_se) = mean_with_se(abs(length_scale-length_scale_gp)/length_scale)
    ell_errs_gp[i] = ell_err
    ell_errs_se_gp[i] = ell_err_se

    (ell_err, ell_err_se) = mean_with_se(abs(length_scale-length_scale_fg)/length_scale)
    ell_errs_fg[i] = ell_err
    ell_errs_se_fg[i] = ell_err_se

    indices_d2 = np.where(length_scale_d2 > 0.1)[0]
    (ell_err, ell_err_se) = mean_with_se(abs(length_scale[indices_d2]-length_scale_d2[indices_d2])/length_scale[indices_d2])
    ell_errs_d2[i] = ell_err
    ell_errs_se_d2[i] = ell_err_se

    (ell_err, ell_err_se) = mean_with_se(abs(length_scale-length_scale_kalman)/length_scale)
    ell_errs_kalman[i] = ell_err
    ell_errs_se_kalman[i] = ell_err_se
   
    i += 1    

print "total_d2_filtered_out", total_d2_filtered_out/total

fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
fig.set_size_inches(6, 4)

ax.text(0.05, 0.9,'(a)', horizontalalignment='center', transform=ax.transAxes, fontsize=panel_label_fs)

snr = -sig_vars_plot/(sig_vars_plot-1.0)

handles_gp = ax.plot(snr, freq_errs_gp, 'r--', alpha=0.9)
handles_fg = ax.plot(snr, freq_errs_fg, 'g-', alpha=0.9)
handles_d2 = ax.plot(snr, freq_errs_d2, 'k-.', alpha=0.9)
handles_kalman = ax.plot(snr, freq_errs_kalman, 'b:', alpha=0.9)
ax.fill_between(snr, freq_errs_gp + freq_errs_se_gp, freq_errs_gp - freq_errs_se_gp, alpha=0.5, facecolor='lightsalmon', interpolate=True)
ax.fill_between(snr, freq_errs_fg + freq_errs_se_fg, freq_errs_fg - freq_errs_se_fg, alpha=0.5, facecolor='lightgreen', interpolate=True)
ax.fill_between(snr, freq_errs_d2 + freq_errs_se_d2, freq_errs_d2 - freq_errs_se_d2, alpha=0.5, facecolor='gray', interpolate=True)
ax.fill_between(snr, freq_errs_kalman + freq_errs_se_kalman, freq_errs_kalman - freq_errs_se_kalman, alpha=0.5, facecolor='lightblue', interpolate=True)

ax.legend(handles_gp + handles_fg + handles_d2 + handles_kalman, ["Gaussian process", "Factor graph", "$D^2$", "Kalman filter"],
            numpoints = 1,
            scatterpoints=1,
            bbox_to_anchor=(0., 0.9, 1., .1),
            #loc='upper right', 
            ncol=1,
            fontsize=10, labelspacing=0.7)

ax.set_ylabel(r'$\overline{\Delta f}$', fontsize=axis_label_fs)#,fontsize=20)
ax.set_xlabel(r'$S/N$', fontsize=axis_label_fs)#,fontsize=20)
ax.set_xlim(min(snr), max(snr))

fig.savefig("stats_freq.pdf")

###############################################################################

fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
fig.set_size_inches(6, 4)

ax.text(0.05, 0.9,'(b)', horizontalalignment='center', transform=ax.transAxes, fontsize=panel_label_fs)

snr = -sig_vars_plot/(sig_vars_plot-1.0)

handles_gp = ax.plot(snr, ell_errs_gp, 'r--', alpha=0.9)
handles_fg = ax.plot(snr, ell_errs_fg, 'g-', alpha=0.9)
handles_d2 = ax.plot(snr, ell_errs_d2, 'k-.', alpha=0.9)
handles_kalman = ax.plot(snr, ell_errs_kalman, 'b:', alpha=0.9)
ax.fill_between(snr, ell_errs_gp + ell_errs_se_gp, ell_errs_gp - ell_errs_se_gp, alpha=0.5, facecolor='lightsalmon', interpolate=True)
ax.fill_between(snr, ell_errs_fg + ell_errs_se_fg, ell_errs_fg - ell_errs_se_fg, alpha=0.5, facecolor='lightgreen', interpolate=True)
ax.fill_between(snr, ell_errs_d2 + ell_errs_se_d2, ell_errs_d2 - ell_errs_se_d2, alpha=0.5, facecolor='gray', interpolate=True)
ax.fill_between(snr, ell_errs_kalman + ell_errs_se_kalman, ell_errs_kalman - ell_errs_se_kalman, alpha=0.5, facecolor='lightblue', interpolate=True)

#ax.legend(handles_gp + handles_fg + handles_d2 + handles_kalman, ["Gaussian process", "Factor graph", "$D^2$", "Kalman filter"],
#            numpoints = 1,
#            scatterpoints=1,
#            bbox_to_anchor=(0., 0.9, 1., .1),
#            #loc='upper right', 
#            ncol=1,
#            fontsize=10, labelspacing=0.7)

ax.set_ylabel(r'$\overline{\Delta \ell}$', fontsize=axis_label_fs)#,fontsize=20)
ax.set_xlabel(r'$S/N$', fontsize=axis_label_fs)#,fontsize=20)
ax.set_xlim(min(snr), max(snr))

fig.savefig("stats_ell.pdf")
