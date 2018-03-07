# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 20:20:51 2017

@author: nigul
"""

import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['figure.figsize'] = (20, 15)
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from astropy.stats import LombScargle
import os
import os.path
import BGLST
import mw_utils
from scipy import stats
import pickle
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter


offset = 1979.3452
down_sample_factor = 32

num_bs = 1000
n_sparse = 5

def calc_cov(t, f, sig_var, trend_var, c):
    k = np.zeros((len(t), len(t)))
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            k[i, j] = sig_var*(np.cos(2 * np.pi*f*(t[i] - t[j]))) + trend_var * (t[i] - c) * (t[j] - c)
            k[j, i] = k[i, j]
    return k



real_dats = []

axis_label_fs = 15
panel_label_fs = 15

max_count_per_bin = 1000
ns = np.array([1, 2, 5, 8, 10, 13, 15, 18, 20])
sig_vars = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.999])
#sig_vars = np.array([0.9999, 0.999, 0.99, 0.9])

num_ns = len(ns)
num_sig_vars = len(sig_vars)

bglst_err_means = np.zeros((num_ns, num_sig_vars))
gls_err_means = np.zeros((num_ns, num_sig_vars))

bglst_err_means_bs = np.zeros((num_ns, num_sig_vars, num_bs))
gls_err_means_bs = np.zeros((num_ns, num_sig_vars, num_bs))

if os.path.exists('offset_tests/offset_limit_exp.pkl'):
    (true_freqs, bglst_freqs, gls_freqs, ls_freqs) = pickle.load(open('offset_tests/offset_limit_exp.pkl', 'rb'))
else:

    true_freqs = np.zeros((num_ns, num_sig_vars, max_count_per_bin))
    bglst_freqs = np.zeros((num_ns, num_sig_vars, max_count_per_bin))
    gls_freqs = np.zeros((num_ns, num_sig_vars, max_count_per_bin))
    
    exp_no_1 = 0
    for n1 in ns:
        exp_no_2 = 0
        for sig_var in sig_vars:
            max_err = 0
            for rep_no in np.arange(0, max_count_per_bin):
                time_range = 30.0
                n = n_sparse
                
                p = np.random.uniform(time_range/float(n))
                f = 1.0/p
                
                t_orig = np.random.uniform(0, time_range, n)
                #t_orig = np.linspace(0, time_range, n) + np.random.rand(n)*0.2
                t = t_orig
                for i in np.arange(0, n1):
                    t = np.concatenate((t, t_orig + np.random.rand(n)*time_range/n*0.1))
    
                n = len(t)
                t = np.sort(t)
                min_t = min(t)
                max_t = max(t)
                duration = max_t - min_t
                            
                var = 1.0
                #sig_var = np.random.uniform(0.9999, 0.9999)
                noise_var = np.ones(n) * (var - sig_var)
    
                mean = np.random.uniform(-5.0, 5.0)
            
                k = calc_cov(t, f, sig_var, 0.0, 0.0) + np.diag(noise_var)
                l = la.cholesky(k)
                s = np.random.normal(0, 1, n)
                
                y = np.repeat(mean, n) + np.dot(l, s)
                emp_mean_err = abs(np.mean(y) - mean)
    
                true_freqs[exp_no_1, exp_no_2, rep_no] = f
                #y += mean
                
                #data = np.column_stack((t, y))
                #print data
                #np.savetxt("cyclic.txt", data, fmt='%f')
                #noise_var_prop = mw_utils.get_seasonal_noise_var(t, y)
                min_freq = 0.001
                max_freq = 2.1*f
                freqs = np.linspace(min_freq, max_freq, 1000)
           
                gls = LombScargle(t, y, np.sqrt(noise_var))
                power_gls = gls.power(freqs, normalization='psd')
                
                f_opt_gls_ind = np.argmax(power_gls)
                f_opt_gls = freqs[f_opt_gls_ind]
                gls_freqs[exp_no_1, exp_no_2, rep_no] = f_opt_gls
                
                min_power_gls = min(power_gls)
                max_power_gls = max(power_gls)
                norm_powers_gls = (power_gls - min_power_gls) / (max_power_gls - min_power_gls)
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(t, y)
                
                w = np.ones(n) / noise_var
            
                bglst = BGLST.BGLST(t, y, w, 
                                w_A = 2.0/np.var(y), A_hat = 0.0,
                                w_B = 2.0/np.var(y), B_hat = 0.0,
                                w_alpha = duration**2 / np.var(y), alpha_hat = slope, 
                                w_beta = 1.0 / (np.var(y) + intercept**2), beta_hat = intercept)
                
                
                (freqs, probs) = bglst.calc_all(min(freqs), max(freqs), len(freqs))
                f_opt_bglst_ind = np.argmax(probs)
                f_opt_bglst = freqs[f_opt_bglst_ind]
                bglst_freqs[exp_no_1, exp_no_2, rep_no] = f_opt_bglst
                
                min_prob_bglst = min(probs)
                max_prob_bglst = max(probs)
                norm_probs_bglst = (probs - min_prob_bglst) / (max_prob_bglst - min_prob_bglst)
                
                err = abs(abs(f_opt_bglst - f) - abs(f_opt_gls - f))
                if err > max_err:
                    max_err = err
                    ###############################################################
                    # Plot some most deviating results
                    fig1, (ax_a, ax_b) = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False)
                    fig1.set_size_inches(6, 7)
                
                    ax_a.text(0.95, 0.9,'(a)', horizontalalignment='center', transform=ax_a.transAxes, fontsize=panel_label_fs)
                    ax_b.text(0.95, 0.9,'(b)', horizontalalignment='center', transform=ax_b.transAxes, fontsize=panel_label_fs)
            
                    ax_a.plot(t, y, "k+")
                    t_model = np.linspace(min_t, max_t, 200)
                    y_model = gls.model(t_model, f)   
                    ax_a.plot(t_model, y_model, 'g-')
                    y_model = gls.model(t_model, f_opt_gls)   
                    ax_a.plot(t_model, y_model, 'b:')
                    (tau, mean, cov, y_model, loglik) = bglst.model(f_opt_bglst, t_model)   
                    ax_a.plot(t_model, y_model, 'r:')
                    
                    freqs = np.linspace(min_freq, max_freq, 1000)
                    
                    ax_b.plot(freqs, norm_probs_bglst, 'r-')
                    ax_b.plot([f_opt_bglst, f_opt_bglst], [0, 1], 'r-')
            
                    ax_b.plot(freqs, norm_powers_gls, 'b--')
                    ax_b.plot([f_opt_gls, f_opt_gls], [0, 1], 'b--')
                   
                    ax_b.plot([f, f], [0, 1], 'k:')
                    
                    ax_a.set_xlabel(r'$t$', fontsize=axis_label_fs)#,fontsize=20)
                    ax_a.set_ylabel(r'y', fontsize=axis_label_fs)#,fontsize=20)
                    ax_a.set_xlim([min(t), max(t)])
                    
                    ax_b.set_xlabel(r'$f$', fontsize=axis_label_fs)#,fontsize=20)
                    ax_b.set_ylabel(r'Power', fontsize=axis_label_fs)#,fontsize=20)
                    ax_b.set_xlim([min_freq, max_freq])
                    
                    fig1.savefig("offset_tests/fig_limit/" + str(exp_no_1) + "_" + str(exp_no_2) + ".eps")
                    plt.close(fig1)            
                    ###############################################################
            
            
            print f, f_opt_gls, f_opt_bglst
            exp_no_2 += 1
        
        exp_no_1 += 1

    with open('offset_tests/offset_limit_exp.pkl', 'wb') as f:
        pickle.dump((true_freqs, bglst_freqs, gls_freqs), f)

bglst_errs = np.abs(bglst_freqs-true_freqs)/true_freqs
gls_errs = np.abs(gls_freqs-true_freqs)/true_freqs

bglst_err_means = np.mean(bglst_errs, axis=2)
gls_err_means = np.mean(gls_errs, axis=2)

print gls_err_means
print bglst_err_means

if num_sig_vars == 1: # 2d plot
    for exp_index_1 in np.arange(0, num_ns):
        for exp_index_2 in np.arange(0, num_sig_vars):
            for bs_index in np.arange(0, num_bs):
                bglst_errs_bs = np.random.choice(bglst_errs[exp_index_1, exp_index_2,:], size=np.shape(bglst_errs)[1])
                gls_errs_bs = np.random.choice(gls_errs[exp_index_1, exp_index_2,:], size=np.shape(gls_errs)[1])
                bglst_err_means_bs[exp_index_1, exp_index_2, bs_index] = np.mean(bglst_errs_bs)
                gls_err_means_bs[exp_index_1, exp_index_2, bs_index] = np.mean(gls_errs_bs)
    
    ###############################################################################
    # Plot experiment statistics
    
    fig_stats, ax_stats_1 = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=False)
    fig_stats.set_size_inches(6, 4)
    #ax_stats_1.text(0.05, 0.9,'(b)', horizontalalignment='center', transform=ax_stats_1.transAxes, fontsize=panel_label_fs)
    ax_stats_1.set_xlabel(r'$n$', fontsize=axis_label_fs, labelpad = -1)#,fontsize=20)
    #ax_stats_1.set_xlim([0, 1])
    #ax_stats_1.set_ylim([0, 1])
    ax_stats_1.set_ylabel(r'$S_1$', fontsize=axis_label_fs)#,fontsize=20)

    n_sparse = float(n_sparse)
    
    line1, = ax_stats_1.plot(ns*n_sparse, bglst_err_means, 'r-', markerfacecolor='None', label = "BGLST")
    line2, = ax_stats_1.plot(ns*n_sparse, gls_err_means, 'b--', markerfacecolor='None', label = "GLS")
    
    ax_stats_1.fill_between(ns*n_sparse, np.percentile(bglst_err_means_bs, 2.5, axis=1), np.percentile(bglst_err_means_bs, 97.5, axis=1), alpha=0.2, facecolor='lightsalmon', interpolate=True)
    ax_stats_1.fill_between(ns*n_sparse, np.percentile(gls_err_means_bs, 2.5, axis=1), np.percentile(gls_err_means_bs, 97.5, axis=1), alpha=0.2, facecolor='lightblue', interpolate=True)
    
    #ax_stats_1.set_position([0.12, 0.12, 0.85, 0.75])
    #ax_stats_1.legend(handles=[line1, line2], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0., handletextpad=0., fontsize=panel_label_fs)#, columnspacing=10)
    
    ax_stats_1.legend(handles=[line1, line2],
            numpoints = 1,
            scatterpoints=1,
            loc='upper right', ncol=1,
            fontsize=11, labelspacing=0.7)
    
    ax_stats_1.set_xlim([min(ns*n_sparse), max(ns*n_sparse)])

else: #3d plot
    def reverse_colourmap(cmap, name = 'my_cmap_r'):
         return mpl.colors.LinearSegmentedColormap(name, cm.revcmap(cmap._segmentdata))
    
    #my_cmap = reverse_colourmap(plt.get_cmap('gnuplot'))
    my_cmap = plt.get_cmap('binary')#'afmhot')#''bwr')
    
    fig_stats = plt.figure()
    #fig_stats.tight_layout()
    fig_stats.set_size_inches(7, 4)
    ###############################################################################
    
    plot_data = (bglst_err_means - gls_err_means)/gls_err_means
    log_snr = np.log10(sig_vars/(sig_vars-1.0)*-1.0)
    #log_snr = sig_vars/(sig_vars-1.0)*-1.0
    
    extent=[min(ns)*n_sparse,max(ns)*n_sparse,min(log_snr),max(log_snr)]
    plot_aspect=(extent[1]-extent[0])/(extent[3]-extent[2])*2/3 
    
    cmin=np.min(plot_data)
    cmax=np.max(plot_data)
    
    ax = fig_stats.gca(projection='3d')

    x = np.zeros((len(ns), len(sig_vars)))
    y = np.zeros((len(ns), len(sig_vars)))
    
    i = 0
    for n in ns:
        j = 0
        for sig_var in sig_vars:
            x[i, j] = ns[i]*n_sparse
            y[i, j] = (sig_vars[j]-1.0)*-1.0
            j +=1
        i += 1
    
    ax.text(0, 2, 4.5, '(b)', horizontalalignment='center', transform=ax.transAxes, fontsize=panel_label_fs)
    ax.plot_surface(x, y, bglst_err_means, linewidth=0.2, antialiased=True, color = 'r', alpha = 0.5)
    ax.plot_surface(x, y, gls_err_means, linewidth=0.2, antialiased=True, color = 'b', alpha = 0.5)
    
    #im = ax.imshow(plot_data.T,extent=extent,cmap=my_cmap,origin='lower', vmin=cmin, vmax=2.0*cmax)
    ##ax.set_yticklabels(["{:10.1f}".format(1/t) for t in ax.get_yticks()])
    ##ax.yaxis.labelpad = -16
    ax.set_xlabel(r'$n$ ', fontsize=axis_label_fs)#, labelpad=1)
    ##start, end = ax.get_xlim()
    ##ax.xaxis.set_ticks(np.arange(5, end, 4.9999999))
    ##ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
    ##ax.xaxis.labelpad = -1
    ax.set_ylabel(r'$\sigma^2_{\rm N}$', fontsize=axis_label_fs)#, labelpad=-1)
    ax.set_zlabel(r'$S_1$', fontsize=axis_label_fs)#, labelpad=-1)
    #ax.set_aspect(aspect=plot_aspect)
    #ax.set_adjustable('box-forced')
    
    #l_f = FormatStrFormatter('%1.2f')
    #fig_stats.subplots_adjust(left=0.1, right=0.85, wspace=0.05)
    
    #cbar_ax = fig_stats.add_axes([0.84, 0.14, 0.02, 0.74])
    #cb = fig_stats.colorbar(im, cax=cbar_ax, format=l_f, label=r'$\Delta S_1$')
    #font = cb.ax.yaxis.label.get_font_properties()
    #font.set_size(axis_label_fs)
    ##matplotlib.font_manager.FontProperties(family='times new roman', style='italic', size=16)
    ##cb.ax.yaxis.label.set_font_properties(font)
    

fig_stats.savefig("offset_tests/offset_limit_stats.pdf")
plt.close()


###############################################################################
    
        
        
        
            
