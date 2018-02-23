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

offset = 1979.3452
down_sample_factor = 8

num_bs = 1000

def calc_cov(t, f, sig_var, trend_var, c):
    k = np.zeros((len(t), len(t)))
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            k[i, j] = sig_var*(np.cos(2 * np.pi*f*(t[i] - t[j]))) + trend_var * (t[i] - c) * (t[j] - c)
            k[j, i] = k[i, j]
    return k



real_dats = []

for root, dirs, dir_files in os.walk("cleaned_wo_rot"):
    for file in dir_files:
        if file[-4:] == ".dat":
            star = file[:-4]
            star = star.upper()
            if (star[-3:] == '.CL'):
                star = star[0:-3]
            if (star[0:2] == 'HD'):
                star = star[2:]
            if star == "SUNALL":
                continue
            dat = np.loadtxt("cleaned_wo_rot/"+file, usecols=(0,1), skiprows=0)
            t = dat[:,0]
            y = dat[:,1]
            t /= 365.25
            t += offset
            noise_var = mw_utils.get_seasonal_noise_var(t, y)

            if down_sample_factor >= 2:
                indices = np.random.choice(len(t), len(t)/down_sample_factor, replace=False, p=None)
                indices = np.sort(indices)
                t = t[indices]
                y = y[indices]
                noise_var = noise_var[indices]
    
            if max(t) - min(t) >= 30:
                real_dats.append(t)

files = []

for root, dirs, dir_files in os.walk("cleaned"):
    for file in dir_files:
        if file[-4:] == ".dat":
            star = file[:-4]
            star = star.upper()
            if (star[-3:] == '.CL'):
                star = star[0:-3]
            if (star[0:2] == 'HD'):
                star = star[2:]
            files.append(file)

def select_dataset():
    file = files[np.random.choice(len(files))]
    
    dat = np.loadtxt("cleaned/"+file, usecols=(0,1), skiprows=1)

    t_orig = dat[:,0]
    t_orig /= 365.25
    t_orig += offset


    if down_sample_factor >= 2:
        indices = np.random.choice(len(t_orig), len(t_orig)/down_sample_factor, replace=False, p=None)
        indices = np.sort(indices)
    
        t = t_orig[indices]

    t -= (max(t) + min(t)) / 2

    return t

means = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0]

axis_label_fs = 15
panel_label_fs = 15



setup_no = 0
for real_sampling in [False, True]:

    fig_stats, ax_stats_1 = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=False)
    fig_stats.set_size_inches(6, 4)
    if setup_no == 0:
        ax_stats_1.text(0.05, 0.9,'(a)', horizontalalignment='center', transform=ax_stats_1.transAxes, fontsize=panel_label_fs)
    else:
        ax_stats_1.text(0.05, 0.9,'(b)', horizontalalignment='center', transform=ax_stats_1.transAxes, fontsize=panel_label_fs)
        ax_stats_1.set_xlabel(r'$k$', fontsize=axis_label_fs)#,fontsize=20)
    ax_stats_1.set_xlim([min(means), max(means)])
    ax_stats_1.set_ylabel(r'$S_1$', fontsize=axis_label_fs)#,fontsize=20)
    
    #fig_stats.tight_layout()
    
    num_exp = len(means)
    bglst_err_means = np.zeros(num_exp)
    bglst_err_stds = np.zeros(num_exp)
    ls_err_means = np.zeros(num_exp)
    ls_err_stds = np.zeros(num_exp)
    gls_err_means = np.zeros(num_exp)
    gls_err_stds = np.zeros(num_exp)
    
    bglst_err_low_perc = np.zeros(num_exp)
    bglst_err_mean_perc = np.zeros(num_exp)
    bglst_err_high_perc = np.zeros(num_exp)
    ls_err_low_perc = np.zeros(num_exp)
    ls_err_mean_perc = np.zeros(num_exp)
    ls_err_high_perc = np.zeros(num_exp)
    gls_err_low_perc = np.zeros(num_exp)
    gls_err_mean_perc = np.zeros(num_exp)
    gls_err_high_perc = np.zeros(num_exp)
    
    bglst_err_means_bs = np.zeros((num_exp, num_bs))
    ls_err_means_bs = np.zeros((num_exp, num_bs))
    gls_err_means_bs = np.zeros((num_exp, num_bs))
    
    exp_no = 0
    for mean in means:
        print "Experiment with mean", mean

        if os.path.exists('offset_tests/'+str(setup_no)+'/offset_exp_' + str(exp_no) + '.pkl'):
            (true_freqs, bglst_freqs, gls_freqs, ls_freqs, dats) = pickle.load(open('offset_tests/'+str(setup_no)+'/offset_exp_' + str(exp_no) + '.pkl', 'rb'))
        else:

            dats = []
            num_rep = 2000
            true_freqs = np.zeros(num_rep)
            bglst_freqs = np.zeros(num_rep)
            gls_freqs = np.zeros(num_rep)
            ls_freqs = np.zeros(num_rep)
            
            for rep_index in np.arange(0, num_rep):
                if real_sampling:
                    t = real_dats[np.random.choice(len(real_dats))]
                    n = len(t)
                else:
                    time_range = 30.0
                    n = 200
                    t = np.random.uniform(0, time_range, n)
                    t = np.sort(t)
                min_t = min(t)
                duration = max(t) - min_t
                            
                var = 1.0
                sig_var = np.random.uniform(0.2, 0.8)
                noise_var = np.ones(n) * (var - sig_var)
                
                p = np.random.uniform(2.0, duration/1.5)
                f = 1.0/p
                true_freqs[rep_index] = f
            
                k = calc_cov(t, f, sig_var, 0.0, 0.0) + np.diag(noise_var)
                l = la.cholesky(k)
                s = np.random.normal(0, 1, n)
                
                y = np.repeat(mean, n) + np.dot(l, s)
                #y += mean
                
                #data = np.column_stack((t, y))
                #print data
                #np.savetxt("cyclic.txt", data, fmt='%f')
                #noise_var_prop = mw_utils.get_seasonal_noise_var(t, y)
                freqs = np.linspace(0.001, 0.5, 1000)

                power_ls = LombScargle(t, y-np.mean(y), np.sqrt(noise_var)).power(freqs, normalization='psd')
                
                ls_local_maxima_inds = mw_utils.find_local_maxima(power_ls)
                f_opt_ls_ind = np.argmax(power_ls[ls_local_maxima_inds])
                f_opt_ls = freqs[ls_local_maxima_inds][f_opt_ls_ind]
                ls_freqs[rep_index] = f_opt_ls
    
                power_gls = LombScargle(t, y, np.sqrt(noise_var)).power(freqs, normalization='psd')
                
                gls_local_maxima_inds = mw_utils.find_local_maxima(power_gls)
                f_opt_gls_ind = np.argmax(power_gls[gls_local_maxima_inds])
                f_opt_gls = freqs[gls_local_maxima_inds][f_opt_gls_ind]
                gls_freqs[rep_index] = f_opt_gls
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(t, y)
                
                w = np.ones(n) / noise_var
            
                bglst = BGLST.BGLST(t, y, w, 
                                w_A = 2.0/np.var(y), A_hat = 0.0,
                                w_B = 2.0/np.var(y), B_hat = 0.0,
                                w_alpha = duration**2 / np.var(y), alpha_hat = slope, 
                                w_beta = 1.0 / (np.var(y) + intercept**2), beta_hat = intercept)
                
                
                (freqs, probs) = bglst.calc_all(min(freqs), max(freqs), len(freqs))
                bglst_local_maxima_inds = mw_utils.find_local_maxima(probs)
                f_opt_bglst_ind = np.argmax(probs[bglst_local_maxima_inds])
                f_opt_bglst = freqs[bglst_local_maxima_inds][f_opt_bglst_ind]
                bglst_freqs[rep_index] = f_opt_bglst
                dats.append((t, y, w))
        
            dats = np.asarray(dats)
    
            with open('offset_tests/'+str(setup_no)+'/offset_exp_' + str(exp_no) + '.pkl', 'wb') as f:
                pickle.dump((true_freqs, bglst_freqs, gls_freqs, ls_freqs, dats), f)

        bglst_errs = np.abs(bglst_freqs-true_freqs)/true_freqs
        gls_errs = np.abs(gls_freqs-true_freqs)/true_freqs
        ls_errs = np.abs(ls_freqs-true_freqs)/true_freqs
        
        bglst_err_means[exp_no] = np.mean(bglst_errs)
        bglst_err_stds[exp_no] = np.std(bglst_errs)
        gls_err_means[exp_no] = np.mean(gls_errs)
        gls_err_stds[exp_no] = np.std(gls_errs)
        ls_err_means[exp_no] = np.mean(ls_errs)
        ls_err_stds[exp_no] = np.std(ls_errs)
        
        bglst_err_low_perc[exp_no] = np.percentile(bglst_errs, 5)
        bglst_err_mean_perc[exp_no] = np.percentile(bglst_errs, 50)
        bglst_err_high_perc[exp_no] = np.percentile(bglst_errs, 95)
        
        gls_err_low_perc[exp_no] = np.percentile(gls_errs, 5)
        gls_err_mean_perc[exp_no] = np.percentile(gls_errs, 50)
        gls_err_high_perc[exp_no] = np.percentile(gls_errs, 95)

        ls_err_low_perc[exp_no] = np.percentile(ls_errs, 5)
        ls_err_mean_perc[exp_no] = np.percentile(ls_errs, 50)
        ls_err_high_perc[exp_no] = np.percentile(ls_errs, 95)

        for bs_index in np.arange(0, num_bs):
            bglst_errs_bs = np.random.choice(bglst_errs, size=len(bglst_errs))
            gls_errs_bs = np.random.choice(gls_errs, size=len(gls_errs))
            ls_errs_bs = np.random.choice(ls_errs, size=len(ls_errs))
            bglst_err_means_bs[exp_no, bs_index] = np.mean(bglst_errs_bs)
            gls_err_means_bs[exp_no, bs_index] = np.mean(gls_errs_bs)
            ls_errs_bs = np.random.choice(ls_errs, size=len(ls_errs))
    
        exp_no += 1
    
    
    ###############################################################################
    # Plot experiment statistics
    
    line1, = ax_stats_1.plot(means, bglst_err_means, 'r-', label = "BGLST")
    line2, = ax_stats_1.plot(means, ls_err_means, 'b--', label = "GLS")
    line3, = ax_stats_1.plot(means, ls_err_means, 'g-.', label = "LS")

    ax_stats_1.fill_between(means, np.percentile(bglst_err_means_bs, 2.5, axis=1), np.percentile(bglst_err_means_bs, 97.5, axis=1), alpha=0.2, facecolor='lightsalmon', interpolate=True)
    ax_stats_1.fill_between(means, np.percentile(gls_err_means_bs, 2.5, axis=1), np.percentile(gls_err_means_bs, 97.5, axis=1), alpha=0.2, facecolor='lightblue', interpolate=True)
    ax_stats_1.fill_between(means, np.percentile(ls_err_means_bs, 2.5, axis=1), np.percentile(ls_err_means_bs, 97.5, axis=1), alpha=0.2, facecolor='lightgreen', interpolate=True)
    
    #ax_stats_2.plot(means, bglst_err_stds, 'r-')
    #ax_stats_2.plot(means, ls_t_err_stds, 'b--')
    #ax_stats_2.plot(means, ls_err_stds, 'g-.')

    #ax_stats_2.plot(means, bglst_err_high_perc - bglst_err_low_perc, 'r-')
    #ax_stats_2.plot(means, ls_t_err_high_perc - ls_t_err_low_perc, 'b--')
    #ax_stats_2.plot(means, ls_err_high_perc - ls_err_low_perc, 'g-.')
    
    ax_stats_1.set_position([0.12, 0.12, 0.85, 0.75])
    if setup_no == 0:
        ax_stats_1.legend(handles=[line1, line2, line3], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0., handletextpad=0., fontsize=panel_label_fs)#, columnspacing=10)
    
    #ax_stats_1.legend(handles=[line1, line2, line3],
    #        numpoints = 1,
    #        scatterpoints=1,
    #        loc='upper left', ncol=1,
    #        fontsize=11, labelspacing=0.7)
    
    #ax_stats_1.plot([min(means), max(means)], [0.5, 0.5], 'k:')
    #ax_stats_2.plot([min(means), max(means)], [0.0, 0.0], 'k:')
    fig_stats.savefig("offset_tests/offset_stats_" + str(setup_no+1) + ".pdf")
    plt.close()

    setup_no += 1        
    
###############################################################################
    
        
        
        
            
