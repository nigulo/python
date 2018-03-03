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
down_sample_factor = 32

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

axis_label_fs = 15
panel_label_fs = 15

max_count_per_bin = 1000
max_emp_mean_err = 0.5
num_bins = 10
bin_locs = np.linspace(0.0, max_emp_mean_err, num_bins)

fig_stats, ax_stats_1 = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=False)
fig_stats.set_size_inches(6, 4)
#ax_stats_1.text(0.05, 0.9,'(b)', horizontalalignment='center', transform=ax_stats_1.transAxes, fontsize=panel_label_fs)
ax_stats_1.set_xlabel(r'$k$', fontsize=axis_label_fs)#,fontsize=20)
#ax_stats_1.set_xlim([0, 1])
#ax_stats_1.set_ylim([0, 1])
ax_stats_1.set_ylabel(r'$S_1$', fontsize=axis_label_fs)#,fontsize=20)

#fig_stats.tight_layout()

bglst_err_means = np.zeros(num_bins)
ls_err_means = np.zeros(num_bins)
gls_err_means = np.zeros(num_bins)

bglst_err_means_bs = np.zeros((num_bins, num_bs))
ls_err_means_bs = np.zeros((num_bins, num_bs))
gls_err_means_bs = np.zeros((num_bins, num_bs))

counts = np.zeros(num_bins, dtype=int)

if os.path.exists('offset_tests/offset_exp.pkl'):
    (true_freqs, bglst_freqs, gls_freqs, ls_freqs) = pickle.load(open('offset_tests/offset_exp.pkl', 'rb'))
else:

    true_freqs = np.zeros((num_bins, max_count_per_bin))
    bglst_freqs = np.zeros((num_bins, max_count_per_bin))
    gls_freqs = np.zeros((num_bins, max_count_per_bin))
    ls_freqs = np.zeros((num_bins, max_count_per_bin))
    
    while sum(counts) < max_count_per_bin * num_bins:
        bin_index = -1
        while bin_index < 0 or bin_index >= num_bins or counts[bin_index] >= max_count_per_bin:
            time_range = 30.0
            n = 5
            
            p = np.random.uniform(time_range/float(n))
            f = 1.0/p
            
            t_orig = np.random.uniform(0, time_range, n)
            #t_orig = np.linspace(0, time_range, n) + np.random.rand(n)*0.2
            t = t_orig
            for i in np.arange(0, 5):
                t = np.concatenate((t, t_orig + np.random.rand(n)*time_range/n*0.1))
            n = len(t)
            t = np.sort(t)
            min_t = min(t)
            max_t = max(t)
            duration = max_t - min_t
                        
            var = 1.0
            sig_var = np.random.uniform(0.99, 0.99)
            noise_var = np.ones(n) * (var - sig_var)

            mean = 0.0#np.random.uniform(-5.0, 5.0)
        
            k = calc_cov(t, f, sig_var, 0.0, 0.0) + np.diag(noise_var)
            l = la.cholesky(k)
            s = np.random.normal(0, 1, n)
            
            y = np.repeat(mean, n) + np.dot(l, s)
            emp_mean_err = abs(np.mean(y) - mean)

            bin_index = int(emp_mean_err*num_bins/max_emp_mean_err)

        true_freqs[bin_index, counts[bin_index]] = f
        #y += mean
        
        #data = np.column_stack((t, y))
        #print data
        #np.savetxt("cyclic.txt", data, fmt='%f')
        #noise_var_prop = mw_utils.get_seasonal_noise_var(t, y)
        min_freq = 0.001
        max_freq = 2.0*f
        freqs = np.linspace(min_freq, max_freq, 1000)

        ls = LombScargle(t, y-np.mean(y), np.sqrt(noise_var), fit_mean=False)
        power_ls = ls.power(freqs, normalization='psd')
        
        f_opt_ls_ind = np.argmax(power_ls)
        f_opt_ls = freqs[f_opt_ls_ind]
        ls_freqs[bin_index, counts[bin_index]] = f_opt_ls

        min_power_ls = min(power_ls)
        max_power_ls = max(power_ls)
        norm_powers_ls = (power_ls - min_power_ls) / (max_power_ls - min_power_ls)

        gls = LombScargle(t, y, np.sqrt(noise_var))
        power_gls = gls.power(freqs, normalization='psd')
        
        f_opt_gls_ind = np.argmax(power_gls)
        f_opt_gls = freqs[f_opt_gls_ind]
        gls_freqs[bin_index, counts[bin_index]] = f_opt_gls
        
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
        bglst_freqs[bin_index, counts[bin_index]] = f_opt_bglst
        
        min_prob_bglst = min(probs)
        max_prob_bglst = max(probs)
        norm_probs_bglst = (probs - min_prob_bglst) / (max_prob_bglst - min_prob_bglst)
        
        
        if counts[bin_index] % 10 == 0:
            fig1, (ax_a, ax_b) = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False)
            fig1.set_size_inches(6, 7)
        
            ax_a.text(0.95, 0.9,'(a)', horizontalalignment='center', transform=ax_a.transAxes, fontsize=panel_label_fs)
            ax_b.text(0.95, 0.9,'(b)', horizontalalignment='center', transform=ax_b.transAxes, fontsize=panel_label_fs)

            ax_a.plot(t, y, "k+")
            t_model = np.linspace(max_t, min_t, 200)
            y_model = gls.model(t_model, f)   
            ax_a.plot(t_model, y_model, 'r-')
            y_model = gls.model(t_model, f_opt_gls)   
            ax_a.plot(t_model, y_model, 'b:')
            y_model = ls.model(t_model, f_opt_ls)   
            ax_a.plot(t_model, y_model+np.mean(y), 'g:')
            (tau, mean, cov, y_model, loglik) = bglst.model(f_opt_bglst, t_model)   
            ax_a.plot(t_model, y_model, 'k:')
            
            ax_b.plot(freqs, norm_probs_bglst, 'r-')
            ax_b.plot([f_opt_bglst, f_opt_bglst], [0, 1], 'r-')
    
            ax_b.plot(freqs, norm_powers_gls, 'b--')
            ax_b.plot([f_opt_gls, f_opt_gls], [0, 1], 'b--')
    
            ax_b.plot(freqs, norm_powers_ls, 'g-.')
            ax_b.plot([f_opt_ls, f_opt_ls], [0, 1], 'g-.')
            
            ax_b.plot([f, f], [0, 1], 'k:')
            
            ax_a.set_xlabel(r'$t$', fontsize=axis_label_fs)#,fontsize=20)
            ax_a.set_ylabel(r'y', fontsize=axis_label_fs)#,fontsize=20)
            ax_a.set_xlim([min(t), max(t)])
            
            ax_b.set_xlabel(r'$f$', fontsize=axis_label_fs)#,fontsize=20)
            ax_b.set_ylabel(r'Power', fontsize=axis_label_fs)#,fontsize=20)
            ax_b.set_xlim([min_freq, max_freq])
            
            fig1.savefig("offset_tests/temp/" + str(bin_index) + "_" + str(counts[bin_index]) + ".eps")
            plt.close(fig1)
        
        print f, f_opt_ls, f_opt_gls, f_opt_bglst
        
        counts[bin_index] += 1
        #print counts

    with open('offset_tests/offset_exp.pkl', 'wb') as f:
        pickle.dump((true_freqs, bglst_freqs, gls_freqs, ls_freqs), f)

bglst_errs = np.abs(bglst_freqs-true_freqs)/true_freqs
gls_errs = np.abs(gls_freqs-true_freqs)/true_freqs
ls_errs = np.abs(ls_freqs-true_freqs)/true_freqs

bglst_err_means = np.mean(bglst_errs, axis=1)
gls_err_means = np.mean(gls_errs, axis=1)
ls_err_means = np.mean(ls_errs, axis=1)

print gls_err_means, ls_err_means

for bin_index in np.arange(0, num_bins):
    for bs_index in np.arange(0, num_bs):
        bglst_errs_bs = np.random.choice(bglst_errs[bin_index,:], size=np.shape(bglst_errs)[1])
        gls_errs_bs = np.random.choice(gls_errs[bin_index,:], size=np.shape(gls_errs)[1])
        ls_errs_bs = np.random.choice(ls_errs[bin_index,:], size=np.shape(ls_errs)[1])
        bglst_err_means_bs[bin_index, bs_index] = np.mean(bglst_errs_bs)
        gls_err_means_bs[bin_index, bs_index] = np.mean(gls_errs_bs)
        ls_err_means_bs[bin_index, bs_index] = np.mean(ls_errs_bs)

###############################################################################
# Plot experiment statistics

line1, = ax_stats_1.plot(bin_locs, bglst_err_means, 'r-', markerfacecolor='None', label = "BGLST")
line2, = ax_stats_1.plot(bin_locs, gls_err_means, 'b--', markerfacecolor='None', label = "GLS")
line3, = ax_stats_1.plot(bin_locs, ls_err_means, 'g-.', markerfacecolor='None', label = "LS")

ax_stats_1.fill_between(bin_locs, np.percentile(bglst_err_means_bs, 2.5, axis=1), np.percentile(bglst_err_means_bs, 97.5, axis=1), alpha=0.2, facecolor='lightsalmon', interpolate=True)
ax_stats_1.fill_between(bin_locs, np.percentile(gls_err_means_bs, 2.5, axis=1), np.percentile(gls_err_means_bs, 97.5, axis=1), alpha=0.2, facecolor='lightblue', interpolate=True)
ax_stats_1.fill_between(bin_locs, np.percentile(ls_err_means_bs, 2.5, axis=1), np.percentile(ls_err_means_bs, 97.5, axis=1), alpha=0.2, facecolor='lightgreen', interpolate=True)

ax_stats_1.set_position([0.12, 0.12, 0.85, 0.75])
ax_stats_1.legend(handles=[line1, line2, line3], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0., handletextpad=0., fontsize=panel_label_fs)#, columnspacing=10)

#ax_stats_1.legend(handles=[line1, line2, line3],
#        numpoints = 1,
#        scatterpoints=1,
#        loc='upper left', ncol=1,
#        fontsize=11, labelspacing=0.7)

#ax_stats_1.plot([min(means), max(means)], [0.5, 0.5], 'k:')
#ax_stats_2.plot([min(means), max(means)], [0.0, 0.0], 'k:')
fig_stats.savefig("offset_tests/offset_stats.pdf")
plt.close()


###############################################################################
    
        
        
        
            
