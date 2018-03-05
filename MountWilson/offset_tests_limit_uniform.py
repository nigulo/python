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

axis_label_fs = 15
panel_label_fs = 15

max_count_per_bin = 500
ns = np.array([10, 20, 50, 100, 200])
num_bins = len(ns)

fig_stats, ax_stats_1 = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=False)
fig_stats.set_size_inches(6, 4)
#ax_stats_1.text(0.05, 0.9,'(b)', horizontalalignment='center', transform=ax_stats_1.transAxes, fontsize=panel_label_fs)
ax_stats_1.set_xlabel(r'$n$', fontsize=axis_label_fs, labelpad = -1)#,fontsize=20)
#ax_stats_1.set_xlim([0, 1])
#ax_stats_1.set_ylim([0, 1])
ax_stats_1.set_ylabel(r'$S_1$', fontsize=axis_label_fs)#,fontsize=20)

#fig_stats.tight_layout()

bglst_err_means = np.zeros(num_bins)
gls_err_means = np.zeros(num_bins)

bglst_err_means_bs = np.zeros((num_bins, num_bs))
gls_err_means_bs = np.zeros((num_bins, num_bs))

counts = np.zeros(num_bins, dtype=int)

if os.path.exists('offset_tests/offset_limit_uniform_exp.pkl'):
    (true_freqs, bglst_freqs, gls_freqs, ls_freqs) = pickle.load(open('offset_tests/offset_limit_uniform_exp.pkl', 'rb'))
else:

    true_freqs = np.zeros((num_bins, max_count_per_bin))
    bglst_freqs = np.zeros((num_bins, max_count_per_bin))
    gls_freqs = np.zeros((num_bins, max_count_per_bin))
    ls_freqs = np.zeros((num_bins, max_count_per_bin))
    
    exp_no = 0
    for n in ns:
        for rep_no in np.arange(0, max_count_per_bin):
            time_range = 30.0
            
            p = np.random.uniform(time_range*2/3)
            f = 1.0/p
            
            t_orig = np.random.uniform(0, time_range, n)
            #t_orig = np.linspace(0, time_range, n) + np.random.rand(n)*0.2
            t = t_orig

            t = np.sort(t)
            min_t = min(t)
            max_t = max(t)
            duration = max_t - min_t
                        
            var = 1.0
            sig_var = np.random.uniform(0.99, 0.99)
            noise_var = np.ones(n) * (var - sig_var)

            mean = np.random.uniform(-5.0, 5.0)
        
            k = calc_cov(t, f, sig_var, 0.0, 0.0) + np.diag(noise_var)
            l = la.cholesky(k)
            s = np.random.normal(0, 1, n)
            
            y = np.repeat(mean, n) + np.dot(l, s)
            emp_mean_err = abs(np.mean(y) - mean)

            true_freqs[exp_no, rep_no] = f
            #y += mean
            
            #data = np.column_stack((t, y))
            #print data
            #np.savetxt("cyclic.txt", data, fmt='%f')
            #noise_var_prop = mw_utils.get_seasonal_noise_var(t, y)
            min_freq = 0.001
            max_freq = 2.0*f
            freqs = np.linspace(min_freq, max_freq, 1000)
       
            gls = LombScargle(t, y, np.sqrt(noise_var))
            power_gls = gls.power(freqs, normalization='psd')
            
            f_opt_gls_ind = np.argmax(power_gls)
            f_opt_gls = freqs[f_opt_gls_ind]
            gls_freqs[exp_no, rep_no] = f_opt_gls
            
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
            bglst_freqs[exp_no, rep_no] = f_opt_bglst
            
            min_prob_bglst = min(probs)
            max_prob_bglst = max(probs)
            norm_probs_bglst = (probs - min_prob_bglst) / (max_prob_bglst - min_prob_bglst)
            
            print f, f_opt_gls, f_opt_bglst
            
        exp_no += 1
        #print counts

    with open('offset_tests/offset_limit_uniform_exp.pkl', 'wb') as f:
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
        bglst_err_means_bs[bin_index, bs_index] = np.mean(bglst_errs_bs)
        gls_err_means_bs[bin_index, bs_index] = np.mean(gls_errs_bs)

###############################################################################
# Plot experiment statistics

line1, = ax_stats_1.plot(ns, bglst_err_means, 'r-', markerfacecolor='None', label = "BGLST")
line2, = ax_stats_1.plot(ns, gls_err_means, 'b--', markerfacecolor='None', label = "GLS")

ax_stats_1.fill_between(ns, np.percentile(bglst_err_means_bs, 2.5, axis=1), np.percentile(bglst_err_means_bs, 97.5, axis=1), alpha=0.2, facecolor='lightsalmon', interpolate=True)
ax_stats_1.fill_between(ns, np.percentile(gls_err_means_bs, 2.5, axis=1), np.percentile(gls_err_means_bs, 97.5, axis=1), alpha=0.2, facecolor='lightblue', interpolate=True)

#ax_stats_1.set_position([0.12, 0.12, 0.85, 0.75])
#ax_stats_1.legend(handles=[line1, line2], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0., handletextpad=0., fontsize=panel_label_fs)#, columnspacing=10)

ax_stats_1.legend(handles=[line1, line2],
        numpoints = 1,
        scatterpoints=1,
        loc='upper right', ncol=1,
        fontsize=11, labelspacing=0.7)

ax_stats_1.set_xlim([min(ns), max(ns)])
fig_stats.savefig("offset_tests/offset_limit_uniform_stats.pdf")
plt.close()


###############################################################################
    
        
        
        
            
