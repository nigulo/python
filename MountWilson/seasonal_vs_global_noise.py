# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 13:29:14 2017

@author: olspern1
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import BGLST
import bayes_lin_reg
import os
from scipy import stats
from astropy.stats import LombScargle
import mw_utils


real_sampling = False
down_sample_factor = 16

uniform_sampling = True


axis_label_fs = 15
panel_label_fs = 15

offset = 1979.3452
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
                real_dats.append((t, y, noise_var))

def get_local_noise_var(t, y, window_size=1.0):
    total_var = np.var(y)
    seasons = mw_utils.get_seasons(zip(t, y), window_size, False)
    noise_var = np.zeros(len(t))
    i = 0
    for season in seasons:
        if np.shape(season[:,1])[0] < 10:
            var = total_var # Is it good idea?
        else:
            var = np.var(season[:,1])
        season_len = np.shape(season)[0]
        for j in np.arange(i, i + season_len):
            noise_var[j] = var
        i += season_len
    assert(i == len(noise_var))
    return noise_var

def calc_BGLS(t, y, w, freq):
    tau = 0.5 * np.arctan(sum(w * np.sin(4 * np.pi * t * freq))/sum(w * np.cos(4 * np.pi * t * freq)))
    c = sum(w * np.cos(2.0 * np.pi * t * freq - tau))
    s = sum(w * np.sin(2.0 * np.pi * t * freq - tau))
    cc = sum(w * np.cos(2.0 * np.pi * t * freq - tau)**2)
    ss = sum(w * np.sin(2.0 * np.pi * t * freq - tau)**2)
    yc = sum(w * y * np.cos(2.0 * np.pi * t * freq - tau))
    ys = sum(w * y * np.sin(2.0 * np.pi * t * freq - tau))
    Y = sum(w * y)
    W = sum(w)

    assert(cc > 0)
    assert(ss > 0)
    
    K = (c**2/cc + s**2/ss - W)/2.0
    L = Y - c*yc/cc - s*ys/ss
    M = (yc**2/cc + ys**2/ss)/2.0
    log_prob = np.log(1.0 / np.sqrt(abs(K) * cc * ss)) + (M - L**2/4.0/K)
    return log_prob


    
#dat = np.loadtxt("cleaned/37394.cl.dat", usecols=(0,1), skiprows=1)

#t = dat[:,0]
#t /= 365.25
#t += offset
#y = dat[:,1]
#noise_var = mw_utils.get_seasonal_noise_var(t, y)

num_exp = 10000
true_freqs = np.zeros(num_exp)
bglst_freqs = np.zeros(num_exp)
ls_freqs = np.zeros(num_exp)

dats = []

for exp_no in np.arange(0, num_exp):
    print exp_no

    sig_var = 1.0
    initial_sn_ratio = np.random.uniform(0.1, 0.5)
    final_sn_ratio = np.random.uniform(2.0*initial_sn_ratio, 10.0*initial_sn_ratio)

    if real_sampling:
        t, y, noise_var = real_dats[np.random.choice(len(real_dats))]
        n = len(t)
        noise_var -= min(noise_var)
        noise_var /= max(noise_var)
        noise_var *= sig_var/(final_sn_ratio - initial_sn_ratio)
        noise_var += sig_var/final_sn_ratio
    else:
        time_range = 30.0
        n = 100
        if uniform_sampling:
            t = np.random.uniform(0, time_range, n)
        else:
            t = np.zeros(100)
            n = len(t)
            for i in np.arange(0, n):
                t[i] = np.random.uniform(time_range * float(i)/n, time_range)
        #noise_var = np.linspace(sig_var/initial_sn_ratio, sig_var/final_sn_ratio, n)
        noise_var = np.concatenate((np.ones(n/2)*initial_sn_ratio, np.ones(n/2)*final_sn_ratio))#np.linspace(sig_var/initial_sn_ratio, sig_var/final_sn_ratio, n)
    # Read real data to get some meaningful noise variance

    #sig_var = total_var - noise_var
    
    # Now generate synthetic data
    real_period = np.random.uniform(5.0, 15.0)
    y = np.sqrt(sig_var) * np.cos(2 * np.pi * t / real_period + np.random.uniform() * 2.0 * np.pi) + np.random.normal(np.zeros(n), np.sqrt(noise_var), n)#np.random.normal(0.0, np.sqrt(np.mean(noise_var)), n)
    #print "real freq:", 1.0/real_period
    true_freq = 1.0/real_period
    true_freqs[exp_no] = true_freq
    #t -= np.mean(t)
    
    #duration = max(t) - min(t)
    
    freq_start = 0.001
    freq_end = 2.0*true_freq
    freq_count = 1000
    
    #now produce empirical noise_var from sample variance
    noise_var = get_local_noise_var(t, y, 2.0)
    w = np.ones(n)/noise_var
    
    start = time.time()
    #slope, intercept, r_value, p_value, std_err = stats.linregress(t, y)
    #print "slope, intercept", slope, intercept
    bglst = BGLST.BGLST(t, y, w, 
                        w_A = 2.0/np.var(y), A_hat = 0.0,
                        w_B = 2.0/np.var(y), B_hat = 0.0,
                        #w_alpha = duration**2 / np.var(y), alpha_hat = slope, 
                        #w_beta = 1.0 / (np.var(y) + intercept**2), beta_hat = intercept)
                        w_alpha = 1e10, alpha_hat = 0.0, 
                        w_beta = 1e10, beta_hat = 0.0)
    
    (freqs, probs) = bglst.calc_all(freq_start, freq_end, freq_count)
    end = time.time()
    
    max_prob = max(probs)
    max_prob_index = np.argmax(probs)
    best_freq_bglst = freqs[max_prob_index]
    
    bglst_freqs[exp_no] = best_freq_bglst
    
    ###############################################################################
    # LS
    
    sigma = np.sqrt(noise_var)
    ls = LombScargle(t, y)#, sigma)
    power = ls.power(freqs, normalization='psd')#/np.var(y)
    
    max_power_ind = np.argmax(power)
    max_power = power[max_power_ind]
    best_freq_ls = freqs[max_power_ind]

    #print "LS: ", best_freq, max_power
    ls_freqs[exp_no] = best_freq_ls
    
    #print exp_no, initial_sn_ratio, final_sn_ratio, true_freq, bglst_freqs[exp_no], ls_freqs[exp_no]
    bglst_err = np.abs(best_freq_bglst-true_freq)
    ls_err = np.abs(best_freq_ls-true_freq)

    dats.append((t, y))
    
dats = np.asarray(dats)

bglst_errs = np.abs(bglst_freqs-true_freqs)/true_freqs
ls_errs = np.abs(ls_freqs-true_freqs)/true_freqs
err_ratios = bglst_errs/ls_errs

positive_indices = np.where(err_ratios < 0.95)[0]
negative_indices = np.where(err_ratios > 1.05)[0]
num_positive = float(len(positive_indices))
num_negative = float(len(negative_indices))

#print "----------Before 3sigma----------"
print num_positive + num_negative
print "bglst_errs < ls_errs", num_positive/(num_positive + num_negative)
print "BGLST_ERR_MEAN:", np.mean(bglst_errs)
print "BGLST_ERR_STD:", np.std(bglst_errs)
print "LS_ERR_MEAN:", np.mean(ls_errs)
print "LS_ERR_STD:", np.std(ls_errs)

#indices1 = np.where(bglst_errs < 3.0*np.std(bglst_errs))[0]
#bglst_errs = bglst_errs[indices1]
#ls_errs = ls_errs[indices1]
#dats = dats[indices1]
#true_freqs = true_freqs[indices1]
#indices2 = np.where(ls_errs < 3.0*np.std(ls_errs))[0]
#bglst_errs = bglst_errs[indices2]
#ls_errs = ls_errs[indices2]
#dats = dats[indices2]
#true_freqs = true_freqs[indices2]

#err_ratios = bglst_errs/ls_errs
#positive_indices = np.where(err_ratios < 0.95)[0]
#negative_indices = np.where(err_ratios > 1.05)[0]
#num_positive = float(len(positive_indices))
#num_negative = float(len(negative_indices))

#print "----------After 3sigma----------"
#print num_positive + num_negative
#print "bglst_errs < ls_errs", num_positive/(num_positive + num_negative)
#print "BGLST_ERR_MEAN:", np.mean(bglst_errs)
#print "BGLST_ERR_STD:", np.std(bglst_errs)
#print "LS_ERR_MEAN:", np.mean(ls_errs)
#print "LS_ERR_STD:", np.std(ls_errs)

diffs = ls_errs - bglst_errs
# Remove outliers
diffs = diffs[np.where(np.abs(diffs) < 5.0*np.std(diffs))[0]]

diff_sort_indices = np.argsort(diffs)
extreme_index1 = diff_sort_indices[-1]
extreme_index2 = diff_sort_indices[0]

diff = diffs[extreme_index1]
true_freq = true_freqs[extreme_index1]
(t, y) = dats[extreme_index1]
extreme_data_1 = (diff, t, y, true_freq)

diff = diffs[extreme_index2]
true_freq = true_freqs[extreme_index2]
(t, y) = dats[extreme_index2]
extreme_data_2 = (diff, t, y, true_freq)

#extreme_data_1 = (0, None, None, None)
#extreme_data_2 = (0, None, None, None)

#for i in np.arange(0, len(bglst_errs)):
#    bglst_err = bglst_errs[i]
#    ls_err = ls_errs[i]
#    true_freq = true_freqs[i]
#    (t, y) = dats[i]
#    if bglst_err < ls_err:
#        (max_diff, _, _, _) = extreme_data_1
#        if ls_err - bglst_err > max_diff:
#            extreme_data_1 = (ls_err - bglst_err, t, y, true_freq)
#    if ls_err < bglst_err:
#        (max_diff, _, _, _) = extreme_data_2
#        if bglst_err - ls_err > max_diff:
#            extreme_data_2 = (bglst_err - ls_err, t, y, true_freq)

fig, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False)
fig.set_size_inches(12, 7)

ax_a.text(0.95, 0.9,'(a)', horizontalalignment='center', transform=ax_a.transAxes, fontsize=panel_label_fs)
ax_b.text(0.95, 0.9,'(b)', horizontalalignment='center', transform=ax_b.transAxes, fontsize=panel_label_fs)
ax_c.text(0.95, 0.9,'(c)', horizontalalignment='center', transform=ax_c.transAxes, fontsize=panel_label_fs)
ax_d.text(0.95, 0.9,'(d)', horizontalalignment='center', transform=ax_d.transAxes, fontsize=panel_label_fs)

for (ax_a, ax_b), (_, t, y, true_freq) in [((ax_a, ax_c), extreme_data_1), ((ax_b, ax_d), extreme_data_2)]:
    freq_start = 0.001
    freq_end = 2.0*true_freq
    freq_count = 1000

    ax_a.scatter(t, y, marker='+', color ='k', lw=0.5)
    bglst = BGLST.BGLST(t, y, w, 
                        w_A = 2.0/np.var(y), A_hat = 0.0,
                        w_B = 2.0/np.var(y), B_hat = 0.0,
                        #w_alpha = duration**2 / np.var(y), alpha_hat = slope, 
                        #w_beta = 1.0 / (np.var(y) + intercept**2), beta_hat = intercept)
                        w_alpha = 1e10, alpha_hat = 0.0, 
                        w_beta = 1e10, beta_hat = 0.0)
    (freqs, probs) = bglst.calc_all(freq_start, freq_end, freq_count)

    max_prob = max(probs)
    min_prob = min(probs)
    max_prob_index = np.argmax(probs)
    best_freq_bglst = freqs[max_prob_index]

    tau, (A, B, alpha, beta), _, y_model, loglik = bglst.model(best_freq_bglst)
    
    t_model = np.linspace(min(t), max(t), 1000)
    y_model = np.cos(t_model * 2.0 * np.pi * best_freq_bglst - tau) * A  + np.sin(t_model * 2.0 * np.pi * best_freq_bglst - tau) * B + t_model * alpha + beta
    
    ax_a.plot(t_model, y_model, 'r-')
    
    norm_probs = (probs - min_prob) / (max_prob - min_prob)
    ax_b.plot(freqs, norm_probs, 'r-')
    ax_b.plot([best_freq_bglst, best_freq_bglst], [0, norm_probs[max_prob_index]], 'r-')
    
    ls = LombScargle(t, y)#, sigma)
    power = ls.power(freqs, normalization='psd')#/np.var(y)

    min_power = min(power)
    max_power_ind = np.argmax(power)
    max_power = power[max_power_ind]
    best_freq_ls = freqs[max_power_ind]


    y_model = ls.model(t_model, best_freq_ls)   
    ax_a.plot(t_model, y_model, 'b--')
    
    norm_powers = (power - min_power) / (max_power - min_power)
    
    ax_b.plot(freqs, norm_powers, 'b--')
    ax_b.plot([best_freq_ls, best_freq_ls], [0, norm_powers[max_power_ind]], 'b--')
    
    ax_b.plot([true_freq, true_freq], [0, norm_powers[max_power_ind]], 'k-.')
    
    ax_a.set_xlabel(r'$t$', fontsize=axis_label_fs)#,fontsize=20)
    ax_a.set_ylabel(r'y', fontsize=axis_label_fs)#,fontsize=20)
    ax_a.set_xlim([min(t), max(t)])
    
    ax_b.set_xlabel(r'$f$', fontsize=axis_label_fs)#,fontsize=20)
    ax_b.set_ylabel(r'Power', fontsize=axis_label_fs)#,fontsize=20)
    ax_b.set_xlim([0.001, 2.0*true_freq])

#print exp_no, initial_sn_ratio, final_sn_ratio, true_freq, bglst_freqs[exp_no], ls_freqs[exp_no]
fig.savefig("seasonal_vs_global_noise.eps")

plt.close()

#n, bins, patches = plt.hist([bglst_errs[positive_indices], ls_errs[negative_indices]], bins=50, normed=True, histtype='step', color=['red', 'blue'], alpha=0.5)
n, bins, patches = plt.hist(diffs, bins=50, normed=True)
plt.xlabel(r'$\left< \Delta f/f \right>$', fontsize=axis_label_fs)
plt.savefig("bglst_ls_hist.eps")
plt.close()
