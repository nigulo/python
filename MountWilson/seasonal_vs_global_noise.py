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
dats = []

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
                dats.append((t, y, noise_var))

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

num_exp = 2000
true_freqs = np.zeros(num_exp)
bglst_freqs = np.zeros(num_exp)
ls_freqs = np.zeros(num_exp)
for exp_no in np.arange(0, num_exp):
    print exp_no
    
    sig_var = 1.0
    initial_sn_ratio = 2.0
    final_sn_ratio = 0.5

    if real_sampling:
        t, y, noise_var = dats[np.random.choice(len(dats))]
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
        noise_var = np.linspace(sig_var/initial_sn_ratio, sig_var/final_sn_ratio, n)
    # Read real data to get some meaningful noise variance

    #sig_var = total_var - noise_var
    
    # Now generate synthetic data
    real_period = np.random.uniform(5.0, 15.0)
    y = np.sqrt(sig_var) * np.cos(2 * np.pi * t / real_period + np.random.uniform() * 2.0 * np.pi) + np.random.normal(np.zeros(n), np.sqrt(noise_var), n)#np.random.normal(0.0, np.sqrt(np.mean(noise_var)), n)
    #print "real freq:", 1.0/real_period
    true_freqs[exp_no] = 1.0/real_period
    #t -= np.mean(t)
    
    duration = max(t) - min(t)
    
    freq_start = 0.001
    freq_end = 0.5
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
    
    #probs = np.zeros(freq_count)
    #bglst = BGLST.BGLST(t, y, w)
    #start = time.time()
    #i = 0
    #for f in freqs:
    #    probs[i] = bglst.calc(f)
    #    i += 1
    #    #probs.append(calc_BGLS(t, y, w, f))
    #end = time.time()
    #print(end - start)
    
    
    #print probs - probs1
    
    max_prob = max(probs)
    max_prob_index = np.argmax(probs)
    best_freq = freqs[max_prob_index]
    
    if exp_no == 0:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=False)
        fig.set_size_inches(6, 7)
        
        ax1.text(0.95, 0.9,'(a)', horizontalalignment='center', transform=ax1.transAxes, fontsize=panel_label_fs)
        ax2.text(0.95, 0.9,'(b)', horizontalalignment='center', transform=ax2.transAxes, fontsize=panel_label_fs)
    
        ax1.scatter(t, y, marker='+', color ='k', lw=0.5)
    tau, (A, B, alpha, beta), _, y_model_1, loglik = bglst.model(best_freq)

    t_model = np.linspace(min(t), max(t), 1000)
    y_model = np.cos(t_model * 2.0 * np.pi * best_freq - tau) * A  + np.sin(t_model * 2.0 * np.pi * best_freq - tau) * B + t_model * alpha + beta

    if exp_no == 0:
        ax1.plot(t_model, y_model, 'r-')
        ax1.plot(t_model, t_model * alpha + beta, 'r-')
    
        min_prob = min(probs)
        norm_probs = (probs - min_prob) / (max_prob - min_prob)
        ax2.plot(freqs, norm_probs, 'r-')
        ax2.plot([best_freq, best_freq], [0, norm_probs[max_prob_index]], 'r-')
    
    #bglst_m = BGLST.BGLST(t, y_model_1, np.ones(n)/np.var(y),
    #                    w_A = 2.0/np.var(y), A_hat = 0.0,
    #                    w_B = 2.0/np.var(y), B_hat = 0.0,
    #                    w_alpha = duration**2 / np.var(y), alpha_hat = slope, 
    #                    w_beta = 1.0 / (np.var(y) + intercept**2), beta_hat = intercept)
    
    #(freqs, probs_m) = bglst_m.calc_all(freq_start, freq_end, freq_count)
    #max_prob_m = max(probs_m)
    #max_prob_index_m = np.argmax(probs_m)
    #best_freq_m = freqs[max_prob_index_m]
    #min_prob_m = min(probs_m)
    #norm_probs_m = (probs_m- min_prob_m) / (max_prob_m - min_prob_m)
    #max_prob_m = max(probs_m)
    #max_prob_index_m = np.argmax(probs_m)
    #best_freq_m = freqs[max_prob_index_m]
    #min_prob_m = min(probs_m)
    #norm_probs_m = (probs_m- min_prob_m) / (max_prob_m - min_prob_m)
    
    
    #print "BGLST: ", best_freq, max_prob
    bglst_freqs[exp_no] = best_freq
    
    ###############################################################################
    # LS
    
    sigma = np.sqrt(noise_var)
    ls = LombScargle(t, y)#, sigma)
    power = ls.power(freqs, normalization='psd')#/np.var(y)
    
    max_power_ind = np.argmax(power)
    max_power = power[max_power_ind]
    best_freq = freqs[max_power_ind]
    y_model = ls.model(t_model, best_freq)
    #print "LS: ", best_freq, max_power
    ls_freqs[exp_no] = best_freq
    
    if exp_no == 0:    
        ax1.plot(t_model, y_model, 'g-.')
        
        min_power = min(power)
        norm_powers = (power - min_power) / (max_power - min_power)
        
        ax2.plot(freqs, norm_powers, 'g-.')
        ax2.plot([best_freq, best_freq], [0, norm_powers[max_power_ind]], 'g-.')
    
    ###############################################################################
    # LS detrended
    
    #y_fit = t * slope + intercept
    #y -= y_fit
    #ls = LombScargle(t, y, sigma)
    #power = ls.power(freqs, normalization='psd')#/np.var(y)
    
    #max_power_ind = np.argmax(power)
    #max_power = power[max_power_ind]
    #best_freq = freqs[max_power_ind]
    #y_model = ls.model(t_model, best_freq)
    #print "LS detrended: ", best_freq, max_power

    if exp_no == 0:    
        #ax1.plot(t_model, y_model+t_model * slope + intercept, 'b--')
        #ax1.plot(t_model, t_model * slope + intercept, 'b--')
        
        #min_power = min(power)
        #norm_powers = (power - min_power) / (max_power - min_power)
        
        #ax2.plot(freqs, norm_powers, 'b--')
        #ax2.plot([best_freq, best_freq], [0, norm_powers[max_power_ind]], 'b--')
        
        ax1.set_xlabel(r'$t$ [yr]', fontsize=axis_label_fs)#,fontsize=20)
        ax1.set_ylabel(r'S-index', fontsize=axis_label_fs)#,fontsize=20)
        ax1.set_xlim([min(t), max(t)])
        
        ax2.set_xlabel(r'$f$', fontsize=axis_label_fs)#,fontsize=20)
        ax2.set_ylabel(r'Power', fontsize=axis_label_fs)#,fontsize=20)
        ax2.set_xlim([0.001, 0.5])
        
        fig.savefig("seasonal_vs_global_noise.eps")

bglst_errs = np.abs(bglst_freqs-true_freqs)/true_freqs
ls_errs = np.abs(ls_freqs-true_freqs)/true_freqs
bglst_errs = bglst_errs[np.where(bglst_errs < 3.0*np.std(bglst_errs))[0]]
ls_errs = ls_errs[np.where(bglst_errs < 3.0*np.std(bglst_errs))[0]]
bglst_errs = bglst_errs[np.where(ls_errs < 3.0*np.std(ls_errs))[0]]
ls_errs = ls_errs[np.where(ls_errs < 3.0*np.std(ls_errs))[0]]

not_equal = len(np.where(bglst_errs != ls_errs)[0])
print "bglst_errs != ls_errs", not_equal
print "bglst_errs < ls_errs", float(len(np.where(bglst_errs < ls_errs)[0]))/not_equal
print "BGLST_ERR_MEAN:", np.mean(bglst_errs)
print "BGLST_ERR_STD:", np.std(bglst_errs)
print "LS_ERR_MEAN:", np.mean(ls_errs)
print "LS_ERR_STD:", np.std(ls_errs)

plt.close()

n, bins, patches = plt.hist([bglst_errs, ls_errs], bins=50, normed=True, histtype='step', color=['red', 'blue'], alpha=0.5)
plt.xlabel(r'$\left< \Delta f/f \right>$', fontsize=axis_label_fs)
plt.savefig("bglst_ls_hist.eps")
plt.close()
