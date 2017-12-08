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
from scipy import stats
from astropy.stats import LombScargle
import mw_utils

axis_label_fs = 15
panel_label_fs = 15

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

offset = 1979.3452

    
#star = "37394"
star = "3651"
dat = np.loadtxt("cleaned/"+ star +".cl.dat", usecols=(0,1), skiprows=1)

y = dat[:,1]
t = dat[:,0]
t /= 365.25
t += offset
#t -= np.mean(t)

n = len(t)
duration = max(t) - min(t)

freq_start = 0.001
freq_end = 0.5
freq_count = 1000

noise_var = mw_utils.get_seasonal_noise_var(t, y)

start = time.time()

fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(nrows=2, ncols=2, sharex=False)
fig.set_size_inches(12, 7)

ax11.text(0.95, 0.9,'(a)', horizontalalignment='center', transform=ax11.transAxes, fontsize=panel_label_fs)
ax12.text(0.95, 0.9,'(b)', horizontalalignment='center', transform=ax12.transAxes, fontsize=panel_label_fs)
ax21.text(0.95, 0.9,'(c)', horizontalalignment='center', transform=ax21.transAxes, fontsize=panel_label_fs)
ax22.text(0.95, 0.9,'(d)', horizontalalignment='center', transform=ax22.transAxes, fontsize=panel_label_fs)

ax11.set_ylabel(r'S-index', fontsize=axis_label_fs)#,fontsize=20)
ax21.set_ylabel(r'Power', fontsize=axis_label_fs)#,fontsize=20)

for const_noise_var in [True, False]:
    if const_noise_var:
        ax1 = ax11
        ax2 = ax21
        w = np.ones(n)/np.var(y)
    else:
        ax1 = ax12
        ax2 = ax22
        w = np.ones(n)/noise_var

    slope, intercept, r_value, p_value, std_err = stats.linregress(t, y)
    print "slope, intercept", slope, intercept
    bglst = BGLST.BGLST(t, y, w, 
                        w_A = 2.0/np.var(y), A_hat = 0.0,
                        w_B = 2.0/np.var(y), B_hat = 0.0,
                        w_alpha = duration**2 / np.var(y), alpha_hat = slope, 
                        w_beta = 1.0 / (np.var(y) + intercept**2), beta_hat = intercept)
    
    (freqs, probs) = bglst.calc_all(freq_start, freq_end, freq_count)
    end = time.time()
    print(end - start)
    
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


    ax1.scatter(t, y, marker='+', color ='k', lw=0.5)
    tau, (A, B, alpha, beta), _, y_model_1, loglik = bglst.model(best_freq)
    bic = 2 * loglik - np.log(n) * 5
    print A, B, alpha, beta
    t_model = np.linspace(min(t), max(t), 1000)
    y_model = np.cos(t_model * 2.0 * np.pi * best_freq - tau) * A  + np.sin(t_model * 2.0 * np.pi * best_freq - tau) * B + t_model * alpha + beta
    line1, = ax1.plot(t_model, y_model, 'r-', label = "BGLST")
    ax1.plot(t_model, t_model * alpha + beta, 'r-')
    
    min_prob = min(probs)
    norm_probs = (probs - min_prob) / (max_prob - min_prob)
    ax2.plot(freqs, norm_probs, 'r-')
    ax2.plot([best_freq, best_freq], [0, norm_probs[max_prob_index]], 'r-')
    
    bglst_m = BGLST.BGLST(t, y_model_1, w,
                        w_A = 2.0/np.var(y), A_hat = 0.0,
                        w_B = 2.0/np.var(y), B_hat = 0.0,
                        w_alpha = duration**2 / np.var(y), alpha_hat = slope, 
                        w_beta = 1.0 / (np.var(y) + intercept**2), beta_hat = intercept)
    
    (freqs, probs_m) = bglst_m.calc_all(freq_start, freq_end, freq_count)
    max_prob_m = max(probs_m)
    max_prob_index_m = np.argmax(probs_m)
    best_freq_m = freqs[max_prob_index_m]
    min_prob_m = min(probs_m)
    norm_probs_m = (probs_m- min_prob_m) / (max_prob_m - min_prob_m)
    #ax2.plot(freqs, norm_probs_m, 'g-')
    max_prob_m = max(probs_m)
    max_prob_index_m = np.argmax(probs_m)
    best_freq_m = freqs[max_prob_index_m]
    min_prob_m = min(probs_m)
    norm_probs_m = (probs_m- min_prob_m) / (max_prob_m - min_prob_m)
    
    
    print "BGLST: ", best_freq, max_prob
    
    _, _, _, loglik_null = bayes_lin_reg.bayes_lin_reg(t, y, w)
    bic_null = 2 * loglik_null - np.log(n) * 2
    
    print bic - bic_null
    
    
    ###############################################################################
    # LS
    
    # Switch comments to enable seasonal noise vs. constant noise
    if const_noise_var:
        sigma = np.sqrt(np.var(y))
    else:
        sigma = np.sqrt(noise_var)
        
    ls = LombScargle(t, y, sigma)
    power = ls.power(freqs, normalization='psd')#/np.var(y)
    
    max_power_ind = np.argmax(power)
    max_power = power[max_power_ind]
    best_freq = freqs[max_power_ind]
    y_model = ls.model(t_model, best_freq)
    line3, = ax1.plot(t_model, y_model, 'g-.', label = "GLS")
    print "LS: ", best_freq, max_power
    
    min_power = min(power)
    norm_powers = (power - min_power) / (max_power - min_power)
    
    ax2.plot(freqs, norm_powers, 'g-.')
    ax2.plot([best_freq, best_freq], [0, norm_powers[max_power_ind]], 'g-.')
    
    ###############################################################################
    # LS detrended
    
    y_fit = t * slope + intercept
    y_detrended = y - y_fit
    ls = LombScargle(t, y_detrended, sigma)
    power = ls.power(freqs, normalization='psd')#/np.var(y)
    
    max_power_ind = np.argmax(power)
    max_power = power[max_power_ind]
    best_freq = freqs[max_power_ind]
    y_model = ls.model(t_model, best_freq)
    line2, = ax1.plot(t_model, y_model+t_model * slope + intercept, 'b--', label = 'GLS-T')
    ax1.plot(t_model, t_model * slope + intercept, 'b--')
    print "LS detrended: ", best_freq, max_power
    
    min_power = min(power)
    norm_powers = (power - min_power) / (max_power - min_power)
    
    ax2.plot(freqs, norm_powers, 'b--')
    ax2.plot([best_freq, best_freq], [0, norm_powers[max_power_ind]], 'b--')
    
    ax1.set_xlabel(r'$t$ [yr]', fontsize=axis_label_fs)#,fontsize=20)
    #ax1.set_ylabel(r'S-index', fontsize=axis_label_fs)#,fontsize=20)
    ax1.set_xlim([min(t), max(t)])
    
    ax2.set_xlabel(r'$f$', fontsize=axis_label_fs)#,fontsize=20)
    #ax2.set_ylabel(r'Power', fontsize=axis_label_fs)#,fontsize=20)
    ax2.set_xlim([0.001, 0.5])

ax12.legend(handles=[line1, line2, line3], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0., handletextpad=0.)#, columnspacing=10)

fig.savefig("model_comp_hd" + star +".eps")
