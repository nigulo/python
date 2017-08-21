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
import sys
from filelock import FileLock
import mw_utils
from scipy import stats

offset = 1979.3452
down_sample_factor = 8

group_no = 0
if len(sys.argv) > 1:
    group_no = int(sys.argv[1])

num_experiments = 2000

def calc_cov(t, f, sig_var, trend_var, c):
    k = np.zeros((len(t), len(t)))
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            k[i, j] = sig_var*(np.cos(2 * np.pi*f*(t[i] - t[j]))) + trend_var * (t[i] - c) * (t[j] - c)
            k[j, i] = k[i, j]
    return k


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

for experiment_index in np.arange(0, num_experiments):
    print experiment_index
    duration = 0
    while duration < 30:
        t = select_dataset()
        min_t = min(t)
        duration = max(t) - min_t

        
    n = len(t)
    
    var = 1.0
    sig_var = np.random.uniform(0.2, 0.8)
    noise_var = np.ones(n) * (var - sig_var)
    trend_var = np.random.uniform(0.0, 1.0) * var / duration
    mean = 0.5
    
    p = np.random.uniform(2.0, duration/1.5)
    f = 1.0/p
    mean = 0.5

    k = calc_cov(t, f, sig_var, trend_var, 0.0) + np.diag(noise_var)
    l = la.cholesky(k)
    s = np.random.normal(0, 1, n)
    
    y = np.repeat(mean, n) + np.dot(l, s)
    y += mean
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
    fig.set_size_inches(10, 8)
    
    ax1.scatter(t, y, s=1)
    min_y = min(y)
    max_y = max(y)
    for i in np.arange(1, duration*f):
        ax1.plot((min_t + p * i, min_t + p * i), (min_y, max_y), 'k--')    
    ax1.set_xlim([min_t, max(t)])
    
    #data = np.column_stack((t, y))
    #print data
    #np.savetxt("cyclic.txt", data, fmt='%f')
    noise_var_prop = mw_utils.get_seasonal_noise_var(t, y)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(t, y)
    print "slope, intercept", slope, intercept
    fit_trend = t * slope + intercept
    y_detrended = y - fit_trend
    
    freqs = np.linspace(0.001, 0.5, 1000)
    power = LombScargle(t, y_detrended, np.sqrt(noise_var_prop)).power(freqs, normalization='psd')
    
    ls_local_maxima_inds = mw_utils.find_local_maxima(power)
    f_opt_ls_ind = np.argmax(power[ls_local_maxima_inds])
    f_opt_ls = freqs[ls_local_maxima_inds][f_opt_ls_ind]
    #f_opt_ls = freqs[np.argmax(power[1:])+1]
    ax2.plot(freqs, power)
    ax2.plot([f_opt_ls, f_opt_ls], [min(power), power[ls_local_maxima_inds][f_opt_ls_ind]], 'k--')
    
    
    w = np.ones(n) / noise_var_prop

    bglst = BGLST.BGLST(t, y, w, 
                    w_A = 2.0/np.var(y), A_hat = 0.0,
                    w_B = 2.0/np.var(y), B_hat = 0.0,
                    w_alpha = duration**2 / np.var(y), alpha_hat = slope, 
                    w_beta = 1.0 / (np.var(y) + intercept**2), beta_hat = intercept)
    
    
    (freqs, probs) = bglst.calc_all(min(freqs), max(freqs), len(freqs))
    bglst_local_maxima_inds = mw_utils.find_local_maxima(probs)
    f_opt_bglst_ind = np.argmax(probs[bglst_local_maxima_inds])
    f_opt_bglst = freqs[bglst_local_maxima_inds][f_opt_bglst_ind]
    #f_opt_bglst = freqs[np.argmax(probs[1:])+1]
    ax3.plot(freqs, probs)
    ax3.plot([f_opt_bglst, f_opt_bglst], [min(probs), probs[bglst_local_maxima_inds][f_opt_bglst_ind]], 'k--')
    
    seasonal_means = mw_utils.get_seasonal_means(t, y)
    #op = model.optimizing(data=dict(x=t,N=n,y=y,noise_var=noise_var, var_y=np.var(y), harmonicity=1.0))
    #freq_gp_opt = op['freq'];
    
    index = group_no * num_experiments + experiment_index
    with FileLock("GPRLock"):
        with open("comp/results.txt", "a") as output:
            output.write("%s %s %s %s %s %s %s\n" % (index, f, f_opt_ls, f_opt_bglst, sig_var, trend_var, duration))  

    fig.savefig("comp/cyclic_" + str(index) + ".png")

    plt.close()
    
