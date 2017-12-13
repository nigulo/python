# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 11:48:48 2017

@author: olspern1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
import random
from matplotlib import colors as mcolors
import matplotlib.markers as markers
from scipy.stats import gaussian_kde

import os
import os.path
import mw_utils
from bayes_lin_reg import bayes_lin_reg

#input_path = "downsampling/results"
input_path = "BGLST_input"
    
output_path = "no_cycles"
offset = 1979.3452
num_r_hk_bins = 1000
num_bootstrap = 1000
kde_cov_start = 0.01
kde_cov_end = 1.0
kde_num_covs = 100

rot_periods = mw_utils.load_rot_periods()

min_bic, max_bic, all_cycles = mw_utils.read_bglst_cycles("BGLST_BIC_6/results.txt")

all_cycles_gp_p = mw_utils.read_gp_cycles("GP_periodic/results_combined.txt")
all_cycles_gp_qp = mw_utils.read_gp_cycles("GP_quasiperiodic/results_combined.txt")

r_hks = mw_utils.load_r_hk()
min_r_hk = None
max_r_hk = None
for star in r_hks.keys():
    r_hk = r_hks[star]
    if min_r_hk is None or r_hk < min_r_hk:
        min_r_hk = r_hk
    if max_r_hk is None or r_hk > max_r_hk:
        max_r_hk = r_hk
print min_r_hk, max_r_hk
r_hk_bins_values = np.linspace(min_r_hk, max_r_hk, num=num_r_hk_bins)
#r_hk_bin_counts_a = np.zeros(num_r_hk_bins)
#r_hk_bin_counts_na = np.zeros(num_r_hk_bins)

num_stars = len(r_hks.keys())
print num_stars

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=False)
fig.set_size_inches(6, 12)
ax1.set_ylabel(r'$\sigma^2/\overline{\sigma}^2_{\rm s}$', fontsize=15)
ax2.set_ylabel(r'$p_{\rm cyc}/p$', fontsize=15)
ax2.set_xlabel(r'$\log \langle R^\prime_{\rm HK}\rangle$', fontsize=15)
ax1.text(0.95, 0.9,'(a)', horizontalalignment='center', transform=ax1.transAxes, fontsize=15)
ax2.text(0.95, 0.9,'(b)', horizontalalignment='center', transform=ax2.transAxes, fontsize=15)

r_hks_a = []
r_hks_na = []
r_hks_na_t = []
r_hks_na_wot = []
var_ratios_a = []
var_ratios_na = []
var_ratios_na_t = []
var_ratios_na_wot = []


def constant_model(y, w, y_test, w_test):
    W = sum(w)
    wy_arr = w * y

    Y = sum(wy_arr)

    sigma_beta = 1.0 / W
    mu_beta = Y * sigma_beta

    norm_term = sum(np.log(np.sqrt(w)) - np.log(np.sqrt(2.0*np.pi)))

    y_model = mu_beta
    loglik = norm_term - 0.5 * sum(w_test * (y_test - y_model)**2)

    return (mu_beta, sigma_beta, y_model, loglik)


for root, dirs, files in os.walk(input_path):
    for file in files:
        if file[-4:] == ".dat":
            star = file[:-4]
            star = star.upper()
            if (star[-3:] == '.CL'):
                star = star[0:-3]
            if (star[0:2] == 'HD'):
                star = star[2:]
            while star[0] == '0': # remove leading zeros
                star = star[1:]
            rot_period = 0
            if (rot_periods.has_key(star)):
                rot_period = rot_periods[star]
            #print star + " period is " + str(rot_period)
            data = np.loadtxt(input_path+"/"+file, usecols=(0,1), skiprows=0)
            print "Analysing star without cycles: " + star
            t_orig = data[:,0]
            y_orig = data[:,1]
            indices = np.argsort(t_orig)
            t_orig = t_orig[indices]            
            y_orig = y_orig[indices]            
            t = t_orig/365.25 + offset
            y = y_orig
            time_range = max(t) - min(t)
            mean_seasonal_var = np.mean(mw_utils.get_seasonal_noise_var(t, y, False))
            total_var = np.var(y)
            #print total_var/mean_seasonal_var
            if (r_hks.has_key(star)):
                r_hk = r_hks[star]
                if all_cycles.has_key(star) or all_cycles_gp_p.has_key(star) or all_cycles_gp_qp.has_key(star):
                    r_hks_a.append(r_hk)
                    var_ratios_a.append(total_var/mean_seasonal_var)
                else:
                    r_hks_na.append(r_hk)
                    var_ratios_na.append(total_var/mean_seasonal_var)
                    ############################
                    # Fit the model to get trend
                    ############################
                    data = np.loadtxt(input_path+"/"+file, usecols=(0,1), skiprows=1)
                    t_orig = data[:,0]
                    y_orig = data[:,1]
                    indices = np.argsort(t_orig)
                    t_orig = t_orig[indices]            
                    y_orig = y_orig[indices]            
                    t = t_orig/365.25 + offset
                    y = y_orig
                    time_range = max(t) - min(t)
                    noise_var = mw_utils.get_seasonal_noise_var(t, y)
                    w = np.ones(len(t))/noise_var
                    seasonal_means = mw_utils.get_seasonal_means(t, y)
                    seasonal_noise_var = mw_utils.get_seasonal_noise_var(t, y, False)
                    seasonal_weights = np.ones(len(seasonal_noise_var))/seasonal_noise_var
                    _, _, _, loglik_seasons = bayes_lin_reg(t, y, w, seasonal_means[:,0], seasonal_means[:,1], seasonal_weights)
                    _, _, _, loglik_seasons_null = constant_model(y, w, seasonal_means[:,1], seasonal_weights)
                    log_n = np.log(np.shape(seasonal_means)[0])
                    bic = log_n * 2 - 2.0*loglik_seasons
                    bic_null = log_n  - 2.0*loglik_seasons_null
                    
                    #print bic_null, bic
                    delta_bic = bic_null - bic
                    if delta_bic >= 6.0:
                        # Significant trend
                        r_hks_na_t.append(r_hk)
                        var_ratios_na_t.append(total_var/mean_seasonal_var)
                    else:
                        r_hks_na_wot.append(r_hk)
                        var_ratios_na_wot.append(total_var/mean_seasonal_var)
                    ############################

print "Num cyclic:", len(r_hks_a)
print "Num noncyclic with trend:", len(r_hks_na_t)
print "Num noncyclic without trend:", len(r_hks_na_wot)
ax1.scatter(r_hks_na_t, var_ratios_na_t, marker=markers.MarkerStyle("x", fillstyle=None), lw=1.5, facecolors="green", color="green", s=50, edgecolors="green")
ax1.scatter(r_hks_na_wot, var_ratios_na_wot, marker=markers.MarkerStyle('d', fillstyle=None), lw=1.5, facecolors="none", color="blue", s=50, edgecolors="blue")
ax1.scatter(r_hks_a, var_ratios_a, marker=markers.MarkerStyle("+", fillstyle=None), lw=1.5, facecolors="red", color="red", s=50, edgecolors="red")
slope, intercept, r_value, p_value, std_err = stats.linregress(r_hks_na_wot, var_ratios_na_wot)
ax1.plot(r_hk_bins_values, r_hk_bins_values*slope + intercept, "b--")
#slope, intercept, r_value, p_value, std_err = stats.linregress(r_hks_a, var_ratios_a)
#ax1.plot(r_hk_bins_values, r_hk_bins_values*slope + intercept, "r-")


def calc_kde(data):
    max_lik = None
    opt_kde = None
    num_sets = 10
    set_len = len(data) / num_sets
    for kde_cov in np.linspace(kde_cov_start, kde_cov_end, num=kde_num_covs):
        log_lik = 0
        for i in np.arange(0, num_sets):
            train_data = np.concatenate((data[:i*set_len], data[(i+1)*set_len:]))
            if i == num_sets - 1:
                valid_data = data[i*set_len:]
            else:
                valid_data = data[i*set_len:(i+1)*set_len]
            density = gaussian_kde(train_data, bw_method = kde_cov)
            d = density(valid_data)
            log_lik += np.sum(np.log(d))
        if max_lik is None or log_lik > max_lik:
            max_lik = log_lik
            opt_kde = kde_cov
    print opt_kde
    return gaussian_kde(data, bw_method = opt_kde), opt_kde
        
density_a, cov_a = calc_kde(r_hks_a)
d_a = density_a(r_hk_bins_values)

density_na, cov_na = calc_kde(r_hks_na)
d_na = density_na(r_hk_bins_values)

density_all, cov_all = calc_kde(r_hks_a + r_hks_na)
d_all = density_all(r_hk_bins_values)

density_na_t, cov_na_t = calc_kde(r_hks_na_t)
d_na_t = density_na_t(r_hk_bins_values)

density_a_and_na_t, cov_a_and_na_t = calc_kde(r_hks_a + r_hks_na_t)
d_a_and_na_t = density_a_and_na_t(r_hk_bins_values)


#ax2.plot(r_hk_bins_values, r_hk_bin_counts_na/(r_hk_bin_counts_a+r_hk_bin_counts_na), "k-")
#ax2.plot(r_hk_bins_values, d_a/(d_a+d_na), "k-")
ax2.plot(r_hk_bins_values, d_all, "k--")
ax2.plot(r_hk_bins_values, d_na_t, "g-.")
ax2.plot(r_hk_bins_values, d_a_and_na_t, ":", color="saddlebrown")

r_hks_a = np.asarray(r_hks_a)
r_hks_na = np.asarray(r_hks_na)

rel_d_a_bs = []

def get_conf_ints(data, percent=0.95, num_bins=1000):
    n = np.shape(data)[0]
    min_value = np.min(data)
    max_value = np.max(data)
    lower = None
    upper = None
    #print n, "----------------------------"
    for value in np.linspace(min_value, max_value, num=num_bins):
        #print float(np.shape(np.where(data > value)[0])[0])/n, float(np.shape(np.where(data < value)[0])[0])/n
        if lower is None and float(np.shape(np.where(data < value)[0])[0])/n > (1.0-percent)/2:
            lower = value
        elif upper is None and float(np.shape(np.where(data > value)[0])[0])/n < (1.0-percent)/2:
            #print np.shape(np.where(data < value)[0])[0], np.shape(np.where(data > value)[0])[0]
            #print value, float(np.shape(np.where(data > value))[0])/n, (1.0-percent)/2
            upper = value
    return lower, upper


def get_cdf(xs, pdf):
    dx = (max(xs) - min(xs))/len(xs)
    cdf = np.zeros(len(pdf)+1)
    for i in np.arange(1, len(pdf) + 1):
       cdf[i] = pdf[i-1]*dx + cdf[i-1]
    print cdf
    return cdf
    
def draw_points(xs, cdf):
    ys = np.random.uniform(size=1000)
    vals = np.zeros_as(ys)
    j = 0
    for y in ys:
        for i in np.arange(0, len(cdf)):
            if cdf[i] >= y:
                vals[j] = xs[i]
                break
        j += 1
    return vals
    
for i in np.arange(0, num_bootstrap):
    indices = np.random.randint(0, len(r_hks_a) + len(r_hks_na), size=(len(r_hks_a) + len(r_hks_na)))
    indices_a = indices[np.where(indices < len(r_hks_a))[0]]
    indices_na = indices[np.where(indices >= len(r_hks_a))[0]]
    indices_na -= len(r_hks_a)
    #r_hks_a_bs = np.random.choice(r_hks_a, len(r_hks_a))
    #r_hks_na_bs = np.random.choice(r_hks_na, len(r_hks_na))
    r_hks_a_bs = r_hks_a[indices_a]
    r_hks_na_bs = r_hks_na[indices_na]
    density_a_bs = gaussian_kde(r_hks_a_bs, bw_method = cov_a)#calc_kde(r_hks_a_bs, r_hk_bins_values)#gaussian_kde(r_hks_a_bs, bw_method = kde_cov)
    density_na_bs = gaussian_kde(r_hks_na_bs, bw_method = cov_na)#calc_kde(r_hks_na_bs, r_hk_bins_values)#gaussian_kde(r_hks_na_bs, bw_method = kde_cov)
    d_a_bs = density_a_bs(r_hk_bins_values)
    d_na_bs = density_na_bs(r_hk_bins_values)
    rel_d_a_bs.append(d_a_bs/(d_a_bs + d_na_bs))
    #ax2.plot(r_hk_bins_values, d_a_bs/(d_a_bs + d_na_bs), "k.")

rel_d_a_bs = np.asarray(rel_d_a_bs)
rel_d_a_bs_mean = np.mean(rel_d_a_bs, axis=0)
rel_d_a_bs_std = np.std(rel_d_a_bs, axis=0)
ax2.plot(r_hk_bins_values, rel_d_a_bs_mean, "k-")
#ax2.plot(r_hk_bins_values, d_a*len(r_hks_a)/(len(r_hks_a) + len(r_hks_na)), "r-")
#ax2.plot(r_hk_bins_values, d_na*len(r_hks_na)/(len(r_hks_a) + len(r_hks_na)), "b-")

lowers = np.zeros(np.shape(rel_d_a_bs)[1])
uppers = np.zeros(np.shape(rel_d_a_bs)[1])
for i in np.arange(0, np.shape(rel_d_a_bs)[1]):
    lower, upper = get_conf_ints(rel_d_a_bs[:,i])
    lowers[i] = lower
    uppers[i] = upper

#ax2.fill_between(r_hk_bins_values, rel_d_a_bs_mean + 2.0 * rel_d_a_bs_std, rel_d_a_bs_mean - 2.0 * rel_d_a_bs_std, alpha=0.1, facecolor='gray', interpolate=True)
ax2.fill_between(r_hk_bins_values, uppers, lowers, alpha=0.1, facecolor='gray', interpolate=True)

ax1.set_xlim([min(r_hk_bins_values), max(r_hk_bins_values)])
ax2.set_xlim([min(r_hk_bins_values), max(r_hk_bins_values)])


fig.savefig("cyc_vs_act.pdf")
plt.close(fig)
