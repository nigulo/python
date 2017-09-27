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

#input_path = "downsampling/results"
input_path = "BGLST_input"
    
output_path = "no_cycles"
offset = 1979.3452
num_r_hk_bins = 1000
num_bootstrap = 1000

rot_periods = mw_utils.load_rot_periods()

min_bic, max_bic, all_cycles = mw_utils.read_bglst_cycles("BGLST_BIC_6/results.txt")
r_hks = mw_utils.load_r_hk()
min_r_hk = None
max_r_hk = None
for star in r_hks.keys():
    r_hk = r_hks[star]
    if min_r_hk is None or r_hk < min_r_hk:
        min_r_hk = r_hk
    if max_r_hk is None or r_hk > max_r_hk:
        max_r_hk = r_hk
r_hk_counts_na = []
r_hk_counts_a = []
print min_r_hk, max_r_hk
r_hk_bins_values = np.linspace(min_r_hk, max_r_hk, num=num_r_hk_bins)
#r_hk_bin_counts_a = np.zeros(num_r_hk_bins)
#r_hk_bin_counts_na = np.zeros(num_r_hk_bins)

num_stars = len(r_hks.keys())
print num_stars

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=False)
fig.set_size_inches(6, 12)
ax1.set_ylabel(r'$\sigma^2_{\rm s}/\sigma^2$')
ax2.set_ylabel(r'$N_{\rm cyc}/N$')

ax2.set_xlabel(r'$\langle R\prime_{\rm HK}\rangle$')
ax1.text(0.9, 0.9,'(a)', horizontalalignment='center', transform=ax1.transAxes)
ax2.text(0.9, 0.9,'(b)', horizontalalignment='center', transform=ax2.transAxes)

r_hks_a = []
r_hks_na = []
var_ratios_a = []
var_ratios_na = []

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
            print mean_seasonal_var/total_var
            if (r_hks.has_key(star)):
                r_hk = r_hks[star]
                if all_cycles.has_key(star):
                    r_hk_counts_a.append(round((r_hk - min_r_hk)/(max_r_hk - min_r_hk)*num_r_hk_bins)*(max_r_hk - min_r_hk)/num_r_hk_bins+min_r_hk)
                    #r_hk_bin_counts_a[min(num_r_hk_bins - 1, int((r_hk - min_r_hk)/(max_r_hk - min_r_hk)*num_r_hk_bins))] += 1
                else:
                    r_hk_counts_na.append(round((r_hk - min_r_hk)/(max_r_hk - min_r_hk)*num_r_hk_bins)*(max_r_hk - min_r_hk)/num_r_hk_bins+min_r_hk)
                    #r_hk_bin_counts_na[min(num_r_hk_bins - 1, int((r_hk - min_r_hk)/(max_r_hk - min_r_hk)*num_r_hk_bins))] += 1
                
                if all_cycles.has_key(star):
                    r_hks_a.append(r_hk)
                    var_ratios_a.append(mean_seasonal_var/total_var)
                else:
                    r_hks_na.append(r_hk)
                    var_ratios_na.append(mean_seasonal_var/total_var)
                #ax2.scatter(r_hk_bins_values, r_hk_bins_counts/num_stars, marker=markers.MarkerStyle("o", fillstyle=None), lw=1, facecolors="blue", color="blue", s=10, edgecolors="blue")
                #y = mlab.normpdf(bins, np.mean(star_cycle_samples), np.std(star_cycle_samples))
                #l = plt.plot(bins, y, 'r--', linewidth=1)

#n, bins, patches = ax2.hist(r_hk_counts_na, 10, normed=1, facecolor='green', alpha=0.75)
#n, bins, patches = ax2.hist(r_hk_counts_a, 10, normed=1, facecolor='blue', alpha=0.75)

ax1.scatter(r_hks_na, var_ratios_na, marker=markers.MarkerStyle("o", fillstyle=None), lw=1, facecolors="blue", color="blue", s=10, edgecolors="blue")
ax1.scatter(r_hks_a, var_ratios_a, marker=markers.MarkerStyle("+", fillstyle=None), lw=1, facecolors="red", color="red", s=10, edgecolors="red")
slope, intercept, r_value, p_value, std_err = stats.linregress(r_hks_na, var_ratios_na)
ax1.plot(r_hk_bins_values, r_hk_bins_values*slope + intercept, "b-")
slope, intercept, r_value, p_value, std_err = stats.linregress(r_hks_a, var_ratios_a)
ax1.plot(r_hk_bins_values, r_hk_bins_values*slope + intercept, "r-")


density_a = gaussian_kde(r_hk_counts_a)
density_na = gaussian_kde(r_hk_counts_na)

density_all = gaussian_kde(r_hk_counts_a + r_hk_counts_na)
d_a = density_a(r_hk_bins_values)
d_na = density_na(r_hk_bins_values)
d_all = density_all(r_hk_bins_values)
#ax2.plot(r_hk_bins_values, r_hk_bin_counts_na/(r_hk_bin_counts_a+r_hk_bin_counts_na), "k-")
#ax2.plot(r_hk_bins_values, d_a/(d_a+d_na), "k-")
ax2.plot(r_hk_bins_values, d_all, "k--")


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

for i in np.arange(0, num_bootstrap):
    r_hk_counts_a_bs = np.random.choice(r_hk_counts_a, len(r_hk_counts_a))
    r_hk_counts_na_bs = np.random.choice(r_hk_counts_na, len(r_hk_counts_na))
    density_a_bs = gaussian_kde(r_hk_counts_a_bs)
    density_na_bs = gaussian_kde(r_hk_counts_na_bs)
    d_a_bs = density_a_bs(r_hk_bins_values)
    d_na_bs = density_na_bs(r_hk_bins_values)
    rel_d_a_bs.append(d_a_bs/(d_a_bs + d_na_bs))
    #ax2.plot(r_hk_bins_values, d_a_bs/(d_a_bs + d_na_bs), "k.")

rel_d_a_bs = np.asarray(rel_d_a_bs)
rel_d_a_bs_mean = np.mean(rel_d_a_bs, axis=0)
rel_d_a_bs_std = np.std(rel_d_a_bs, axis=0)
ax2.plot(r_hk_bins_values, rel_d_a_bs_mean, "k-")

lowers = np.zeros(np.shape(rel_d_a_bs)[1])
uppers = np.zeros(np.shape(rel_d_a_bs)[1])
for i in np.arange(0, np.shape(rel_d_a_bs)[1]):
    lower, upper = get_conf_ints(rel_d_a_bs[:,i])
    lowers[i] = lower
    uppers[i] = upper

#ax2.fill_between(r_hk_bins_values, rel_d_a_bs_mean + 2.0 * rel_d_a_bs_std, rel_d_a_bs_mean - 2.0 * rel_d_a_bs_std, alpha=0.1, facecolor='gray', interpolate=True)
ax2.fill_between(r_hk_bins_values, uppers, lowers, alpha=0.1, facecolor='gray', interpolate=True)

fig.savefig("non_active_stars.pdf")
plt.close(fig)
