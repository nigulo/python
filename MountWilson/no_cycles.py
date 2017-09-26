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

import os
import os.path
import mw_utils

num_resamples = 1000
num_bootstrap = 0
#input_path = "downsampling/results"
input_path = "BGLST_input"
    
output_path = "no_cycles"
offset = 1979.3452
num_r_hk_bins = 7

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
r_hk_bin_counts_a = np.zeros(num_r_hk_bins)
r_hk_bin_counts_na = np.zeros(num_r_hk_bins)

num_stars = len(r_hks.keys())
print num_stars

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=False)
fig.set_size_inches(6, 12)
ax2.set_xlabel(r'$\langle R\prime_{\rm HK}\rangle$')
ax1.text(0.9, 0.9,'(a)', horizontalalignment='center', transform=ax1.transAxes)
ax2.text(0.9, 0.9,'(b)', horizontalalignment='center', transform=ax2.transAxes)

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
                    r_hk_bin_counts_a[min(num_r_hk_bins - 1, int((r_hk - min_r_hk)/(max_r_hk - min_r_hk)*num_r_hk_bins))] += 1
                else:
                    r_hk_counts_na.append(round((r_hk - min_r_hk)/(max_r_hk - min_r_hk)*num_r_hk_bins)*(max_r_hk - min_r_hk)/num_r_hk_bins+min_r_hk)
                    r_hk_bin_counts_na[min(num_r_hk_bins - 1, int((r_hk - min_r_hk)/(max_r_hk - min_r_hk)*num_r_hk_bins))] += 1
                    
                ax1.scatter(r_hk, mean_seasonal_var/total_var, marker=markers.MarkerStyle("o", fillstyle=None), lw=1, facecolors="blue", color="blue", s=10, edgecolors="blue")
                #ax2.scatter(r_hk_bins_values, r_hk_bins_counts/num_stars, marker=markers.MarkerStyle("o", fillstyle=None), lw=1, facecolors="blue", color="blue", s=10, edgecolors="blue")
                #y = mlab.normpdf(bins, np.mean(star_cycle_samples), np.std(star_cycle_samples))
                #l = plt.plot(bins, y, 'r--', linewidth=1)

#n, bins, patches = ax2.hist(r_hk_counts_na, 10, normed=1, facecolor='green', alpha=0.75)
#n, bins, patches = ax2.hist(r_hk_counts_a, 10, normed=1, facecolor='blue', alpha=0.75)
print r_hk_bin_counts_na
print r_hk_bin_counts_a
print r_hk_bins_values
ax2.plot(r_hk_bins_values, r_hk_bin_counts_na/(r_hk_bin_counts_a+r_hk_bin_counts_na), "k-")

fig.savefig("non_active_stars.pdf")
plt.close(fig)
