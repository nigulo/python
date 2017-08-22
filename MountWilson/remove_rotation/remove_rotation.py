# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 09:35:12 2016

@author: nigul
"""

import sys
sys.path.append('../')
import matplotlib as mpl

mpl.use('Agg')
mpl.rcParams['figure.figsize'] = (20, 30)
import numpy as np
import pylab as plt
import mw_utils
import GPR_QP
from prewhitener import Prewhitener
from filelock import FileLock

import os
import os.path
#import pandas as pd

down_sample_factor = 1

num_groups = 1
group_no = 0
if len(sys.argv) > 1:
    num_groups = int(sys.argv[1])
if len(sys.argv) > 2:
    group_no = int(sys.argv[2])

#data_dir = "../downsampling/results"
data_dir = "../cleaned"
skiprows = 1

files = []

for root, dirs, dir_files in os.walk(data_dir):
    for file in dir_files:
        if file[-4:] == ".dat":
            star = file[:-4]
            star = star.upper()
            if (star[-3:] == '.CL'):
                star = star[0:-3]
            if (star[0:2] == 'HD'):
                star = star[2:]
            while star[0] == '0': # remove leading zeros
                star = star[1:]
            if not os.path.isfile("results/" + star + ".dat"):
                files.append(file)

modulo = len(files) % num_groups
group_size = len(files) / num_groups
if modulo > 0:
    group_size +=1

#output = open("GPR_stan/results.txt", 'w')
#output.close()
#output = open("GPR_stan/all_results.txt", 'w')
#output.close()

offset = 1979.3452

rot_periods = mw_utils.load_rot_periods("../")

for i in np.arange(0, len(files)):
    if i < group_no * group_size or i >= (group_no + 1) * group_size:
        continue
    file = files[i]
    star = file[:-4]
    star = star.upper()
    if (star[-3:] == '.CL'):
        star = star[0:-3]
    if (star[0:2] == 'HD'):
        star = star[2:]
    while star[0] == '0': # remove leading zeros
        star = star[1:]
    #if star != "201091":
    #    continue
    print star
    dat = np.loadtxt(data_dir+"/"+file, usecols=(0,1), skiprows=skiprows)
    t_orig = dat[:,0]
    y_orig = dat[:,1]

    n_orig = len(t_orig)
    
    if (rot_periods.has_key(star)):
        rot_period = rot_periods[star]
    
        if down_sample_factor >= 2:
            #indices = np.random.choice(len(t), len(t)/down_sample_factor, replace=False, p=None)
            #indices = np.sort(indices)
        
            #t = t[indices]
            #y = y[indices]
    
            t = t_orig[0::down_sample_factor] 
            y = y_orig[0::down_sample_factor] 
        else:
            t = t_orig
            y = y_orig
            
        #(t, y, noise_var_prop) = mw_utils.daily_averages(t, y, mw_utils.get_seasonal_noise_var(t/365.25, y))
        #noise_var_prop = mw_utils.get_seasonal_noise_var(t/365.25, y)
        #np.savetxt("GPR_stan/" + star + ".dat", np.column_stack((t_daily, y_daily)), fmt='%f')
    
        t_jd = t
        t /= 365.25
        t += offset
    
        noise_var_prop = mw_utils.get_seasonal_noise_var(t, y)
        seasonal_means_var = np.var(mw_utils.get_seasonal_means(t, y)[:,1])
    
        n = len(t)
        
        # Fit GP
        
        m = np.mean(y)
        noise_var = np.var(y) - seasonal_means_var
        gpr_gp = GPR_QP.GPR_QP(sig_var = seasonal_means_var, length_scale=1.0, freq=0, noise_var=noise_var, rot_freq=0, rot_amplitude=0, trend_var=0, c=0.0)
        t_test = np.linspace(min(t), max(t), 500)
        #(f_mean, pred_var, loglik) = gpr_gp.fit(t, y-m, t_test)
        #f_mean += m
        (f_t, _, _) = gpr_gp.fit(t, y-m, t)
        f_t += m

        # Remove rotational period
        y1 = y - f_t
        rot_freq = 365.25/rot_period

        rot_freqs = np.linspace(rot_freq-rot_freq/5.0, rot_freq+rot_freq/5.0, 10000)
        prewhitener = Prewhitener(star, "LS", t, y1, rot_freqs, num_resamples=1000, num_bootstrap = 0, spectra_path = None, max_iters=100, max_peaks=1, num_indep = 0, white_noise=True)
        res = prewhitener.prewhiten_with_higher_harmonics(p_value=0.01, num_higher_harmonics=0)
        harmonic_index = 0
        assert(len(res) == 1) # There is a for loop, but we currently assume only one result
        for (rot_powers, found_lines_r, y_res, _) in res:
            if len(found_lines_r) > 0:
                r_line_freqs, r_line_powers, _, _, _ = zip(*found_lines_r)
            else:
                r_line_freqs = np.array([])
                r_line_powers = np.array([])
            
            
            y2 = y_res + f_t
            harmonic_index += 1
            var_reduction = abs(np.var(y) - np.var(y2))/np.var(y)
            print "Variance reduction after rot. period removal: ", var_reduction
        y = y2
        fig, (period_plot, ts_plot) = plt.subplots(2, 1, figsize=(20, 8))
        period_plot.set_title(str(rot_period))
        period_plot.plot(t, y1, 'r+', t, y_res+0.2, 'k+')
        ts_plot.scatter(t, y, c='r', marker='o', )
        ts_plot.scatter(t, f_t, c='b', marker='x')
        fig.savefig("temp/" + star + '_rot_period.png')
        plt.close()
    
    else:
        y = y_orig
        t_jd = t_orig

    dat = np.column_stack((t_jd, y))
    np.savetxt("results/" + star + ".dat", dat, fmt='%f')
    with FileLock("GPRLock"):
        with open("results/rot_var_reduction.txt", "a") as output:
            output.write(star + " " + str(var_reduction) + "\n")

    if os.path.isfile("../cleaned_wo_rot/" + star + ".dat"):
        dat1 = np.loadtxt("../cleaned_wo_rot/" + star + ".dat", usecols=(0,1), skiprows=0)
        t_orig1 = dat1[:,0]
        y_orig1 = dat1[:,1]
        if len(t_orig1 == len(t_jd)):
            with FileLock("GPRLock"):
                with open("results/comparison.txt", "a") as output:
                    output.write(star + " " + str(np.std(y_orig1 - y)/np.std(y)) + "\n")
