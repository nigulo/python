# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 09:35:12 2016

@author: nigul
"""

import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['figure.figsize'] = (20, 30)
import numpy as np
import pylab as plt
import sys
from astropy.stats import LombScargle
#import mw_utils

import os
import os.path

offset = 1979.3452

coef = 2

num_groups = 1
group_no = 0
if len(sys.argv) > 1:
    num_groups = int(sys.argv[1])
if len(sys.argv) > 2:
    group_no = int(sys.argv[2])

files = []

data_dir = "../remove_rotation/results"
skiprows = 0#1

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
            files.append(file)

modulo = len(files) % num_groups
group_size = len(files) / num_groups
if modulo > 0:
    group_size +=1


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
    if star != "SUN":
        continue
    dat = np.loadtxt(data_dir+"/"+file, usecols=(0,1), skiprows=skiprows)
    t = dat[:,0]
    y = dat[:,1]

    t_orig = np.array(t)

    orig_len = len(t)

    t /= 365.25
    t += offset

    sample_size = len(t)/coef

    freqs = np.linspace(0.001, 1.0, 3000)
    
    orig_power = LombScargle(t, y).power(freqs, normalization='psd')
    orig_power /= max(orig_power)

    #orig_power_maxima = orig_power[mw_utils.find_local_maxima(orig_power)]
    min_power_down = orig_power
    while len(t) > 2000:
        power = LombScargle(t, y).power(freqs, normalization='psd')
        power /= max(power)
        #power_maxima = orig_power[mw_utils.find_local_maxima(power)]
        min_error = -1
        if len(t) - sample_size < 2000:
            sample_size = len(t) - 2000
        for i in np.arange(100000):
            print i
            indices = np.sort(np.random.choice(len(t), len(t) - sample_size, replace=False))
            #t_down = np.concatenate([t[:i], t[i+1:]])
            #y_down = np.concatenate([y[:i], y[i+1:]])
            t_down = t[indices]
            y_down = y[indices]
            power_down = LombScargle(t_down, y_down).power(freqs, normalization='psd')
            power_down /= max(power_down)
            #power_down_maxima = power_down[mw_utils.find_local_maxima(power_down)]
            error = np.sqrt(sum((power - power_down)**2))
            if (min_error < 0 or error < min_error):
                min_error = error
                min_indices = indices
                min_power_down = power_down
        print len(t)
        #t = np.concatenate([t[:min_i], t[min_i+1:]])
        #y = np.concatenate([y[:min_i], y[min_i+1:]])
        t = t[min_indices]
        y = y[min_indices]
        t_orig = t_orig[min_indices]
    total_error = np.sqrt(sum((orig_power - min_power_down)**2)/sum(orig_power**2))
    print star, float(orig_len)/len(t), total_error

    dat = np.column_stack((t_orig, y))
    np.savetxt("results/" + star + ".dat", dat, fmt='%f')
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(10, 8)
    ax1.plot(freqs, orig_power)
    ax1.plot(freqs, min_power_down)
    fig.savefig("results/" + star + ".png")
    plt.close(fig)

