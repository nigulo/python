# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 13:42:07 2017

@author: olspern1
"""

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

import os
import os.path
import mw_utils

offset = 1979.3452

coef = 10

num_groups = 1
group_no = 0
if len(sys.argv) > 1:
    num_groups = int(sys.argv[1])
if len(sys.argv) > 2:
    group_no = int(sys.argv[2])

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
    #if star != "10476":
    #    continue
    print star
    dat = np.loadtxt("cleaned/"+file, usecols=(0,1), skiprows=1)
    t = dat[:,0]
    y = dat[:,1]

    t_orig = dat[:,0]

    sample_size = len(t)/coef

    freqs = np.linspace(0.001, 1.0, 1000)
    
    power1 = LombScargle(t/365.25, y).power(freqs, normalization='psd')
    power1 /= max(power1)
    
    (t_ds, y_ds, noise_var_prop) = mw_utils.daily_averages(t, y, mw_utils.get_seasonal_noise_var(t/365.25, y))
    #noise_var_prop = mw_utils.get_seasonal_noise_var(t/365.25, y)
    #np.savetxt("GPR_stan/" + star + ".dat", np.column_stack((t_daily, y_daily)), fmt='%f')
    
    power2 = LombScargle(t_ds/365.25, y_ds).power(freqs, normalization='psd')
    power2 /= max(power2)
    
    total_error = sum(abs(power1 - power2)) / sum(power1)
    print star, total_error

    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(10, 8)
    ax1.plot(freqs, power1)
    ax1.plot(freqs, power2)
    fig.savefig("temp/" + star + ".png")
    plt.close(fig)
