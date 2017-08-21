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
    dat1 = np.loadtxt("cleaned/"+file, usecols=(0,1), skiprows=1)
    t1 = dat1[:,0]
    y1 = dat1[:,1]

    t1 /= 365.25
    t1 += offset

    dat2 = np.loadtxt("cleaned_wo_rot/"+star+".dat", usecols=(0,1), skiprows=0)
    t2 = dat2[:,0]
    y2 = dat2[:,1]

    t2 /= 365.25
    t2 += offset

    freqs = np.linspace(0.001, 1.0, 1000)
    
    power1 = LombScargle(t1, y1).power(freqs, normalization='psd')
    power2 = LombScargle(t2, y2).power(freqs, normalization='psd')
    
    total_error = sum(abs(power1 - power2)) / sum(power1)
    print star, total_error

    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(10, 8)
    ax1.plot(freqs, power1)
    ax1.plot(freqs, power2)
    fig.savefig("temp/" + star + ".png")
    plt.close(fig)
