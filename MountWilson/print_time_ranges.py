# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:47:36 2017

@author: olspern1
"""
import numpy as np
import os
import os.path

offset = 1979.3452

time_ranges = list()
for root, dirs, files in os.walk("cleaned"):
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
            data = np.loadtxt("cleaned/"+file, usecols=(0,1), skiprows=1)
            t = data[:,0]
            y = data[:,1]
            t = t/365.25 + offset
            max_t = max(t)
            time_range = max_t - min(t)
            time_ranges.append([star, time_range, max_t])
np.savetxt("time_ranges.dat", time_ranges, fmt='%s')
