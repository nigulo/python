# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 10:48:24 2017

@author: olspern1
"""
#import scipy.signal as signal
#from scipy.signal import spectral
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import os
import os.path

num_resamples = 100
input_path = "cleaned"
detrended_path = "detrended"
spectra_path = "spectra0.05"

def resample(data):
    return 

def get_seasons(dat, num_days, seasonal):
    seasons = list()
    #res = list()
    last_t = float('-inf')
    season_start = float('-inf')
    season = list()
    for t, y in dat:
        if (seasonal and t - last_t > num_days/3) or t - season_start >= num_days:
            if np.shape(season)[0] > 0:
                #res.append([(last_t + season_start)/2, season_mean/np.shape(season)[0]])
                seasons.append(np.asarray(season))
            season_start = t
            season = list()
        last_t = t
        season.append([t, y])
    if np.shape(season)[0] > 0:
        #res.append([(last_t + season_start)/2, season_mean/np.shape(season)[0]])
        seasons.append(np.asarray(season))
    return seasons


offset = 1979.3452

for root, dirs, files in os.walk(input_path):
    for file in files:
        if file[-4:] == ".dat":
            star = file[:-4]
            star = star.upper()
            if (star[-3:] == '.CL'):
                star = star[0:-3]
            if (star[0:2] == 'HD'):
                star = star[2:]
            data = np.loadtxt(input_path+"/"+file, usecols=(0,1), skiprows=1)
            t = data[:,0]/365.25 + offset
            y = data[:,1]
            y_gen = np.sin(t*2*np.pi*0.25)

            seasons = get_seasons(zip(t, y), 1.0, True)
            y_wo_season_mean = list()
            for s in seasons:
                season_mean = np.mean(s[:,1])
                for d in s:
                    y_wo_season_mean.append(d[1] - season_mean)
            y_wo_season_mean = np.asarray(y_wo_season_mean)

            std1 = np.std(y)
            std2 = np.std(y_wo_season_mean)

            seasons = get_seasons(zip(t, y_gen), 1.0, True)
            y_wo_season_mean = list()
            for s in seasons:
                season_mean = np.mean(s[:,1])
                for d in s:
                    y_wo_season_mean.append(d[1] - season_mean)
            y_wo_season_mean = np.asarray(y_wo_season_mean)

            std3 = np.std(y_gen)
            std4 = np.std(y_wo_season_mean)

            print "Seasonal vs. global variance comparison for " + star + " " + str((std1 - std2)/std1) + " " + str((std3 - std4)/std3)
             
            
            fig, plots = plt.subplots(2, 1, figsize=(6, 6))

            (plot1) = plots[0]            
            (plot2) = plots[1]
            plot1.hist(y)
            plot2.hist(y_wo_season_mean)
            
            fig.savefig('variations/' + star + '.png')
            plt.close(fig)
