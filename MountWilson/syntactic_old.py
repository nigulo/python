# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 09:35:12 2016

@author: nigul
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers
import scipy.stats
import itertools
import matplotlib.lines as mlines

import os
import os.path

offset = 1979.3452
bootstrap_count = 100
conf_int = 0.99

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
    return np.asarray(seasons)

def resample_season(season):
    indices = np.random.choice(np.shape(season)[0], np.shape(season)[0], replace=True, p=None)
    return season[indices,:]

def resample(seasons):
    resampled_seasons = np.zeros_like(seasons)
    i = 0
    for season in seasons:
        resampled_seasons[i] = resample_season(season)        
        i = i + 1
    return resampled_seasons

def calc_means(seasons):
    means = np.zeros((np.shape(seasons)[0], 2))
    i = 0
    for season in seasons:
        means[i] = [(np.min(season[:,0]) + np.max(season[:,0]))/2, np.mean(season[:,1])]
        i = i + 1
    return means

#based on significance of single up or down
#def calc_cycle_count(ups_and_downs):
#    direction = None
#    ups = 0
#    downs = 0
#    conf_int_per_season = pow(conf_int, 1/float(np.shape(ups_and_downs)[0]))
#    #print conf_int_per_season
#    for up_or_down in ups_and_downs:
#        if direction != True and (1 + up_or_down) / 2 > conf_int_per_season:
#            direction = True
#            ups += 1
#        if direction != False and (1 + up_or_down) / 2 < conf_int_per_season:
#            direction = False
#            downs += 1
#    return (ups + downs) / 2

def calc_cycle_count(ups_and_downs):
    direction = None
    ups = 0
    downs = 0
    #print conf_int_per_season
    for up_or_down in ups_and_downs:
        if direction != True and up_or_down > 0:
            direction = True
            ups += 1
        if direction != False and up_or_down < 0:
            direction = False
            downs += 1
    return (float(ups) + float(downs)) / 2.0

def calc_ups_and_downs(yearly_avgs):
    ups_and_downs = np.zeros(np.shape(yearly_avgs)[0]-1);
    last_y = yearly_avgs[0,1]
    i = 0
    for y in yearly_avgs[1:,1]:
        if y > last_y:
            ups_and_downs[i] = 1.0
        if y < last_y:
            ups_and_downs[i] = -1.0
        i = i + 1
        last_y = y
    return ups_and_downs

def to_years(jd):
    return jd/365.25

f1=open('syntactic/results.txt', 'w+')
first = True
for root, dirs, files in os.walk("cleaned"):
    for file in files:
        if file[-4:] == ".dat":
            star = file[:-4]
            star = star.upper()
            if (star[-3:] == '.CL'):
                star = star[0:-3]
            if (star[0:2] == 'HD'):
                star = star[2:]
            dat = np.loadtxt("cleaned/"+file, usecols=(0,1), skiprows=1)
            dat[:,0] = dat[:,0]
            t = dat[:,0]
            y = dat[:,1]
            time_range = max(t) - min(t)
            
            fig = plt.gcf()
            fig.set_size_inches(18, 6)
            plt.plot(to_years(t)+offset, y, 'b+')
            num_days = 365.25
            smoothing_index = 0
            colors = itertools.cycle(['g', 'r', 'c', 'm', 'y', 'k'])
            markers = itertools.cycle(mlines.Line2D.filled_markers)
            all_cycle_lengths = list()
            all_cycle_length_stds = list()
            all_num_days = list()
            while num_days < time_range/2:
                color = colors.next()
                marker = markers.next()
                seasons = get_seasons(dat, num_days, smoothing_index == 0)
                yearly_avgs = calc_means(seasons)
                plt.plot(to_years(yearly_avgs[:,0])+offset, yearly_avgs[:,1], c=color, marker=marker, markersize=5+smoothing_index)
                ups_and_downs = calc_ups_and_downs(yearly_avgs)
                cycle_counts = np.zeros(bootstrap_count + 1)
                cycle_counts[0] = calc_cycle_count(ups_and_downs)
                for i in range(1, bootstrap_count + 1):
                    seasons_bs = resample(seasons)
                    yearly_avgs_bs = calc_means(seasons_bs)
                    ups_and_downs_bs = calc_ups_and_downs(yearly_avgs_bs)
                    cycle_counts[i] = calc_cycle_count(ups_and_downs_bs)
                #print cycle_counts
                cycle_lengths = np.ones_like(cycle_counts) * to_years(time_range) / cycle_counts;
                #(skewKurt, normality) = scipy.stats.normaltest(cycle_lengths)
                #print("skewKurt and normality: %f %f" % (skewKurt, normality))
                cycle_length_mean = np.mean(cycle_lengths)
                cycle_length_std = np.std(cycle_lengths)
                all_num_days.append(num_days)
                if cycle_length_mean >= to_years(time_range):
                    cycle_length_mean = 0
                    cycle_length_std = 0
                all_cycle_lengths.append(cycle_length_mean)
                all_cycle_length_stds.append(cycle_length_std)
    
                #################################################
                num_days += 365.25
                smoothing_index += 1


            plt.savefig("syntactic/"+star + '.png')
            plt.close()

            if (first):
                f1.write(" " + (' '.join(['%s' % (num_days) for num_days in all_num_days])) + "\n")
                first = False
                
            f1.write(star + " " + (' '.join(['%s' % (cycle_length) for cycle_length in all_cycle_lengths])) + "\n")
            f1.write(" " + (' '.join(['%s' % (cycle_length_std) for cycle_length_std in all_cycle_length_stds]))+ "\n")
            
            #print (star + " " + str(cycle_lengths))
                
            #break

f1.close()