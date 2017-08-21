# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 09:35:12 2016

@author: nigul
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import os
import os.path
import itertools

offset = 1979.3452
bootstrap_count = 1000
p_value = 0.01

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

#def resample_season(season):
#    indices = np.random.choice(np.shape(season)[0], np.shape(season)[0], replace=True, p=None)
#    return season[indices,:]

#def resample(seasons):
#    resampled_seasons = np.zeros_like(seasons)
#    i = 0
#    for season in seasons:
#        resampled_seasons[i] = resample_season(season)        
#        i = i + 1
#    return resampled_seasons

def resample_seasons(seasons):
    indices = np.random.choice(np.shape(seasons)[0], np.shape(seasons)[0], replace=True, p=None)
    resampled_seasons=list()
    for i in np.arange(0, len(seasons)):
        season = seasons[i]
        season_indices = np.random.choice(len(seasons[indices[i]]), len(season), replace=True, p=None)
        resampled_season = seasons[indices[i]][season_indices]
        resampled_season[:,0] = season[:,0]
        #if i == 0:
        #    print resampled_season
        #    print season
        resampled_seasons.append(resampled_season)
    return np.asarray(resampled_seasons)

def calc_means(seasons):
    means = np.zeros((np.shape(seasons)[0], 2))
    i = 0
    for season in seasons:
        means[i] = [(np.min(season[:,0]) + np.max(season[:,0]))/2, np.mean(season[:,1])]
        i += 1
    return means

def calc_cycle_count(ups_and_downs, all_ups_and_downs_bs):
    direction = None
    ups = 0.0
    downs = 0.0
    ups_and_downs_counts_bs = np.zeros(len(ups_and_downs))
    for ups_and_downs_bs in all_ups_and_downs_bs:
        #print np.shape(ups_and_downs_counts_bs)
        #print np.shape(ups_and_downs_bs)
        ups_and_downs_counts_bs += ups_and_downs_bs
    ups_counts_bs = (ups_and_downs_counts_bs / len(all_ups_and_downs_bs) + 1) / 2 - 0.5
    downs_counts_bs = abs(ups_and_downs_counts_bs / len(all_ups_and_downs_bs) - 1) / 2 - 0.5
    #print ups_counts_bs    
    #print downs_counts_bs    
    
    #p_value_per_season = pow(p_value, 1/float(np.shape(ups_and_downs)[0]))
    for i in np.arange(0, len(ups_and_downs)):
        up_or_down = ups_and_downs[i]
        #up_or_down_bs = ups_and_downs_bs[i]
        if direction != True and up_or_down == 1 and ups_counts_bs[i] < p_value:
            direction = True
            ups += 1
        elif direction != False and up_or_down == -1 and downs_counts_bs[i] < p_value:
            direction = False
            downs += 1
    cycle_count = (ups + downs) / 2
    #match_count = 0.0
    #for ups_and_downs_bs in all_ups_and_downs_bs:
    #    #if (all(ups_and_downs_bs == ups_and_downs)):
    #    #    match_count += 1
    #    
    #    direction_bs = None
    #    ups_bs = 0.0
    #    downs_bs = 0.0
    #    for i in np.arange(0, len(ups_and_downs_bs)):
    #        up_or_down_bs = ups_and_downs_bs[i]
    #        if direction_bs != True and up_or_down_bs == 1: #and up_or_down_bs < p_value_per_season:
    #            direction_bs = True
    #            ups_bs += 1
    #        elif direction_bs != False and up_or_down_bs == -1: #and up_or_down_bs > -p_value_per_season:
    #            direction_bs = False
    #            downs_bs += 1
    #    if ups_bs == ups and downs_bs == downs:
    #        match_count += 1
    #p_value = match_count / len(all_ups_and_downs_bs)
    return (cycle_count, 0)

def calc_ups_and_downs(yearly_avgs):
    ups_and_downs = np.zeros(np.shape(yearly_avgs)[0]-1);
    last_y = yearly_avgs[0,1]
    i = 0
    for y in yearly_avgs[1:,1]:
        if y >= last_y:
            ups_and_downs[i] = 1.0
        else:
            ups_and_downs[i] = -1.0
        i += 1
        last_y = y
    return ups_and_downs

def to_years(jd):
    return jd/365.25

f1=open('syntactic/results.txt', 'w')
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
            cycle_lengths = list()
            initial_seasons = get_seasons(dat, num_days, True)
            while num_days < time_range/3:
                color = colors.next()
                marker = markers.next()
                seasons = get_seasons(dat, num_days, smoothing_index == 0)
                yearly_avgs = calc_means(seasons)
                yearly_avgs_time_range=max(yearly_avgs[:,0]) - min(yearly_avgs[:,0])
                plt.plot(to_years(yearly_avgs[:,0])+offset, yearly_avgs[:,1], c=color, marker=marker, markersize=5+smoothing_index)
                ups_and_downs = calc_ups_and_downs(yearly_avgs)
                ups_and_downs_bs = list()#np.zeros(np.shape(seasons)[0]-1)
                for i in range(0, bootstrap_count):
                    resampled_data = list()
                    resampled_seasons = resample_seasons(initial_seasons)
                    for s in resampled_seasons:
                        for d in s:
                            resampled_data.append(d)
                    #print(np.shape(dat))
                    #print(np.shape(np.asarray(resampled_data)))
                    seasons_bs = get_seasons(np.asarray(resampled_data), num_days, smoothing_index == 0)
                    yearly_avgs_bs = calc_means(seasons_bs)
                    #ups_and_downs_bs += calc_ups_and_downs(yearly_avgs_bs)
                    ups_and_downs_bs.append(calc_ups_and_downs(yearly_avgs_bs))
                    #ups_and_downs += ups_and_downs_bs
                #ups_and_downs_bs /= bootstrap_count
                	    
                #print ups_and_downs
                #ups_and_downs /= (bootstrap_count + 1)
                #ups_and_downs = np.round(ups_and_downs)
                (cycle_count, p_value) = calc_cycle_count(ups_and_downs, ups_and_downs_bs)
                #print np.asarray(yearly_avgs)
                print star + " " + str(num_days) + " " + str(p_value) + " " + str(cycle_count)
                if cycle_count > 2 and p_value <= 0.01:
                    cycle_lengths.append(to_years(yearly_avgs_time_range)/cycle_count)
                else:
                    cycle_lengths.append(0)
                #################################################
                num_days += 365.25
                smoothing_index += 1

            plt.savefig("syntactic/"+star + '.png')
            plt.close()
            
            f1.write(star + " " + str(to_years(time_range)) + " " + (' '.join(['%s' % (cycle_length) for cycle_length in cycle_lengths])) + "\n")
            f1.flush()
            

f1.close()
