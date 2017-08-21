# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 09:35:12 2016

@author: nigul
"""

#import scipy.signal as signal
#from scipy.signal import spectral
from astropy.stats import LombScargle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
import random

import os
import os.path

num_resamples = 1000
num_bootstrap = 500
input_path = "cleaned"
detrended_path = "detrended"
spectra_path = "new_spectra"
p_values = np.array([0.01, 0.05])
min_freq = 0
max_freq = 0.5
n_out = 1000
offset = 1979.3452

star_name = None
if len(sys.argv) > 1:
    star_name = sys.argv[1]

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

def resample_seasons(seasons):
    indices = np.random.choice(np.shape(seasons)[0], np.shape(seasons)[0], replace=True, p=None)
    resampled_seasons=list()
    for i in np.arange(0, len(seasons)):
        season = seasons[i]
        season_std = np.std(season[:,1])
        season_indices = np.random.choice(len(seasons[indices[i]]), len(season), replace=True, p=None)
        resampled_season = seasons[indices[i]][season_indices]
        resampled_season[:,0] = season[:,0]
        resampled_season_mean = np.mean(resampled_season[:,1])
        resampled_season_std = np.std(resampled_season[:,1])
        if resampled_season_std != 0: # Not right, but what can we do in case of seasons with low number of observations
            resampled_season[:,1] = resampled_season_mean + (resampled_season[:,1] - resampled_season_mean) * season_std / resampled_season_std
        resampled_seasons.append(resampled_season)
    return resampled_seasons

def getAvgValues(data, avgSampleTime):
    lastTime = 0
    prevValsBuf = []
    prevValsStart = 0
    tot = 0
    avgs = list()
    for t, v in data:
        avgStart = t - avgSampleTime
        # remove too old values
        while prevValsStart < len(prevValsBuf):
            pt, pv = prevValsBuf[prevValsStart]
            if pt > avgStart:
                break
            tot -= pv
            prevValsStart += 1
        # add new item
        tot += v
        prevValsBuf.append((t, v))
        # yield result
        numItems = len(prevValsBuf) - prevValsStart
        avgs.append(tot / numItems)
        # clean prevVals if it's time
        if prevValsStart * 2 > len(prevValsBuf):
            prevValsBuf = prevValsBuf[prevValsStart:]
            prevValsStart = 0
            # recalculate tot for not accumulating float precision error
            tot = sum(v for (t, v) in prevValsBuf)
    return (np.asarray(avgs))

def prewhiten_with_higher_harmonics(star, t, y1, freqs1, p_value=0.01, max_iters=100, num_indep=0, white_noise=True, num_higher_harmonics=0):
    y = y1
    res = list()
    for i in np.arange(1, num_higher_harmonics+2):
        freqs = freqs1 * i
        r = prewhiten(star, t, y, freqs, np.array([p_value]), max_iters, num_indep, white_noise, num_bootstrap=0)
        #[(z0, freqs, powers, periods, peaks, y)] = r
        #print(np.std(y))
        res.append(r[0])
    return res

def calcFapLevel(star, t, y, p_values, num_indep, white_noise):
    if white_noise == True:
        if num_indep == 0:
            num_indep = len(y)
        z0s = -np.log(np.ones_like(len(p_values))-np.power(np.ones_like(len(p_values))-p_values, 1.0/num_indep))
    else:
        nh_powers = list()
        #fig_nh, (plot_nh) = plt.subplots(1, 1, figsize=(6, 3))
        num_indep = 1
        seasons = get_seasons(zip(t, y), 1.0, True)
        for nh_ind in np.arange(0, num_resamples):
            ###################################################################
            ## Just shuffle everything
            #y_bs = y1[np.random.choice(np.shape(y1)[0], np.shape(y1)[0], replace=True, p=None)]
            ###################################################################
            ## Shuffle first seasons then within each season
            resampled_data = list()
            for s in resample_seasons(seasons):
                for d in s:
                    resampled_data.append(d[1])
            y_nh = np.asarray(resampled_data)
            ###################################################################
            y_nh -= np.mean(y_nh)
            y_nh /= np.std(y_nh)
            power = LombScargle(t, y_nh, nterms=1, center_data = star != None).power(freqs, normalization='psd')#/np.var(y)
            #plot_nh.plot(freqs, power)
            ## Take the maximum peak
            #assert(num_indep == 1) # otherwise something is wrong
            #max_power_ind = np.argmax(power[1:-1])
            #bs_powers.append(power[max_power_ind + 1])
            ## Take one random peak
            assert(num_indep == 1) # otherwise something is wrong
            power_ind = random.randrange(np.shape(power)[0] - 1)
            nh_powers.append(power[power_ind + 1])
            ## Little hack, just taking random spectral lines instead of independent
            #indices = np.random.choice(np.shape(power)[0] - 1, num_indep, replace=True, p=None)
            #bs_powers.extend(power[indices])
        #fig_nh.savefig(spectra_path + "/nullhyp/" + star + ".png")
        #plt.close(fig_nh)
        nh_powers = np.sort(nh_powers)
        z0_indexes = p_values * num_resamples * num_indep
        assert(np.all(z0_indexes >= 1))
        z0s = nh_powers[(-np.floor(z0_indexes)).astype(int)]
    return z0s

def prewhiten(star, t, y1, freqs, p_values=np.array([0.01, 0.05]), max_iters=100, num_indep=0, white_noise=True, num_bootstrap=0):
    
    mean = np.mean(y1)
    std = np.std(y1)
    y = y1 - mean
    y /= std

    z0s = calcFapLevel(star, t, y, p_values, num_indep, white_noise)
   
    y0 = y
    res = list()
    for z0, p_value in zip(z0s, p_values):
        print z0
        y = y0
        significant_lines = list()
        specs = list()
        #y_fits = list()
        for i in np.arange(0, max_iters):
            (best_freq, max_power, power, residue, y_fit) = ls(t, y, freqs, z0)
            if best_freq > 0:
                if num_bootstrap > 0:
                    bs_freqs = list()
                    for j in np.arange(0, num_bootstrap):
                        y_bs = y_fit + residue[np.random.choice(np.shape(residue)[0], np.shape(residue)[0], replace=True, p=None)]
                        (best_freq_bs, _, _, _, _) = ls(t, y_bs, freqs, z0)
                        if best_freq_bs > 0:
                            bs_freqs.append(best_freq_bs)
                    (skewKurt, normality) = stats.normaltest(bs_freqs) # normality is p-value
                    plt.hist(bs_freqs, num_bootstrap/10)
                    plt.savefig(spectra_path+str(p_value)+"/" + star + "_" + str(best_freq) + '.png')
                    plt.close()
                                
                    significant_lines.append((best_freq, max_power, np.std(bs_freqs), normality))
                else:
                    significant_lines.append((best_freq, max_power, 0, 0))
                specs.append(power)
                y = residue
            else:
                specs.append(power)
                break
            z0s = calcFapLevel(star, t, y, np.array([p_value]), num_indep, white_noise)
            z0 = z0s[0]
        res.append((z0, freqs, specs, np.asarray(significant_lines), y * std + mean))
    return res

def ls(t, y, freqs, z0):
    power = LombScargle(t, y, nterms=1).power(freqs, normalization='psd')#/np.var(y)
    max_power_ind = np.argmax(power[1:-1])
    if max_power_ind >= 0:
        max_power = power[max_power_ind+1]
        #print freqs[max_power_ind+1], max_power, z0
        if max_power <= z0:
            return (0, 0, power, np.array([]), np.array([]))
        best_freq = freqs[max_power_ind+1]
        y_fit = LombScargle(t, y, nterms = 1).model(t, best_freq)
        return (best_freq, max_power, power, y - y_fit, y_fit)
    else:
        return (0, 0, power, np.array([]), np.array([]))

rot_period_data = np.genfromtxt("periods.txt", usecols=(0,1), dtype=None)
rot_periods = dict()
for [star, rot_period] in rot_period_data:
    rot_periods[star] = rot_period


f1s= map(lambda p_value: open(spectra_path+str(p_value)+'/results.txt', 'w'), p_values)
for root, dirs, files in os.walk(input_path):
    for file in files:
        if file[-4:] == ".dat":
            star = file[:-4]
            star = star.upper()
            if (star[-3:] == '.CL'):
                star = star[0:-3]
            if (star[0:2] == 'HD'):
                star = star[2:]
            if star_name != None and star != star_name:
                continue
            rot_period = 0
            if (rot_periods.has_key(star)):
                rot_period = rot_periods[star]
            #print star + " period is " + str(rot_period)
            #if star != "78366":
            #    continue
            data = np.loadtxt(input_path+"/"+file, usecols=(0,1), skiprows=1)
            print "Calculating Lomb-Scargle spectrum for " + star
            normval = data.shape[0]
            freqs = np.linspace(min_freq, max_freq, n_out)
            t = data[:,0]/365.25 + offset
            y = data[:,1]
            slope, intercept, r_value, p_value, std_err = stats.linregress(t,y)
            y_fit = t * slope + intercept
            #y = y - np.mean(y)
            y = y - y_fit
            #three_sigma = 5*np.std(y)
            #y_ids = np.where(abs(y) < three_sigma)
            #y = y[y_ids]
            #t = t[y_ids]
            detrended = np.column_stack((t, y))
            np.savetxt(detrended_path+"/" + star + ".dat", detrended, fmt='%f')
            time_range = max(t) - min(t)
            #y_fits = []

            # removing rotational modulation
            if (rot_period > 0):
                seasonal_avgs = getAvgValues(zip(t, y), 1)
                y1 = y - seasonal_avgs
                rot_freq = 365.25/rot_period
                rot_freqs = np.linspace(rot_freq-rot_freq/5, rot_freq+rot_freq/5, n_out)
                res = prewhiten_with_higher_harmonics(star, t, y1, rot_freqs, p_value=0.01, max_iters=100, num_indep = 0, white_noise=True, num_higher_harmonics=1)
                harmonic_index = 0
                for (rot_z0, rot_freqs, rot_powers, found_lines_r, y1) in res:
                    if len(found_lines_r) > 0:
                        r_line_freqs, r_line_powers, _, _ = zip(*found_lines_r)
                    else:
                        r_line_freqs = np.array([])
                        r_line_powers = np.array([])
                    
                    
                    y = y1 + seasonal_avgs
                    #fig, plots = plt.subplots(len(rot_y_fits) + 2, 1, figsize=(6, 2*(len(rot_y_fits)) + 2))
                    #(period_plot) =  plots[0]
                    #period_plot.plot(rot_freqs[1:], rot_powers[0][1:], 'r-', rot_freqs, np.ones(len(rot_freqs))*rot_z0, 'k--')
                    #if len(r_line_freqs > 0):
                    #    period_plot.stem(r_line_freqs, r_line_powers)
                    #(period_plot) =  plots[1]
                    #period_plot.plot(t, y1, 'b+')
                    #for period_index in np.arange(0, len(rot_y_fits)):
                    #    period_fit = rot_powers[period_index]
                    #    (period_plot) = plots[period_index + 2]
                    #    period_plot.plot(rot_freqs[1:], period_fit[1:], 'k--')
                    #plt.savefig("spectra/" + star + '_period_' + str(harmonic_index) + '.png')
                    #plt.close()
                    harmonic_index += 1

 
            ###################################################################
            # Data
            res = prewhiten(star, t, y, freqs, p_values=p_values, max_iters=100, num_indep = 0, white_noise=False, num_bootstrap=num_bootstrap)
            #spec = np.column_stack((freqs, power))
            #np.savetxt("spectra/" + star + ".dat", spec, fmt='%f')
            ###################################################################

            ###################################################################
            # spectral window
            freq_w_range = 10.0*max_freq
            freq_w = 100.0
            freqs_w = np.linspace(freq_w - freq_w_range/2 , freq_w + freq_w_range/2, n_out)
            y_w = np.sin(2*np.pi*freq_w*t)
            res_w = prewhiten(None, t, y_w, freqs_w, p_values=p_values, max_iters=100, num_indep = 0)
            freqs_w -= freq_w
            ###################################################################

            for (z0, _, powers, found_lines, y1), (z0_w, _, powers_w, found_lines_w, _), f1, p_value in zip(res, res_w, f1s, p_values):        
                if len(found_lines) > 0:
                    line_freqs, line_powers, line_freq_stds, line_freq_normalities = zip(*found_lines)
                    line_freqs = np.asarray(line_freqs)
                    line_powers = np.asarray(line_powers)
                    line_freq_stds = np.asarray(line_freq_stds)
                    line_freq_normalities = np.asarray(line_freq_normalities)
                else:
                    line_freqs = np.array([])
                    line_powers = np.array([])
                    line_freq_stds = np.array([])
                    line_freq_normalities = np.array([])
                
                if len(found_lines_w) > 0:
                    line_freqs_w, line_powers_w, _, _ = zip(*found_lines_w)
                    line_freqs_w = np.asarray(line_freqs_w)
                    line_powers_w = np.asarray(line_powers_w)
                else:
                    line_freqs_w = np.array([])
                    line_powers_w = np.array([])

                period_ids = np.where(line_freqs*time_range >= 1.5)
                line_freqs = line_freqs[period_ids]
                line_powers = line_powers[period_ids]
                found_lines = found_lines[period_ids]

                fig, plots = plt.subplots(max(len(line_freqs) + 2, 3), 1, figsize=(6, 2*(max(len(line_freqs) + 2, 3))))
    
                (plot1) = plots[0]            
                (plot2) = plots[1]
                plot1.plot(freqs[1:], powers[0][1:], 'r-', freqs, np.ones(len(freqs))*z0, 'k--')
                if len(line_freqs > 0):
                    plot1.stem(line_freqs, line_powers)
                
                ###################################################################
                line_freqs_w -= freq_w
                powers_w[0] /= len(t)
                line_powers_w /= len(t)

                ###################################################################
                plot2.xaxis.set_ticks(np.arange(-freq_w_range/2, freq_w_range/2, freq_w_range/10))
                plot2.plot(freqs_w, powers_w[0], 'b-',  freqs_w, np.ones(len(freqs_w))*z0_w/len(t), 'k--')
                if len(line_freqs_w > 0):
                   plot2.stem(line_freqs_w, line_powers_w)
                
                freqs_w = np.linspace(-freq_w_range/2, freq_w_range/2, n_out)
                y_w = np.sum(np.exp(np.outer(freqs_w, t*1j)), axis = 1)/len(t)
                powers_w = y_w * y_w.conjugate()
                plot2.plot(freqs_w, powers_w, 'r--')
    
                ###################################################################
    
    
                #y2 = None
                #if (rot_period > 0):
                #    t_fit = np.linspace(min(t), max(t), 1000)
                #    y_fit1 = LombScargle(t, y1).model(t_fit, 365.25/rot_period)
                #    plot2.plot(t_fit, y_fit1, '-')
                #    y_fit1 = LombScargle(t, y).model(t, 365.25/rot_period)
                #    y2 = y - y_fit1
                if (len(line_freqs) == 0):
                    (plot3) = plots[2]
                    plot3.plot(t, y, 'b+')
                else:
                    y1 = y
                    for freq_index in np.arange(0, len(line_freqs)):
                        line_freq = line_freqs[freq_index]
                        print "Freq for " + str(p_value) + " is" + str(line_freq)
                        (plot3) = plots[freq_index + 2]
                        plot3.plot(t, y1, 'b+')
                        t_fit = np.linspace(min(t), max(t), 1000)
                        y_fit = LombScargle(t, y1).model(t_fit, line_freq)
                        plot3.plot(t_fit, y_fit, '-')
                        y_fit = LombScargle(t, y1).model(t, line_freq)
                        y1 = y1 - y_fit
    
                #if y2 != None:
                #    (z0, power, periods, peaks, _) = prewhiten(t, y2, freqs, p_value=0.01, max_iters=100, num_indep = 0)
                #    spec = np.column_stack((freqs, power))
                #    #np.savetxt("spectra/" + star + ".dat", spec, fmt='%f')
                #    plot3.plot(freqs, power, 'r-', freqs, np.ones(len(freqs))*z0, 'k--')
                #    if len(periods > 0):
                #        plot3.stem(periods, peaks)
                    
                fig.savefig(spectra_path+str(p_value)+"/" + star + '.png')
                plt.close(fig)
                f1.write(star + " " + (' '.join(['%s %s %s' % (1/f, 1/(f - std) - 1/(f + std), n) for f, p, std, n in found_lines])) + "\n")
                f1.flush()
