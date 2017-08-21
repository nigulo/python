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
import mw_utils
from PGLST import PGLST


class Prewhitener:
    

def prewhiten_with_higher_harmonics(star, t, y1, freqs1, p_value=0.01, max_iters=100, num_indep=0, white_noise=True, num_higher_harmonics=0):
    y = y1
    res = list()
    for i in np.arange(1, num_higher_harmonics+2):
        freqs = freqs1 * i
        r = prewhiten(star, t, y, freqs, np.array([p_value]), max_iters, num_indep, white_noise, num_bootstrap=0)
        [(freqs, powers, peaks, y_res)] = r
        y = y_res
        res.append(r[0])
    return res

def calcFapLevel(star, t, y, freqs, p_values, num_indep, white_noise, num_resamples=1000):
    if white_noise == True:
        if num_indep == 0:
            num_indep = len(y)
        z0s = -np.log(np.ones_like(len(p_values))-np.power(np.ones_like(len(p_values))-p_values, 1.0/num_indep))
    else:
        nh_powers = list()
        #fig_nh, (plot_nh) = plt.subplots(1, 1, figsize=(6, 3))
        num_indep = 1
        seasons = mw_utils.get_seasons(zip(t, y), 1.0, True)
        for nh_ind in np.arange(0, num_resamples):
            ###################################################################
            ## Just shuffle everything
            #y_bs = y1[np.random.choice(np.shape(y1)[0], np.shape(y1)[0], replace=True, p=None)]
            ###################################################################
            ## Shuffle first seasons then within each season
            resampled_data = list()
            for s in mw_utils.resample_seasons(seasons):
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

    z0s = calcFapLevel(star, t, y, freqs, p_values, num_indep, white_noise)
   
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
                                
                    significant_lines.append((best_freq, max_power, np.std(bs_freqs), normality, z0))
                else:
                    significant_lines.append((best_freq, max_power, 0, 0, z0))
                specs.append(power)
                y = residue
            else:
                specs.append(power)
                break
            z0 = calcFapLevel(star, t, y, freqs, np.array([p_value]), num_indep, white_noise)[0]
        res.append((freqs, specs, np.asarray(significant_lines), y * std + mean))
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

def pglst(t, y, freqs, z0):
    pglst = PGLST(t, y, w)
    (freqs, power) = pglst.calc_all(freqs[0], freqs[-1], len(freqs))
    max_power_ind = np.argmax(power[1:-1])
    if max_power_ind >= 0:
        max_power = power[max_power_ind+1]
        #print freqs[max_power_ind+1], max_power, z0
        if max_power <= z0:
            return (0, 0, power, np.array([]), np.array([]))
        best_freq = freqs[max_power_ind+1]
        tau, (A, B, alpha, beta), _ = pglst.model(best_freq)
        t_model = t
        y_model = np.cos(t_model * 2.0 * np.pi * best_freq - tau) * A  + np.sin(t_model * 2.0 * np.pi * best_freq - tau) * B + t_model * alpha + beta
        return (best_freq, max_power, power, y - y_model, y_model)
    else:
        return (0, 0, power, np.array([]), np.array([]))
