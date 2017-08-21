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

import os
import os.path
import mw_utils
from prewhitener import Prewhitener

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

rot_periods = mw_utils.load_rot_periods()

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
                seasonal_avgs = mw_utils.get_seasonal_avgs(zip(t, y), 1)
                y1 = y - seasonal_avgs
                rot_freq = 365.25/rot_period
                rot_freqs = np.linspace(rot_freq-rot_freq/5, rot_freq+rot_freq/5, n_out)
                res = Prewhitener(star, "LS", t, y1, rot_freqs, sigma=None, max_iters=100).prewhiten_with_higher_harmonics(p_value=0.01, num_indep = 0, white_noise=True, num_higher_harmonics=1)
                harmonic_index = 0
                for (rot_freqs, rot_powers, found_lines_r, y1) in res:
                    if len(found_lines_r) > 0:
                        r_line_freqs, r_line_powers, _, _, _ = zip(*found_lines_r)
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
            res = Prewhitener(star, "LS", t, y, freqs, sigma=None, max_iters=100, num_bootstrap=num_bootstrap).prewhiten(p_values=p_values, num_indep = 0, white_noise=False)
            #spec = np.column_stack((freqs, power))
            #np.savetxt("spectra/" + star + ".dat", spec, fmt='%f')
            ###################################################################


            for (_, powers, found_lines, y1), f1, p_value in zip(res, f1s, p_values):        
                if len(found_lines) > 0:
                    line_freqs, line_powers, line_freq_stds, line_freq_normalities, z0s = zip(*found_lines)
                    line_freqs = np.asarray(line_freqs)
                    line_powers = np.asarray(line_powers)
                    line_freq_stds = np.asarray(line_freq_stds)
                    line_freq_normalities = np.asarray(line_freq_normalities)
                    z0s = np.asarray(z0s)
                else:
                    line_freqs = np.array([])
                    line_powers = np.array([])
                    line_freq_stds = np.array([])
                    line_freq_normalities = np.array([])
                    z0s = np.array([])
                
                # filter out longer periods than 2/3 of the data_span
                filtered_freq_ids = np.where(line_freqs*time_range >= 1.5)
                line_freqs = line_freqs[filtered_freq_ids]
                line_powers = line_powers[filtered_freq_ids]
                found_lines = found_lines[filtered_freq_ids]

                fig, plots = plt.subplots(max(2 * len(line_freqs), 2), 1, figsize=(6, 2*(max(2 * len(line_freqs), 2))))
    
                (plot1) = plots[0]
                plot1.plot(freqs[1:], powers[0][1:], 'r-')

                for freq_index in np.arange(0, len(line_freqs)):
                    line_freq = line_freqs[freq_index]
                    line_power = line_powers[freq_index]
                    z0 = z0s[freq_index]
                    plot1.stem([line_freq], [line_power])
                    plot1.plot(freqs, np.ones(len(freqs))*z0, 'k--')
                    if freq_index < len(line_freqs) - 1:
                        power = powers[freq_index + 1]
                        (plot1) = plots[freq_index + 1]
                        plot1.plot(freqs[1:], power[1:], 'r-')
                
                plot_index = max(1, len(line_freqs))

                if (len(line_freqs) == 0):
                    (plot2) = plots[plot_index]
                    plot2.plot(t, y, 'b+')
                else:
                    y1 = y
                    for freq_index in np.arange(0, len(line_freqs)):
                        line_freq = line_freqs[freq_index]
                        (plot2) = plots[freq_index + plot_index]
                        plot2.plot(t, y1, 'b+')
                        t_fit = np.linspace(min(t), max(t), 1000)
                        y_fit = LombScargle(t, y1).model(t_fit, line_freq)
                        plot2.plot(t_fit, y_fit, '-')
                        y_fit = LombScargle(t, y1).model(t, line_freq)
                        y1 = y1 - y_fit
    
                fig.savefig(spectra_path+str(p_value)+"/" + star + '.png')
                plt.close(fig)
                f1.write(star + " " + (' '.join(['%s %s %s' % (1/f, 1/(f - std) - 1/(f + std), n) for f, p, std, n, z0 in found_lines])) + "\n")
                f1.flush()
