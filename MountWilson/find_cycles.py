# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 11:48:48 2017

@author: olspern1
"""

from astropy.stats import LombScargle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
import random

import os
import os.path
import mw_utils
from prewhitener import Prewhitener
from BGLST import BGLST
from bayes_lin_reg import bayes_lin_reg

num_resamples = 1000
num_bootstrap = 0
#input_path = "cleaned"
input_path = "BGLST_input"
if input_path == "cleaned":
    skiprows = 1
    remove_rotation = True
else:
    skiprows = 0
    remove_rotation = False
    
spectra_path = "new_spectra"
save_cleaned = False
p_values = np.array([0.01])#, 0.05])
min_freq = 0.001
max_freq = 0.5
n_out = 1000
offset = 1979.3452

star_name = None
if len(sys.argv) > 1:
    star_name = sys.argv[1]

rot_periods = mw_utils.load_rot_periods()

def constant_model(y, w, y_test, w_test):
    W = sum(w)
    wy_arr = w * y

    Y = sum(wy_arr)

    sigma_beta = 1.0 / W
    mu_beta = Y * sigma_beta

    norm_term = sum(np.log(np.sqrt(w)) - np.log(np.sqrt(2.0*np.pi)))

    y_model = mu_beta
    loglik = norm_term - 0.5 * sum(w_test * (y_test - y_model)**2)

    return (mu_beta, sigma_beta, y_model, loglik)


f_trends = map(lambda p_value: open(spectra_path+str(p_value)+'/trends.txt', 'w'), p_values)
f1s= map(lambda p_value: open(spectra_path+str(p_value)+'/results.txt', 'w'), p_values)
f2s= map(lambda p_value: open(spectra_path+str(p_value)+'/rot_var_reduction.txt', 'w'), p_values)
for root, dirs, files in os.walk(input_path):
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
            if star_name != None and star != star_name:
                continue
            (rot_period, _) = (0, 0)
            if (rot_periods.has_key(star)):
                (rot_period, _) = rot_periods[star]
            #print star + " period is " + str(rot_period)
            if star != "201091":
                continue
            data = np.loadtxt(input_path+"/"+file, usecols=(0,1), skiprows=skiprows)
            print "Finding cycles for " + star
            normval = data.shape[0]
            freqs = np.linspace(min_freq, max_freq, n_out)
            t_orig = data[:,0]
            y_orig = data[:,1]
            indices = np.argsort(t_orig)
            t_orig = t_orig[indices]            
            y_orig = y_orig[indices]

            t = t_orig/365.25 + offset

            #indices = np.where(t < 1992)[0]
            #y_orig = y_orig[indices]
            #t_orig = t_orig[indices]
            #t = t_orig/365.25 + offset
            y = y_orig
            
            time_range = max(t) - min(t)
            # removing rotational modulation
            if not remove_rotation:
                rot_period = 0
            if (rot_period > 0):
                seasonal_avgs = mw_utils.get_seasonal_means_per_point(t, y)
                
                y1 = y - seasonal_avgs
                rot_freq = 365.25/rot_period

                rot_freqs = np.linspace(rot_freq-rot_freq/5.0, rot_freq+rot_freq/5.0, 10000)
                prewhitener = Prewhitener(star, "LS", t, y1, rot_freqs, num_resamples=num_resamples, num_bootstrap = 0, spectra_path = None, max_iters=100, max_peaks=1, num_indep = 0, white_noise=True)
                res = prewhitener.prewhiten_with_higher_harmonics(p_value=0.01, num_higher_harmonics=0)
                harmonic_index = 0
                assert(len(res) == 1) # There is a for loop, but we currently assume only one result
                for (rot_powers, found_lines_r, y_res, _) in res:
                    if len(found_lines_r) > 0:
                        r_line_freqs, r_line_powers, _, _, _ = zip(*found_lines_r)
                    else:
                        r_line_freqs = np.array([])
                        r_line_powers = np.array([])
                    
                    
                    y = y_res + seasonal_avgs
                    harmonic_index += 1
                    var_reduction = abs(np.var(y_orig) - np.var(y))/np.var(y_orig)
                    print "Variance reduction after rot. period removal: ", var_reduction
                
                fig, (period_plot, ts_plot) = plt.subplots(2, 1, figsize=(20, 8))
                period_plot.set_title(str(rot_period))
                period_plot.plot(t, y1, 'r+', t, y_res+0.2, 'k+')
                ts_plot.scatter(t, y, c='r', marker='o', )
                ts_plot.scatter(t, seasonal_avgs, c='b', marker='x')
                fig.savefig("temp/" + star + '_rot_period.png')
                plt.close()

                #rot_power = LombScargle(t, y1, nterms=1).power(rot_freqs, normalization='psd')#/np.var(y)
                #rot_power_res = LombScargle(t, y_res, nterms=1).power(rot_freqs, normalization='psd')#/np.var(y)

                #fig, period_plot = plt.subplots(1, 1, figsize=(20, 4))
                #period_plot.set_title(str(rot_freq))
                #period_plot.plot(rot_freqs, rot_power, 'b-')
                #period_plot.plot(rot_freqs, rot_power_res, 'r--')
                #plt.savefig("temp/" + star + '_period_spec.png')
                #plt.close()
            else:
                var_reduction = 0
            if save_cleaned:
                dat = np.column_stack((t_orig, y))
                np.savetxt("cleaned_wo_rot/" + star + ".dat", dat, fmt='%f')


            seasons = mw_utils.get_seasons(zip(t, y), 1.0, True)

            ###################################################################
            # Data
            res = Prewhitener(star, "BGLST", t, y, freqs, max_iters=100, num_resamples=0, num_bootstrap=0, num_indep = 0, white_noise=False, max_peaks=1).prewhiten(p_values=p_values)
            #spec = np.column_stack((freqs, power))
            #np.savetxt("spectra/" + star + ".dat", spec, fmt='%f')
            ###################################################################


            for (powers, found_lines, _, _), f1, f2, f_trend, p_value in zip(res, f1s, f2s, f_trends, p_values):
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
                
                
                ###############################################################
                # Significance differences of trend lines
                
                if len(line_freqs) > 0:
                    line_freq = line_freqs[0]
                    if line_freq > 0:
                        noise_var = mw_utils.get_seasonal_noise_var(t, y)
                        w = np.ones(len(t))/noise_var
                        
                        slope, intercept, _, _, _ = stats.linregress(t, y)
                        duration = max(t) - min(t)
                        _, (_, _, mu_alpha, mu_beta), _, y_fit, _, _ = BGLST(t, y, w, 
                                        w_A = 2.0/np.var(y), A_hat = 0.0,
                                        w_B = 2.0/np.var(y), B_hat = 0.0,
                                        w_alpha = duration**2 / np.var(y), alpha_hat = slope, 
                                        w_beta = 1.0 / (np.var(y) + intercept**2), beta_hat = intercept).model(line_freq)
                        
                        seasonal_means = mw_utils.get_seasonal_means(t, y)
                        seasonal_noise_var = mw_utils.get_seasonal_noise_var(t, y, False)
                        seasonal_weights = np.ones(len(seasonal_noise_var))/seasonal_noise_var
                                                
                        norm_term = sum(np.log(np.sqrt(seasonal_weights)) - np.log(np.sqrt(2.0*np.pi)))

                        y_model = seasonal_means[:,0] * mu_alpha + mu_beta
                        loglik_seasons = norm_term - 0.5 * sum(seasonal_weights * (seasonal_means[:,1] - y_model)**2)
                        
                        (mu_alpha_null, mu_beta_null), _, _, loglik_seasons_null = bayes_lin_reg(t, y, w, seasonal_means[:,0], seasonal_means[:,1], seasonal_weights)
                        mu_beta_null2, _, _, loglik_seasons_null2 = constant_model(y, w, seasonal_means[:,1], seasonal_weights)
                        
                        log_n = np.log(np.shape(seasonal_means)[0])
                        bic = log_n * 2 - 2.0*loglik_seasons
                        bic_null = log_n  - 2.0*loglik_seasons_null
                        bic_null2 = log_n  - 2.0*loglik_seasons_null2
                        
                        #print bic_null, bic
                        #print "trend delta_bic:", star, delta_bic, bic_null, bic, mu_alpha, mu_alpha_null, mu_beta, mu_beta_null
                        fig_trend, ax_trend = plt.subplots(1, 1, figsize=(7, 6))
                        ax_trend.plot(t, y, 'k+',lw=0.5) 
                        t_fit = np.linspace(min(t), max(t), 1000)
                        ax_trend.plot(t_fit, t_fit * mu_alpha + mu_beta, 'r-')
                        ax_trend.plot(t_fit, t_fit * mu_alpha_null + mu_beta_null, 'b--')
                        ax_trend.plot(t_fit, np.ones(len(t_fit))*mu_beta_null2, 'g-.')
                        fig_trend.savefig(spectra_path+str(p_value)+"/" + star + '_trend.png')

                        delta_bic1 = bic_null - bic
                        delta_bic2 = bic_null2 - bic
                        if abs(delta_bic1) >= 6.0 and abs(delta_bic2) >= 6.0:
                            # Significant trend
                            f_trend.write(star + "\n")
                        else:
                            f_trend.write("-" + star + "\n")
                        f_trend.flush()
                        
                ###############################################################                
                
                omitted_freq_ids = np.where(line_freqs*time_range < 1.5)
                omitted_line_freqs = line_freqs[omitted_freq_ids]
                # filter out longer periods than 2/3 of the data_span
                filtered_freq_ids = np.where(line_freqs*time_range >= 1.5)
                line_freqs = line_freqs[filtered_freq_ids]
                line_powers = line_powers[filtered_freq_ids]
                found_lines = np.asarray(found_lines)[filtered_freq_ids]
                powers = np.asarray(powers)[filtered_freq_ids]

                fig, plots = plt.subplots(max(2 * len(line_freqs), 2), 1, figsize=(6, 2*(max(2 * len(line_freqs), 2))))
    
                fig_pdf, ax_pdf = plt.subplots(1, 1)
                fig_pdf.set_size_inches(9, 5)
                ax_pdf.set_ylabel(r'S-index', fontsize=15)
                #ax_pdf.set_xlabel(r'$t$ [yr]', fontsize=15)
                ax_pdf.text(0.95, 0.9,'(a)', horizontalalignment='center', transform=ax_pdf.transAxes, fontsize=15)
                ax_pdf.set_xlim([min(t), max(t)])
            
    
                #(plot1) = plots[0]
                #plot1.plot(freqs[1:], powers[0][1:], 'r-')

                for freq_index in np.arange(0, len(line_freqs)):
                    line_freq = line_freqs[freq_index]
                    line_power = line_powers[freq_index]
                    z0 = z0s[freq_index]

                    #plot1.stem([line_freq], [line_power])
                    #plot1.plot(freqs, np.ones(len(freqs))*z0, 'k--')
                    power = powers[freq_index]
                    (plot1) = plots[freq_index]
                    #(plot1) = plots[freq_index + 1]
                    plot1.plot(freqs[1:], power[1:], 'r-')
                    if line_freq > 0:
                        plot1.plot([line_freq, line_freq], [min(power), line_power], 'k--')
                
                plot_index = max(1, len(line_freqs))

                if (len(line_freqs) == 0):
                    (plot2) = plots[plot_index]
                    plot2.plot(t, y, 'b+')
                else:
                    y1 = y
                    for freq_index in np.arange(0, len(omitted_line_freqs)):
                        line_freq = omitted_line_freqs[freq_index]
                        if line_freq > 0:
                            print "Removing too long cycle: ", 1.0/line_freq
                            noise_var = mw_utils.get_seasonal_noise_var(t, y1)
                            w1 = np.ones(len(t))/noise_var
                            _, _, _, y_fit, _, _ = BGLST(t, y1, w1).model(line_freq)
                            y1 = y1 - y_fit
                        
                    for freq_index in np.arange(0, len(line_freqs)):
                        noise_var = mw_utils.get_seasonal_noise_var(t, y1)
                        w1 = np.ones(len(t))/noise_var
                        (plot2) = plots[freq_index + plot_index]
                        plot2.plot(t, y1, 'b+')
                        ax_pdf.plot(t, y1, 'k+',lw=0.5) 
                        line_freq = line_freqs[freq_index]
                        if line_freq > 0:
                            t_fit = np.linspace(min(t), max(t), 1000)
                            w_fit = np.ones(len(t_fit))/mw_utils.get_test_point_noise_var(t, y1, t_fit)
                            _, _, _, y_fit, _, pred_var = BGLST(t, y1, w1).model(line_freq, t_fit, w_fit, calc_pred_var=True)
                            plot2.plot(t_fit, y_fit, '-')
                            ax_pdf.plot(t_fit, y_fit, 'r-', lw=2)
                            std2 = np.sqrt(pred_var)*2.0
                            ax_pdf.fill_between(t_fit, y_fit-std2, y_fit+std2, alpha=0.8, facecolor='lightsalmon', interpolate=True)
                            _, _, _, y_fit, _, _ = BGLST(t, y1, w1).model(line_freq)
                            y1 = y1 - y_fit
                        
    
                fig.savefig(spectra_path+str(p_value)+"/" + star + '.png')
                plt.close(fig)
                fig_pdf.savefig(spectra_path+str(p_value)+"/" + star + '.pdf')
                plt.close(fig_pdf)
                for f, p, std, normality, z0 in found_lines:
                    f1.write(star + ' %s %s %s %s' % (f, std, normality, z0) + "\n")
                    
                #f1.write(star + " " + ('\n'.join(['%s %s %s %s' % (f, std, normality, z0) for f, p, std, normality, z0 in found_lines])) + "\n")
                #f1.write(star + " " + (' '.join(['%s %s %s %s' % (1.0/f, std/f/f, normality, z0) for f, p, std, normality, z0 in found_lines])) + "\n")
                f1.flush()
                f2.write(star + " " + str(var_reduction) + "\n")
                f2.flush()
