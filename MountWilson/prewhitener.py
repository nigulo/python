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
import random
import mw_utils
from BGLST import BGLST
from bayes_lin_reg import bayes_lin_reg
from scipy.signal import argrelextrema
import scipy

bic_threshold = 2.0
seasonal_significance = True

class Prewhitener:
    def __init__(self, name, type, t, y, freqs, num_resamples=1000, num_bootstrap = 0, 
                 spectra_path = None, max_iters=100, max_peaks=1, num_indep=0, white_noise=False):
        self.name = name
        self.type = type
        self.t = np.array(t)
        self.y = np.array(y)
        self.freqs = np.array(freqs)
        self.num_resamples = num_resamples
        self.num_bootstrap = num_bootstrap
        self.spectra_path = spectra_path
        self.max_iters = max_iters
        self.max_peaks = max_peaks
        self.num_indep = num_indep
        self.white_noise = white_noise
        assert(max_peaks >= 1)
        
    def prewhiten_with_higher_harmonics(self, p_value=0.01, num_higher_harmonics=0):
        res = list()
        for i in np.arange(1, num_higher_harmonics+2):
            r = self.prewhiten(np.array([p_value]))
            [(powers, peaks, y_res, total_bic)] = r
            self.y = y_res
            res.append(r[0])
        return res
    
    def calcFapLevel(self, y, p_values):
        if self.white_noise == True:
            if self.num_indep == 0:
                num_indep = len(y)
            z0s = -np.log(np.ones_like(len(p_values))-np.power(np.ones_like(len(p_values))-p_values, 1.0/num_indep))
        else:
            nh_powers = list()
            #fig_nh, (plot_nh) = plt.subplots(1, 1, figsize=(6, 3))
            num_indep = 1
            seasons = mw_utils.get_seasons(zip(self.t, y), 1.0, True)
            for nh_ind in np.arange(0, self.num_resamples):
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
                power = LombScargle(self.t, y_nh, nterms=1).power(self.freqs, normalization='psd')#/np.var(y)
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
            z0_indexes = p_values * self.num_resamples * num_indep
            assert(np.all(z0_indexes >= 1))
            z0s = nh_powers[(-np.floor(z0_indexes)).astype(int)]
        return z0s

    def prewhiten_step(self, y, z0, p_value, prev_mse, peak_index=0, iter_index=0, total_bic = 0):
        results = dict()
        specs=[]
        significant_lines=[]
        if self.type == "LS":
            (best_freq, max_power, power, residue, y_fit) = ls(self.t, y, self.freqs, z0, peak_index)
            sigma = 0
            z0_or_bic = z0
        elif self.type == "BGLST":
            (best_freq, max_power, power, residue, y_fit, bic, sigma) = self.bglst(self.t, y, self.freqs, p_value, peak_index, iter_index)
            total_bic += bic
            z0_or_bic = bic
        else:
            raise Exception("Unsupported type")
        mse = sum(residue**2)/len(residue)
        if best_freq > 0 and prev_mse >= 0 and prev_mse <= mse:
            print "MSE Incresed, probably wrong frequency subracted", iter_index, peak_index
            best_freq = 0
        if best_freq > 0:
            #print z0, p_value, i, best_freq, max_power
            if self.num_bootstrap > 0:
                assert(self.type == "LS")
                bs_freqs = list()
                for j in np.arange(0, self.num_bootstrap):
                    y_bs = y_fit + residue[np.random.choice(np.shape(residue)[0], np.shape(residue)[0], replace=True, p=None)]
                    if self.type == "LS":
                        (best_freq_bs, _, _, _, _) = ls(self.t, y_bs, self.freqs, z0_or_bic, peak_index)
                    else:
                        raise Exception("Unsupported type")
                    if best_freq_bs > 0:
                        bs_freqs.append(best_freq_bs)
                (skewKurt, normality) = stats.normaltest(bs_freqs) # normality is p-value
                if self.spectra_path != None:
                    plt.hist(bs_freqs, self.num_bootstrap/10)
                    plt.savefig(self.spectra_path+str(p_value)+"/" + self.name + "_" + str(best_freq) + '.png')
                    plt.close()
                            
                significant_lines.append((best_freq, max_power, np.std(bs_freqs), normality, z0_or_bic))
            else:
                significant_lines.append((best_freq, max_power, sigma, 0, z0_or_bic))
            specs.append(power)
            if iter_index + 1 < self.max_iters:
                if self.type == "LS" :
                    z0 = self.calcFapLevel(residue, np.array([p_value]))[0]
                    if self.white_noise == True:
                        z0 *= np.var(self.y) # scale it intentionally with the original variance
                for peak_index_1 in np.arange(0, self.max_peaks):
                    sub_results = self.prewhiten_step(residue, z0, p_value, mse, peak_index_1, iter_index + 1, total_bic)
                    for sub_result_key in sub_results.keys():
                        sub_specs, sub_significant_lines, sub_residue, sub_total_bic = sub_results[sub_result_key] 
                        results[str(peak_index) + "," + sub_result_key] = (specs + sub_specs, significant_lines + sub_significant_lines, sub_residue, sub_total_bic)
            else:
                results[str(peak_index)] = (specs, significant_lines, residue, total_bic)
                
        else:
            specs = [power]
            significant_lines = [(0, 0, 0, 0, z0_or_bic)]
            results[str(peak_index)] = (specs, significant_lines, residue, total_bic)
        return results
    
    
    def prewhiten(self, p_values=np.array([0.01, 0.05])):
        z0s = np.zeros(len(p_values))    
        if self.type == "LS" :
            z0s = self.calcFapLevel(self.y, p_values)
            if self.white_noise == True:
                z0s *= np.var(self.y)
        res = list()
        for z0, p_value in zip(z0s, p_values):
            min_mse = -1.0
            max_bic = None
            max_bic = None
            for peak_index in np.arange(0, self.max_peaks):
                for (key, (specs, significant_lines, residue, total_bic)) in self.prewhiten_step(self.y, z0, p_value, -1.0, peak_index).items():
                    mse = sum(residue**2)/len(residue)
                    num_lines_found = np.shape(specs)[0] - 1
                    print "Result: ", "<" + key +">",  z0, num_lines_found, mse, total_bic
                    if self.type == "LS": # Not good solution at all
                        if (min_mse < 0 or mse < min_mse):
                            best_res = (key, (specs, significant_lines, residue, total_bic))
                            min_mse = mse
                    elif self.type == "BGLST":
                        #if (max_bic is None or total_bic/num_lines_found > max_bic):
                        #    best_res = (key, (specs, significant_lines, residue, total_bic))
                        #    max_bic = total_bic/num_lines_found
                        if (max_bic is None or total_bic - bic_threshold * num_lines_found > max_bic):
                            best_res = (key, (specs, significant_lines, residue, total_bic))
                            max_bic = (num_lines_found, total_bic)
                    else:
                        raise Exception("Unsupported type")
            key, best_res_data = best_res
            (specs, significant_lines, residue, total_bic) = best_res_data
            print "Best result:", "<" + key +">", z0, np.shape(specs)[0], min_mse, max_bic
            res.append(best_res_data)
        return res

    def bglst(self, t, y, freqs_in, p_value, peak_index, iter_index):
        noise_var = mw_utils.get_seasonal_noise_var(t, y)
        w = np.ones(len(t))/noise_var
        slope, intercept, r_value, p_value, std_err = stats.linregress(t, y)
        duration = max(t) - min(t)
        bglst = BGLST(t, y, w, 
                        w_A = 2.0/np.var(y), A_hat = 0.0,
                        w_B = 2.0/np.var(y), B_hat = 0.0,
                        w_alpha = duration**2 / np.var(y), alpha_hat = slope, 
                        w_beta = 1.0 / (np.var(y) + intercept**2), beta_hat = intercept)
        
        (freqs, power) = bglst.calc_all(freqs_in[0], freqs_in[-1], len(freqs_in))
        max_power_ind = get_max_power_ind(power, peak_index)
        if max_power_ind >= 0:
            max_power = power[max_power_ind]
            best_freq = freqs[max_power_ind]
            tau, (A, B, alpha, beta), _, y_model, loglik = bglst.model(best_freq)
    
            condition = False
            if seasonal_significance:
                seasonal_means = mw_utils.get_seasonal_means(t, y)
                seasonal_noise_var = mw_utils.get_seasonal_noise_var(t, y, False)
                seasonal_weights = np.ones(len(seasonal_noise_var))/seasonal_noise_var
                bglst = BGLST(seasonal_means[:,0], seasonal_means[:,1], seasonal_weights)
                _, loglik_seasons = bglst.fit(tau, best_freq, A, B, alpha, beta)
                #seasonal_weights_null = np.ones(len(seasonal_noise_var))/(seasonal_noise_var + np.var(seasonal_means[:,1]))
                _, _, _, loglik_seasons_null = bayes_lin_reg(seasonal_means[:,0], seasonal_means[:,1], seasonal_weights)
                log_n = np.log(np.shape(seasonal_means)[0])
                bic = log_n * 5 - 2.0*loglik_seasons
                bic_null = log_n * 2 - 2.0*loglik_seasons_null
                
                print bic_null, bic
                delta_bic = bic_null - bic
                condition = delta_bic >= bic_threshold#6.0:#10.0:#np.log(1.0/p_value):
            else:
                t_left_inds = np.where(t<1980.0)
                t_right_inds = np.where(t>=1980.0)
                t_left = t[t_left_inds]
                y_left = y[t_left_inds]
                t_right = t[t_right_inds]
                y_right = y[t_right_inds]
    
                #print "Num left, right", min(t_left), max(t_left), min(t_right), max(t_right), len(t_left), len(t_right)
                if len(t_left) > 0: 
                    seasonal_noise_var = mw_utils.get_seasonal_noise_var(t_left, y_left)
                    seasonal_weights = np.ones(len(seasonal_noise_var))/seasonal_noise_var
                    bglst = BGLST(t_left, y_left, seasonal_weights)
                    _, loglik_left = bglst.fit(tau, best_freq, A, B, alpha, beta)
                    #seasonal_weights_null = np.ones(len(seasonal_noise_var))/(seasonal_noise_var + np.var(seasonal_means[:,1]))
                    _, _, _, loglik_left_null = bayes_lin_reg(t_left, y_left, seasonal_weights)
    
                    log_n = np.log(len(t_left))
                    bic = log_n * 5 - 2.0*loglik_left
                    bic_null = log_n * 2 - 2.0*loglik_left_null
                    delta_bic = bic_null - bic
                    left_condition = delta_bic >= bic_threshold#6.0:#10.0:#np.log(1.0/p_value):
                else:
                    left_condition = True
    
                if len(t_right) > 0: 
                    seasonal_noise_var = mw_utils.get_seasonal_noise_var(t_right, y_right)
                    seasonal_weights = np.ones(len(seasonal_noise_var))/seasonal_noise_var
                    bglst = BGLST(t_right, y_right, seasonal_weights)
                    _, loglik_right = bglst.fit(tau, best_freq, A, B, alpha, beta)
                    #seasonal_weights_null = np.ones(len(seasonal_noise_var))/(seasonal_noise_var + np.var(seasonal_means[:,1]))
                    _, _, _, loglik_right_null = bayes_lin_reg(t_right, y_right, seasonal_weights)
    
                    log_n = np.log(len(t_right))
                    bic = log_n * 5 - 2.0*loglik_right
                    bic_null = log_n * 2 - 2.0*loglik_right_null
                    delta_bic = bic_null - bic
                    right_condition = delta_bic >= bic_threshold#6.0:#10.0:#np.log(1.0/p_value):
                else:
                    right_condition = True
                
                condition = left_condition and right_condition
    
            if condition:
                bglst_m = BGLST(t, y_model, np.ones(len(t))/np.var(y))
                (freqs_m, log_probs_m) = bglst_m.calc_all(freqs_in[0], freqs_in[-1], len(freqs_in))
                log_probs_m -= scipy.misc.logsumexp(log_probs_m)
                probs_m = np.exp(log_probs_m)
                probs_m /= sum(probs_m)
                #mean = np.exp(scipy.misc.logsumexp(np.log(freqs_m)+probs_m))
                #sigma = np.sqrt(np.exp(scipy.misc.logsumexp(2*np.log(freqs_m-best_freq) + probs_m)))
                mean = sum(freqs_m*probs_m)
                sigma = np.sqrt(sum((freqs_m-best_freq)**2 * probs_m))
                
                fig, (model_plot) = plt.subplots(1, 1, figsize=(20, 8))
                model_plot.plot(freqs_m, log_probs_m)
                fig.savefig("temp/" + self.name + '_' +  str(iter_index) + '_model.png')
                plt.close()
                
                #mean = sum(freqs_m*probs_m)/sum(probs_m)
                print "Maximum:", peak_index, best_freq, max_power, delta_bic, sigma, mean
                return (best_freq, max_power, power, y - y_model, y_model, delta_bic, sigma)
            else:
                print "Insignificant maximum:", peak_index, best_freq, max_power, delta_bic
                return (0, 0, power, y, np.array([]), 0, 0)
        else:
            return (0, 0, power, y, np.array([]), 0, 0)

def get_max_power_ind(power, peak_index):
    local_maxima_inds = mw_utils.find_local_maxima(power)
    #if peak_index == 0:
    #    #max_power_ind = np.argmax(power)
    #    maximum_ind = np.argmax(power[local_maxima_inds])
    if peak_index > 0:
        #local_maxima_inds = argrelextrema(power, np.greater)
        local_maxima = power[local_maxima_inds]
        if len(local_maxima) <= peak_index:
            return -1
        for i in np.arange(0, peak_index):
            maximum_ind = np.argmax(power[local_maxima_inds])
            local_maxima_inds = np.delete(local_maxima_inds, maximum_ind)
    maximum_ind = np.argmax(power[local_maxima_inds])
    return local_maxima_inds[maximum_ind]
    
def ls(t, y, freqs, z0, peak_index):
    power = LombScargle(t, y, nterms=1).power(freqs, normalization='psd')#/np.var(y)
    max_power_ind = get_max_power_ind(power, peak_index)
    if max_power_ind >= 0:
        max_power = power[max_power_ind]
        if max_power <= z0:
            return (0, 0, power, y, np.array([]))
        best_freq = freqs[max_power_ind]
        y_fit = LombScargle(t, y, nterms=1).model(t, best_freq)
        return (best_freq, max_power, power, y - y_fit, y_fit)
    else:
        return (0, 0, power, y, np.array([]))

