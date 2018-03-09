# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 13:29:14 2017

@author: olspern1
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import BGLST
import scipy.stats
import os
import os.path
from scipy import stats
from astropy.stats import LombScargle
import mw_utils
import sys
import pickle

plot_both_metrics = False

setup_selected = None
if len(sys.argv) > 1:
    setup_selected = int(sys.argv[1])

axis_label_fs = 15
panel_label_fs = 15

offset = 1979.3452
real_dats = []

linestyles = ['-', '--', '-.']
linecolors = ['r', 'b', 'g']
errorcolors = ['lightsalmon', 'lightblue', 'lightgreen']

iterative = True

num_points = 1000

for root, dirs, dir_files in os.walk("cleaned_wo_rot"):
    for file in dir_files:
        if file[-4:] == ".dat":
            star = file[:-4]
            star = star.upper()
            if (star[-3:] == '.CL'):
                star = star[0:-3]
            if (star[0:2] == 'HD'):
                star = star[2:]
            if star == "SUNALL":
                continue
            dat = np.loadtxt("cleaned_wo_rot/"+file, usecols=(0,1), skiprows=0)
            t = dat[:,0]
            y = dat[:,1]
            t /= 365.25
            t += offset
            noise_var = mw_utils.get_seasonal_noise_var(t, y)

            down_sample_factor = 1#int(np.floor(len(t)/num_points))
            if down_sample_factor >= 2:
                indices = np.random.choice(len(t), len(t)/down_sample_factor, replace=False, p=None)
                indices = np.sort(indices)
                t = t[indices]
                y = y[indices]
                noise_var = noise_var[indices]
    
            if len(t) >=100 and max(t) - min(t) >= 30:

                #print noise_var
                #if len(np.where(noise_var == noise_var[0])[0]) == len(noise_var):
                #    print star

                real_dats.append((t, y, noise_var, star))

def get_local_noise_var(t, y, window_size=1.0):
    total_var = np.var(y)
    seasons = mw_utils.get_seasons(zip(t, y), window_size, False)
    noise_var = np.zeros(len(t))
    i = 0
    max_var = 0
    for season in seasons:
        if np.shape(season[:,1])[0] < 5:
            #print "OOOPS"
            #var = -1#total_var # Is it good idea?
            #print "OOPS"
            var = total_var # Is it good idea?
        else:
            var = np.var(season[:,1])
        max_var=max(max_var, var)
        season_len = np.shape(season)[0]
        for j in np.arange(i, i + season_len):
            noise_var[j] = var
        i += season_len
    #for j in np.arange(0, len(t)):
    #    if noise_var[j] < 0:
    #        noise_var[j] = max_var
    assert(i == len(noise_var))
    return noise_var

def calc_BGLS(t, y, w, freq):
    tau = 0.5 * np.arctan(sum(w * np.sin(4 * np.pi * t * freq))/sum(w * np.cos(4 * np.pi * t * freq)))
    c = sum(w * np.cos(2.0 * np.pi * t * freq - tau))
    s = sum(w * np.sin(2.0 * np.pi * t * freq - tau))
    cc = sum(w * np.cos(2.0 * np.pi * t * freq - tau)**2)
    ss = sum(w * np.sin(2.0 * np.pi * t * freq - tau)**2)
    yc = sum(w * y * np.cos(2.0 * np.pi * t * freq - tau))
    ys = sum(w * y * np.sin(2.0 * np.pi * t * freq - tau))
    Y = sum(w * y)
    W = sum(w)

    assert(cc > 0)
    assert(ss > 0)
    
    K = (c**2/cc + s**2/ss - W)/2.0
    L = Y - c*yc/cc - s*ys/ss
    M = (yc**2/cc + ys**2/ss)/2.0
    log_prob = np.log(1.0 / np.sqrt(abs(K) * cc * ss)) + (M - L**2/4.0/K)
    return log_prob


    
#dat = np.loadtxt("cleaned/37394.cl.dat", usecols=(0,1), skiprows=1)

#t = dat[:,0]
#t /= 365.25
#t += offset
#y = dat[:,1]
#noise_var = mw_utils.get_seasonal_noise_var(t, y)

sns = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

if plot_both_metrics:
    fig_stats, (ax_stats_1, ax_stats_2) = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False)
    fig_stats.set_size_inches(6, 7)
    
    ax_stats_1.set_ylabel(r'$S_3$', fontsize=axis_label_fs)#,fontsize=20)
    ax_stats_2.set_ylabel(r'$S_4$', fontsize=axis_label_fs, labelpad=-5)#,fontsize=20)
    ax_stats_1.text(0.05, 0.9,'(a)', horizontalalignment='center', transform=ax_stats_1.transAxes, fontsize=panel_label_fs)
    ax_stats_2.text(0.05, 0.9,'(b)', horizontalalignment='center', transform=ax_stats_2.transAxes, fontsize=panel_label_fs)
    
    ax_stats_2.set_xlabel(r'$\sigma_1/\sigma_2$', fontsize=axis_label_fs)#,fontsize=20)
    ax_stats_2.set_xlim([np.sqrt(min(sns)), np.sqrt(max(sns))])
else:
    fig_stats, ax_stats_1 = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=False)
    fig_stats.set_size_inches(6, 4)
    
    ax_stats_1.set_ylabel(r'$S_2$', fontsize=axis_label_fs)#,fontsize=20)
    ax_stats_1.text(0.05, 0.9,'(b)', horizontalalignment='center', transform=ax_stats_1.transAxes, fontsize=panel_label_fs)    
    ax_stats_1.set_xlabel(r'$\sigma_1/\sigma_2$', fontsize=axis_label_fs, labelpad=-5)#,fontsize=20)
    ax_stats_1.set_xlim([np.sqrt(min(sns)), np.sqrt(max(sns))])

fig_stats.tight_layout()    
lines1 = [None, None, None]
lines2 = [None, None, None]

num_exp = len(sns)
num_rep = 2000

num_bs = 1000

for setup_no in [0, 1, 2]:
    if setup_selected is not None and setup_selected != setup_no:
        continue

    uniform_sampling = True
    #if setup_no == 1 or setup_no == 3:
    #    uniform_sampling = True
    
    linear_noise = False
    if setup_no == 0:
        linear_noise = True
    
    real_sampling = False
    if setup_no == 2:
        real_sampling = True

    outperforms = np.zeros(num_exp)
    bglst_err_means = np.zeros(num_exp)
    bglst_err_stds = np.zeros(num_exp)
    ls_err_means = np.zeros(num_exp)
    ls_err_stds = np.zeros(num_exp)
    
    bglst_err_means_bs = np.zeros((num_exp, num_bs))
    ls_err_means_bs = np.zeros((num_exp, num_bs))
    
    for exp_no in np.arange(0, num_exp):
    
        print "---------------------------------"
        print "Setup", setup_no, "Experiment", exp_no
        if os.path.exists('noise_tests/'+str(setup_no)+'/noise_exp_' + str(setup_no) + '_' + str(exp_no) + '.pkl'):
            (true_freqs, bglst_freqs, ls_freqs, dats) = pickle.load(open('noise_tests/'+str(setup_no)+'/noise_exp_' + str(setup_no) + '_' + str(exp_no) + '.pkl', 'rb'))
        else:
            sig_var = 1.0
            initial_sn_ratio = np.random.uniform(0.1, 0.5)
            final_sn_ratio = initial_sn_ratio * sns[exp_no]#np.random.uniform(2.0*initial_sn_ratio, 20.0*initial_sn_ratio)
        
            dats = []
            true_freqs = np.zeros(num_rep)
            bglst_freqs = np.zeros(num_rep)
            ls_freqs = np.zeros(num_rep)
            
            num_worse = 0
            for rep_no in np.arange(0, num_rep):
        
                if real_sampling:
                    t, y, noise_var, star = real_dats[np.random.choice(len(real_dats))]
                    n = len(t)
                    noise_var = np.array(noise_var)
                    noise_var -= min(noise_var)
                    noise_var /= max(noise_var)
                    noise_var *= sig_var/initial_sn_ratio - sig_var/final_sn_ratio
                    noise_var += sig_var/final_sn_ratio
                else:
                    time_range = 30.0
                    n = num_points
                    if uniform_sampling:
                        t = np.random.uniform(0, time_range, n)
                    else:
                        t = np.zeros(n)
                        for i in np.arange(0, n):
                            t[i] = np.random.uniform(time_range * float(i)/n, time_range)
                    t = np.sort(t)
                    if linear_noise:
                        noise_var = np.zeros(n)
                        for i in np.arange(0, n):
                            noise_var[i] = sig_var/final_sn_ratio + t[i]/time_range*(sig_var/initial_sn_ratio - sig_var/final_sn_ratio)
                    else:
                        noise_var = np.zeros(n)
                        for i in np.arange(0, n):
                            if t[i] < time_range/2:
                                noise_var[i] = sig_var/initial_sn_ratio
                            else:
                                noise_var[i] = sig_var/final_sn_ratio
                # Read real data to get some meaningful noise variance
            
                #sig_var = total_var - noise_var
                
                # Now generate synthetic data
                real_period = np.random.uniform(5.0, 15.0)
                y_signal = np.sqrt(sig_var) * np.cos(2 * np.pi * t / real_period + np.random.uniform() * 2.0 * np.pi)
                y_noise = np.random.normal(np.zeros(n), np.sqrt(noise_var), n)
                y = y_signal + y_noise#np.random.normal(0.0, np.sqrt(np.mean(noise_var)), n)
                #print "real freq:", 1.0/real_period
                true_freq = 1.0/real_period
                true_freqs[rep_no] = true_freq
                #t -= np.mean(t)
                
                #duration = max(t) - min(t)
                
                freq_start = 0.001
                freq_end = 2.0*true_freq
                freq_count = 1000
                
                real_noise_var = noise_var
                #now produce empirical noise_var from sample variance
                if real_sampling:
                    noise_var = mw_utils.get_seasonal_noise_var(t, y)
                else:
                    noise_var = get_local_noise_var(t, y, 1.0)
                
                mse_noise = np.sum((real_noise_var - noise_var)**2)/np.dot(real_noise_var, real_noise_var)
                #print "MSE noise_var", mse_noise
                w = np.ones(n)/noise_var
                
                start = time.time()
                #slope, intercept, r_value, p_value, std_err = stats.linregress(t, y)
                #print "slope, intercept", slope, intercept
                if iterative:
                    bglst = BGLST.BGLST(t, y, np.ones(n)/np.var(y), 
                                        w_A = 2.0/np.var(y), A_hat = 0.0,
                                        w_B = 2.0/np.var(y), B_hat = 0.0,
                                        #w_alpha = duration**2 / np.var(y), alpha_hat = slope, 
                                        #w_beta = 1.0 / (np.var(y) + intercept**2), beta_hat = intercept)
                                        w_alpha = 1e10, alpha_hat = 0.0, 
                                        w_beta = 1e10, beta_hat = 0.0)
                else:
                    bglst = BGLST.BGLST(t, y, w, 
                                        w_A = 2.0/np.var(y), A_hat = 0.0,
                                        w_B = 2.0/np.var(y), B_hat = 0.0,
                                        #w_alpha = duration**2 / np.var(y), alpha_hat = slope, 
                                        #w_beta = 1.0 / (np.var(y) + intercept**2), beta_hat = intercept)
                                        w_alpha = 1e10, alpha_hat = 0.0, 
                                        w_beta = 1e10, beta_hat = 0.0)
                (freqs, probs) = bglst.calc_all(freq_start, freq_end, freq_count)
                end = time.time()
                
                max_prob = max(probs)
                max_prob_index = np.argmax(probs)
                best_freq_bglst = freqs[max_prob_index]
                
                if iterative:
                    tau, (A, B, alpha, beta), _, y_model, loglik = bglst.model(best_freq_bglst)
                    if real_sampling:
                        noise_var = mw_utils.get_seasonal_noise_var(t, y - y_model)
                    else:
                        noise_var = get_local_noise_var(t, y - y_model, 1.0)
                    
                    #print "var y, residue", np.std(y), np.std(y - y_model)
                    mse_noise_2 = np.sum((real_noise_var - noise_var)**2)/np.dot(real_noise_var, real_noise_var)
                    if (mse_noise_2 > mse_noise):
                        num_worse += 1
                        print "ERR_freq", abs(true_freq - best_freq_bglst)/true_freq
                        print "MSE y_signal", np.sum((y_signal - y_model)**2)/np.dot(y_signal, y_signal)
                    #print "MSE noise_var2", mse_noise_2
                    w = np.ones(n)/noise_var
                    
                    start = time.time()
                    #slope, intercept, r_value, p_value, std_err = stats.linregress(t, y)
                    #print "slope, intercept", slope, intercept
                    bglst = BGLST.BGLST(t, y, w, 
                                        w_A = 2.0/np.var(y), A_hat = 0.0,
                                        w_B = 2.0/np.var(y), B_hat = 0.0,
                                        #w_alpha = duration**2 / np.var(y), alpha_hat = slope, 
                                        #w_beta = 1.0 / (np.var(y) + intercept**2), beta_hat = intercept)
                                        w_alpha = 1e10, alpha_hat = 0.0, 
                                        w_beta = 1e10, beta_hat = 0.0)
                    
                    (freqs, probs) = bglst.calc_all(freq_start, freq_end, freq_count)
                    end = time.time()
                    
                    max_prob = max(probs)
                    max_prob_index = np.argmax(probs)
                    best_freq_bglst = freqs[max_prob_index]
                    
                bglst_freqs[rep_no] = best_freq_bglst

                
                ###############################################################################
                # LS
                
                sigma = np.sqrt(noise_var)
                ls = LombScargle(t, y)#, sigma)
                power = ls.power(freqs, normalization='psd')#/np.var(y)
                
                max_power_ind = np.argmax(power)
                max_power = power[max_power_ind]
                best_freq_ls = freqs[max_power_ind]
            
                #print "LS: ", best_freq, max_power
                ls_freqs[rep_no] = best_freq_ls
                
                dats.append((t, y, w))
        
            print "Num worse:", float(num_worse)/num_rep
            dats = np.asarray(dats)
        
            with open('noise_tests/'+str(setup_no)+'/noise_exp_' + str(setup_no) + '_' + str(exp_no) + '.pkl', 'wb') as f:
                pickle.dump((true_freqs, bglst_freqs, ls_freqs, dats), f)
        
        bglst_errs = np.abs(bglst_freqs-true_freqs)/true_freqs
        ls_errs = np.abs(ls_freqs-true_freqs)/true_freqs
        err_ratios = bglst_errs/ls_errs
        outperforms[exp_no] = float(len(np.where(bglst_errs < ls_errs)[0]))/float(len(np.where(bglst_errs != ls_errs)[0]))
        bglst_err_means[exp_no] = np.mean(bglst_errs)
        bglst_err_stds[exp_no] = np.std(bglst_errs)
        ls_err_means[exp_no] = np.mean(ls_errs)
        ls_err_stds[exp_no] = np.std(ls_errs)
        
        for bs_index in np.arange(0, num_bs):
            bglst_errs_bs = np.random.choice(bglst_errs, size=len(bglst_errs))
            ls_errs_bs = np.random.choice(ls_errs, size=len(ls_errs))
            bglst_err_means_bs[exp_no, bs_index] = np.mean(bglst_errs_bs)
            ls_err_means_bs[exp_no, bs_index] = np.mean(ls_errs_bs)
    
        ###########################################################################
        # Plot experiment results
        positive_indices = np.where(err_ratios < 0.95)[0]
        negative_indices = np.where(err_ratios > 1.05)[0]
        num_positive = float(len(positive_indices))
        num_negative = float(len(negative_indices))
    
        print "SN:", sns[exp_no]
        #print "----------Before 3sigma----------"
        print num_positive + num_negative
        print "bglst_errs < ls_errs", float(len(np.where(bglst_errs < ls_errs)[0]))/float(len(np.where(bglst_errs != ls_errs)[0]))
        print "bglst_errs < ls_errs (5perc)", num_positive/(num_positive + num_negative)
        print "BGLST_ERR_MEAN:", np.mean(bglst_errs)
        print "BGLST_ERR_STD:", np.std(bglst_errs)
        print "LS_ERR_MEAN:", np.mean(ls_errs)
        print "LS_ERR_STD:", np.std(ls_errs)
        
        # remove outliers
        indices1 = np.where(bglst_errs < 5.0*np.std(bglst_errs))[0]
        bglst_errs = bglst_errs[indices1]
        ls_errs = ls_errs[indices1]
        dats = dats[indices1]
        true_freqs = true_freqs[indices1]
        indices2 = np.where(ls_errs < 5.0*np.std(ls_errs))[0]
        bglst_errs = bglst_errs[indices2]
        ls_errs = ls_errs[indices2]
        dats = dats[indices2]
        true_freqs = true_freqs[indices2]
        
        diffs = ls_errs - bglst_errs
        print "<Diffs>", np.mean(diffs)
        print "Diffs skew", scipy.stats.skew(diffs)
        
        diff_sort_indices = np.argsort(diffs)
        extreme_index1 = diff_sort_indices[-1]
        extreme_index2 = diff_sort_indices[0]
        
        #diff = diffs[extreme_index1]
        true_freq = true_freqs[extreme_index1]
        (t, y, w) = dats[extreme_index1]
        extreme_data_1 = (t, y, w, true_freq)
        
        #diff = diffs[extreme_index2]
        true_freq = true_freqs[extreme_index2]
        (t, y, w) = dats[extreme_index2]
        extreme_data_2 = (t, y, w, true_freq)
        
        fig, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False)
        fig.set_size_inches(12, 7)
    
        ax_a.text(0.95, 0.9,'(a)', horizontalalignment='center', transform=ax_a.transAxes, fontsize=panel_label_fs)
        ax_b.text(0.95, 0.9,'(b)', horizontalalignment='center', transform=ax_b.transAxes, fontsize=panel_label_fs)
        ax_c.text(0.95, 0.9,'(c)', horizontalalignment='center', transform=ax_c.transAxes, fontsize=panel_label_fs)
        ax_d.text(0.95, 0.9,'(d)', horizontalalignment='center', transform=ax_d.transAxes, fontsize=panel_label_fs)
        
        for (ax_a, ax_b), (t, y, w, true_freq) in [((ax_a, ax_c), extreme_data_1), ((ax_b, ax_d), extreme_data_2)]:
            freq_start = 0.001
            freq_end = 2.0*true_freq
            freq_count = 1000
        
            t = t - min(t)
            ax_a.scatter(t, y, marker='+', color ='k', lw=0.5)
            bglst = BGLST.BGLST(t, y, w, 
                                w_A = 2.0/np.var(y), A_hat = 0.0,
                                w_B = 2.0/np.var(y), B_hat = 0.0,
                                #w_alpha = duration**2 / np.var(y), alpha_hat = slope, 
                                #w_beta = 1.0 / (np.var(y) + intercept**2), beta_hat = intercept)
                                w_alpha = 1e10, alpha_hat = 0.0, 
                                w_beta = 1e10, beta_hat = 0.0)
            (freqs, probs) = bglst.calc_all(freq_start, freq_end, freq_count)
        
            max_prob = max(probs)
            min_prob = min(probs)
            max_prob_index = np.argmax(probs)
            best_freq_bglst = freqs[max_prob_index]
        
            tau, (A, B, alpha, beta), _, y_model, loglik = bglst.model(best_freq_bglst)
            
            t_model = np.linspace(min(t), max(t), 1000)
            y_model = np.cos(t_model * 2.0 * np.pi * best_freq_bglst - tau) * A  + np.sin(t_model * 2.0 * np.pi * best_freq_bglst - tau) * B + t_model * alpha + beta
            
            ax_a.plot(t_model, y_model, 'r-')
            
            norm_probs = (probs - min_prob) / (max_prob - min_prob)
            ax_b.plot(freqs, norm_probs, 'r-')
            ax_b.plot([best_freq_bglst, best_freq_bglst], [0, norm_probs[max_prob_index]], 'r-')
            
            ls = LombScargle(t, y)#, sigma)
            power = ls.power(freqs, normalization='psd')#/np.var(y)
        
            min_power = min(power)
            max_power_ind = np.argmax(power)
            max_power = power[max_power_ind]
            best_freq_ls = freqs[max_power_ind]
        
        
            y_model = ls.model(t_model, best_freq_ls)   
            ax_a.plot(t_model, y_model, 'b--')
            
            norm_powers = (power - min_power) / (max_power - min_power)
            
            ax_b.plot(freqs, norm_powers, 'b--')
            ax_b.plot([best_freq_ls, best_freq_ls], [0, norm_powers[max_power_ind]], 'b--')
            
            ax_b.plot([true_freq, true_freq], [0, norm_powers[max_power_ind]], 'k-.')
            
            ax_a.set_xlabel(r'$t$', fontsize=axis_label_fs)#,fontsize=20)
            ax_a.set_ylabel(r'y', fontsize=axis_label_fs)#,fontsize=20)
            ax_a.set_xlim([min(t), max(t)])
            
            ax_b.set_xlabel(r'$f$', fontsize=axis_label_fs)#,fontsize=20)
            ax_b.set_ylabel(r'Power', fontsize=axis_label_fs)#,fontsize=20)
            ax_b.set_xlim([0.001, 2.0*true_freq])
        
        #print exp_no, initial_sn_ratio, final_sn_ratio, true_freq, bglst_freqs[exp_no], ls_freqs[exp_no]
        fig.savefig('noise_tests/'+str(setup_no)+'/example_' + str(exp_no) + '.eps')
        
        plt.close()
        
        ##n, bins, patches = plt.hist([bglst_errs[positive_indices], ls_errs[negative_indices]], bins=50, normed=True, histtype='step', color=['red', 'blue'], alpha=0.5)
        #n, bins, patches = plt.hist(diffs, bins=50, normed=True)
        #plt.xlabel(r'$\left< \Delta f/f \right>$', fontsize=axis_label_fs)
        #plt.savefig('noise_tests/'+str(setup_no)+'/hist_' + str(exp_no) + '.eps')
        #plt.close()
        ###########################################################################
    
    
    ###############################################################################
    # Plot experiment statistics
    if plot_both_metrics:
        line1, = ax_stats_1.plot(np.sqrt(sns), outperforms, label = "Exp. no. " + str(setup_no+1), color=linecolors[setup_no], linestyle=linestyles[setup_no])
        line2, = ax_stats_2.plot(np.sqrt(sns), np.ones(num_exp) - bglst_err_means/ls_err_means, label = "Exp. no. " + str(setup_no+1), color=linecolors[setup_no], linestyle=linestyles[setup_no])
    else:
        line1, = ax_stats_1.plot(np.sqrt(sns), np.ones(num_exp) - bglst_err_means/ls_err_means, label = "Exp. no. " + str(setup_no+1), color=linecolors[setup_no], linestyle=linestyles[setup_no])
        stat_bs = np.ones((num_exp, num_bs)) - bglst_err_means_bs/ls_err_means_bs
        ax_stats_1.fill_between(np.sqrt(sns), np.percentile(stat_bs, 2.5, axis=1), np.percentile(stat_bs, 97.5, axis=1), alpha=0.2, facecolor=errorcolors[setup_no], interpolate=True)
        
    lines1[setup_no] = line1
    #lines2[setup_no] = line2
    
    
if plot_both_metrics:
    ax_stats_1.legend(handles=lines1, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0., handletextpad=0.)#, columnspacing=10)
    ax_stats_1.plot([np.sqrt(min(sns)), np.sqrt(max(sns))], [0.5, 0.5], 'k:')
    ax_stats_2.plot([np.sqrt(min(sns)), np.sqrt(max(sns))], [0.0, 0.0], 'k:')
    #ax_stats_2.legend(handles=lines2, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=1, mode="expand", borderaxespad=0.)
else:
    #ax_stats_1.legend(handles=lines1,
    #        numpoints = 1,
    #        scatterpoints=1,
    #        loc='upper left', ncol=1,
    #        fontsize=11, labelspacing=0.7)
    ax_stats_1.set_position([0.13, 0.12, 0.85, 0.75])      
    ######ax_stats_1.legend(handles=lines1, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0., handletextpad=0., fontsize=panel_label_fs)#, columnspacing=10)
    ax_stats_1.plot([np.sqrt(min(sns)), np.sqrt(max(sns))], [0.0, 0.0], 'k:')
    
fig_stats.savefig("noise_tests/noise_stats.pdf")
plt.close()
###############################################################################
