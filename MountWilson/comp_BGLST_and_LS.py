# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 20:20:51 2017

@author: nigul
"""

import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['figure.figsize'] = (20, 15)
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from astropy.stats import LombScargle
import os
import os.path
import BGLST
import mw_utils
from scipy import stats
import pickle

offset = 1979.3452
down_sample_factor = 8

def calc_cov(t, f, sig_var, trend_var, c):
    k = np.zeros((len(t), len(t)))
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            k[i, j] = sig_var*(np.cos(2 * np.pi*f*(t[i] - t[j]))) + trend_var * (t[i] - c) * (t[j] - c)
            k[j, i] = k[i, j]
    return k



real_dats = []

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

            if down_sample_factor >= 2:
                indices = np.random.choice(len(t), len(t)/down_sample_factor, replace=False, p=None)
                indices = np.sort(indices)
                t = t[indices]
                y = y[indices]
                noise_var = noise_var[indices]
    
            if max(t) - min(t) >= 30:
                real_dats.append(t)

files = []

for root, dirs, dir_files in os.walk("cleaned"):
    for file in dir_files:
        if file[-4:] == ".dat":
            star = file[:-4]
            star = star.upper()
            if (star[-3:] == '.CL'):
                star = star[0:-3]
            if (star[0:2] == 'HD'):
                star = star[2:]
            files.append(file)

def select_dataset():
    file = files[np.random.choice(len(files))]
    
    dat = np.loadtxt("cleaned/"+file, usecols=(0,1), skiprows=1)

    t_orig = dat[:,0]
    t_orig /= 365.25
    t_orig += offset


    if down_sample_factor >= 2:
        indices = np.random.choice(len(t_orig), len(t_orig)/down_sample_factor, replace=False, p=None)
        indices = np.sort(indices)
    
        t = t_orig[indices]

    t -= (max(t) + min(t)) / 2

    return t

trend_var_coefs = [0.0, 0.1, 0.2, 0.4, 0.8, 1.6]

axis_label_fs = 15
panel_label_fs = 15



setup_no = 0
for real_sampling in [False, True]:

    fig_stats, (ax_stats_1, ax_stats_2) = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False)
    fig_stats.set_size_inches(6, 7)
    
    ax_stats_1.set_ylabel(r'$S_3$', fontsize=axis_label_fs)#,fontsize=20)
    ax_stats_2.set_ylabel(r'$S_4$', fontsize=axis_label_fs)#, labelpad=-5)#,fontsize=20)
    ax_stats_1.text(0.05, 0.9,'(a)', horizontalalignment='center', transform=ax_stats_1.transAxes, fontsize=panel_label_fs)
    ax_stats_2.text(0.05, 0.9,'(b)', horizontalalignment='center', transform=ax_stats_2.transAxes, fontsize=panel_label_fs)
    
    ax_stats_2.set_xlabel(r'$k$', fontsize=axis_label_fs)#,fontsize=20)
    ax_stats_2.set_xlim([min(trend_var_coefs), max(trend_var_coefs)])
    
    num_exp = len(trend_var_coefs)
    outperforms_bglst = np.zeros(num_exp)
    outperforms_ls_t = np.zeros(num_exp)
    outperforms_ls = np.zeros(num_exp)
    bglst_err_means = np.zeros(num_exp)
    bglst_err_stds = np.zeros(num_exp)
    ls_err_means = np.zeros(num_exp)
    ls_err_stds = np.zeros(num_exp)
    ls_t_err_means = np.zeros(num_exp)
    ls_t_err_stds = np.zeros(num_exp)
    
    exp_no = 0
    for trend_var_coef in [0.0, 0.1, 0.2, 0.4, 0.8, 1.6]:
        print "Experiment with trend_var_coef", trend_var_coef

        if os.path.exists('comp/'+str(setup_no)+'/noise_exp_' + str(exp_no) + '.pkl'):
            (true_freqs, bglst_freqs, ls_freqs, ls_t_freqs, dats) = pickle.load(open('comp/'+str(setup_no)+'/noise_exp_' + str(exp_no) + '.pkl', 'rb'))
        else:

            dats = []
            num_rep = 2000
            true_freqs = np.zeros(num_rep)
            bglst_freqs = np.zeros(num_rep)
            ls_freqs = np.zeros(num_rep)
            ls_t_freqs = np.zeros(num_rep)
            
            for rep_index in np.arange(0, num_rep):
                if real_sampling:
                    t = real_dats[np.random.choice(len(real_dats))]
                    n = len(t)
                else:
                    time_range = 30.0
                    n = 200
                    t = np.random.uniform(0, time_range, n)
                    t = np.sort(t)
                min_t = min(t)
                duration = max(t) - min_t
                            
                var = 1.0
                sig_var = np.random.uniform(0.2, 0.8)
                noise_var = np.ones(n) * (var - sig_var)
                trend_var = trend_var_coef * var / duration
                mean = np.random.uniform(-1.0, 1.0)
                
                p = np.random.uniform(2.0, duration/1.5)
                f = 1.0/p
                true_freqs[rep_index] = f
            
                k = calc_cov(t, f, sig_var, trend_var, 0.0) + np.diag(noise_var)
                l = la.cholesky(k)
                s = np.random.normal(0, 1, n)
                
                y = np.repeat(mean, n) + np.dot(l, s)
                y += mean
                
                #data = np.column_stack((t, y))
                #print data
                #np.savetxt("cyclic.txt", data, fmt='%f')
                #noise_var_prop = mw_utils.get_seasonal_noise_var(t, y)
                freqs = np.linspace(0.001, 0.5, 1000)
    
                power = LombScargle(t, y, np.sqrt(noise_var)).power(freqs, normalization='psd')
                
                ls_local_maxima_inds = mw_utils.find_local_maxima(power)
                f_opt_ls_ind = np.argmax(power[ls_local_maxima_inds])
                f_opt_ls = freqs[ls_local_maxima_inds][f_opt_ls_ind]
                ls_freqs[rep_index] = f_opt_ls
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(t, y)
                #print "slope, intercept", slope, intercept
                fit_trend = t * slope + intercept
                y_detrended = y - fit_trend
                
                power_t = LombScargle(t, y_detrended, np.sqrt(noise_var)).power(freqs, normalization='psd')
                
                ls_t_local_maxima_inds = mw_utils.find_local_maxima(power_t)
                f_opt_ls_t_ind = np.argmax(power[ls_t_local_maxima_inds])
                f_opt_ls_t = freqs[ls_t_local_maxima_inds][f_opt_ls_t_ind]
                ls_t_freqs[rep_index] = f_opt_ls_t
    
                w = np.ones(n) / noise_var
            
                bglst = BGLST.BGLST(t, y, w, 
                                w_A = 2.0/np.var(y), A_hat = 0.0,
                                w_B = 2.0/np.var(y), B_hat = 0.0,
                                w_alpha = duration**2 / np.var(y), alpha_hat = slope, 
                                w_beta = 1.0 / (np.var(y) + intercept**2), beta_hat = intercept)
                
                
                (freqs, probs) = bglst.calc_all(min(freqs), max(freqs), len(freqs))
                bglst_local_maxima_inds = mw_utils.find_local_maxima(probs)
                f_opt_bglst_ind = np.argmax(probs[bglst_local_maxima_inds])
                f_opt_bglst = freqs[bglst_local_maxima_inds][f_opt_bglst_ind]
                bglst_freqs[rep_index] = f_opt_bglst
                dats.append((t, y, w))
        
            dats = np.asarray(dats)
    
            with open('comp/'+str(setup_no)+'/noise_exp_' + str(exp_no) + '.pkl', 'wb') as f:
                pickle.dump((true_freqs, bglst_freqs, ls_freqs, ls_t_freqs, dats), f)

        bglst_errs = np.abs(bglst_freqs-true_freqs)/true_freqs
        ls_errs = np.abs(ls_freqs-true_freqs)/true_freqs
        ls_t_errs = np.abs(ls_t_freqs-true_freqs)/true_freqs
        
        indices1 = np.where(bglst_errs < ls_errs)[0]
        indices2 = np.where(bglst_errs < ls_t_errs)[0]
        indices1a = np.where(bglst_errs != ls_errs)[0]
        indices2a = np.where(bglst_errs != ls_t_errs)[0]
        outperforms_bglst[exp_no] = float(len(np.intersect1d(indices1, indices2)))/float(len(np.union1d(indices1a, indices2a)))
        
        indices1 = np.where(ls_errs < bglst_errs)[0]
        indices2 = np.where(ls_errs < ls_t_errs)[0]
        indices1a = np.where(ls_errs != bglst_errs)[0]
        indices2a = np.where(ls_errs != ls_t_errs)[0]
        outperforms_ls[exp_no] = float(len(np.intersect1d(indices1, indices2)))/float(len(np.union1d(indices1a, indices2a)))

        indices1 = np.where(ls_t_errs < bglst_errs)[0]
        indices2 = np.where(ls_t_errs < ls_errs)[0]
        indices1a = np.where(ls_t_errs != bglst_errs)[0]
        indices2a = np.where(ls_t_errs != ls_errs)[0]
        outperforms_ls_t[exp_no] = float(len(np.intersect1d(indices1, indices2)))/float(len(np.union1d(indices1a, indices2a)))
        
        bglst_err_means[exp_no] = np.mean(bglst_errs)
        bglst_err_stds[exp_no] = np.std(bglst_errs)
        ls_err_means[exp_no] = np.mean(ls_errs)
        ls_err_stds[exp_no] = np.std(ls_errs)
        ls_t_err_means[exp_no] = np.mean(ls_t_errs)
        ls_t_err_stds[exp_no] = np.std(ls_t_errs)
    
        exp_no += 1
    
    
    ###############################################################################
    # Plot experiment statistics
    line1, = ax_stats_1.plot(trend_var_coefs, outperforms_bglst, 'r-', label = "Our method")
    line2, = ax_stats_1.plot(trend_var_coefs, outperforms_ls_t, 'b--', label = "GLS-T")
    line3, = ax_stats_1.plot(trend_var_coefs, outperforms_ls, 'g-.', label = "GLS")
    
    ax_stats_2.plot(trend_var_coefs, bglst_err_means, 'r-')
    ax_stats_2.plot(trend_var_coefs, ls_t_err_means, 'b--')
    ax_stats_2.plot(trend_var_coefs, ls_err_means, 'g-.')
    
    ax_stats_1.legend(handles=[line1, line2, line3], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0., handletextpad=0.)#, columnspacing=10)
    
    #ax_stats_1.plot([min(trend_var_coefs), max(trend_var_coefs)], [0.5, 0.5], 'k:')
    #ax_stats_2.plot([min(trend_var_coefs), max(trend_var_coefs)], [0.0, 0.0], 'k:')
    fig_stats.savefig("comp/trend_stats_" + str(setup_no+1) + ".eps")
    plt.close()

    setup_no += 1        
    
###############################################################################
    
        
        
        
            
