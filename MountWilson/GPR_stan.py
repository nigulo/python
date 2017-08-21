# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 09:35:12 2016

@author: nigul
"""

import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['figure.figsize'] = (20, 30)
import pickle
import numpy as np
import pylab as plt
import sys
from filelock import FileLock
import mw_utils
import GPR_QP

import os
import os.path
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
#import pandas as pd

num_iters = 300
num_chains = 4
dynamic_downsample = True
down_sample_factor = 1
n_jobs = 4
n_tries = 1
downsample_iters = 1

num_groups = 1
group_no = 0
if len(sys.argv) > 1:
    num_groups = int(sys.argv[1])
if len(sys.argv) > 2:
    group_no = int(sys.argv[2])

files = []

stars_to_recalculate = np.array([])
if os.path.isfile("GPR_stan_stars.txt"):
    #stars_to_recalculate = pd.read_csv("GPR_stan_stars.txt", names=['star'], dtype=None, sep='\s+', engine='python').as_matrix()
    stars_to_recalculate = data = np.genfromtxt("GPR_stan_stars.txt", dtype=None, delimiter=' ')

print stars_to_recalculate        
for root, dirs, dir_files in os.walk("cleaned"):
    for file in dir_files:
        if file[-4:] == ".dat":
            star = file[:-4]
            star = star.upper()
            if (star[-3:] == '.CL'):
                star = star[0:-3]
            if (star[0:2] == 'HD'):
                star = star[2:]
            while star[0] == '0': # remove leading zeros
                star = star[1:]
            star_indices, = np.where(stars_to_recalculate == star)
            if len(star_indices) > 0 or (not os.path.isfile("GPR_stan/" + star + "_results.txt") and not os.path.isfile("GPR_stan/" + star + "_0_results.txt")):
                files.append(file)

modulo = len(files) % num_groups
group_size = len(files) / num_groups
if modulo > 0:
    group_size +=1

#output = open("GPR_stan/results.txt", 'w')
#output.close()
#output = open("GPR_stan/all_results.txt", 'w')
#output.close()

offset = 1979.3452

rot_periods = mw_utils.load_rot_periods()

model = pickle.load(open('model.pkl', 'rb'))
model_rot = pickle.load(open('model_rot.pkl', 'rb'))

for i in np.arange(0, len(files)):
    if i < group_no * group_size or i >= (group_no + 1) * group_size:
        continue
    file = files[i]
    star = file[:-4]
    star = star.upper()
    if (star[-3:] == '.CL'):
        star = star[0:-3]
    if (star[0:2] == 'HD'):
        star = star[2:]
    while star[0] == '0': # remove leading zeros
        star = star[1:]
    #if star != "160346":
    #    continue
    print star
    dat = np.loadtxt("cleaned/"+file, usecols=(0,1), skiprows=1)
    t_orig = dat[:,0]
    y_orig = dat[:,1]

    n_orig = len(t_orig)
    
    if dynamic_downsample:
        down_sample_factor = max(1, n_orig / 500)
        downsample_iters = down_sample_factor
        

    for downsample_iter in np.arange(0, downsample_iters):
        if downsample_iters > 1:
            downsample_iter_str = '_' + str(downsample_iter)
        else:
            downsample_iter_str = ''
        if down_sample_factor >= 2:
            #indices = np.random.choice(len(t), len(t)/down_sample_factor, replace=False, p=None)
            #indices = np.sort(indices)
        
            #t = t[indices]
            #y = y[indices]
    
            t = t_orig[downsample_iter::down_sample_factor] 
            y = y_orig[downsample_iter::down_sample_factor] 
        else:
            t = t_orig
            y = y_orig
            
        #(t, y, noise_var_prop) = mw_utils.daily_averages(t, y, mw_utils.get_seasonal_noise_var(t/365.25, y))
        #noise_var_prop = mw_utils.get_seasonal_noise_var(t/365.25, y)
        #np.savetxt("GPR_stan/" + star + ".dat", np.column_stack((t_daily, y_daily)), fmt='%f')
    
        t /= 365.25
        t += offset

        noise_var_prop = mw_utils.get_seasonal_noise_var(t, y)
        seasonal_means_var =np.var(mw_utils.get_seasonal_means(t, y)[:,1])

        
        n = len(t)
        
        print "Downsample factor", float(n_orig)/n
        
        duration = max(t) - min(t)
        orig_mean = np.mean(y)
        #y -= orig_mean
        orig_std = np.std(y)
        n = len(t)
        t -= np.mean(t)
    
    
        harmonicity = 1.0/duration
        var = np.var(y)
    
        rot_freq = 0.0
        if rot_periods.has_key(star):
            rot_freq = 365.25/rot_periods[star]
    
        
        initial_param_values = []
        for i in np.arange(0, num_chains):        
            initial_freq = np.random.uniform(0.25*i/num_chains,0.25*(i+1)/num_chains)
            initial_m = orig_mean
            initial_trend_var = var / duration
            initial_noise_var = 1.0
            #initial_inv_length_scale = np.random.uniform(0.0, 1.0)
            initial_param_values.append(dict(freq=initial_freq, trend_var=initial_trend_var, m=initial_m, noise_var=initial_noise_var))
            
        if rot_freq > 0: 
            fit = model_rot.sampling(data=dict(x=t,N=n,y=y,noise_var_prop=noise_var_prop, var_y=var, 
                   var_seasonal_means=seasonal_means_var, rot_freq=rot_freq), 
                   init=initial_param_values,
                   iter=num_iters, chains=num_chains, n_jobs=n_jobs)
        else:
            fit = model.sampling(data=dict(x=t,N=n,y=y,noise_var=noise_var_prop, var_y=var,
                    var_seasonal_means=seasonal_means_var), 
                    init=initial_param_values,
                    iter=num_iters, chains=num_chains, n_jobs=n_jobs)
    
        with open("GPR_stan/"+star + downsample_iter_str + "_results.txt", "w") as output:
            output.write(str(fit))    
    
        fit.plot()
        plt.savefig("GPR_stan/"+star + downsample_iter_str + "_results.png")
        plt.close()
    
        results = fit.extract()
    
        loglik_samples = results['lp__']
        loglik = np.mean(loglik_samples)
        
        sig_var_samples = results['sig_var']
        sig_var = np.mean(sig_var_samples)
        
        freq_samples = results['freq'];
        #freq_samples=freq_samples[np.where(freq_samples <= 0.5)]
        #freq = np.mean(freq_samples)
        if rot_freq > 0:
            noise_var_samples = results['noise_var'];
            noise_var = np.mean(noise_var_samples)
            rot_amplitude_samples = results['rot_amplitude'];
            rot_amplitude = np.mean(rot_amplitude_samples)
        else:
            noise_var = 1.0
            rot_amplitude = 0.0
        #period_samples = np.ones(len(freq_samples)) / freq_samples;
        #period = np.mean(period_samples)
        trend_var_samples = results['trend_var'];
        (trend_var, trend_var_se) = mw_utils.mean_with_se(trend_var_samples)
        #c_samples = results['c'];
        #c = np.mean(c_samples)
        m_samples = results['m'];
        m = np.mean(m_samples)
        
        freq_freqs = gaussian_kde(freq_samples)
        freqs = np.linspace(min(freq_samples), max(freq_samples), 1000)
        #density.covariance_factor = lambda : .25
        #density._compute_covariance()
        #freq = freqs[np.argmax(freq_freqs(freqs))]
        (freq, freq_se) = mw_utils.mode_with_se(freq_samples)
        local_maxima_inds = mw_utils.find_local_maxima(freq_freqs(freqs))
        
        freq_kmeans = KMeans(n_clusters=len(local_maxima_inds)).fit(freq_samples.reshape((-1, 1)))
        opt_freq_label = freq_kmeans.predict([freq])
        freq1_samples = np.sort(freq_samples[np.where(freq_kmeans.labels_ == opt_freq_label)])
        
        inds = np.searchsorted(freqs, freq1_samples)
        freqs1 = freqs[inds]
    
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        fig.set_size_inches(18, 12)
    
        ax1.plot(freqs, freq_freqs(freqs), freqs1, freq_freqs(freqs1), 'k--')
    
        length_scale = 1.0/freq

        print "var=", var
        print "sig_var=", sig_var
        print "noise_var=", noise_var
        print "length_scale", length_scale
        print "rot_amplitude", rot_amplitude
        print "freq, freq_se", freq, freq_se
        #print "trend_var", trend_var
        #print "c", c
        print "m", m
        #print "loglik", loglik
        
        
        gpr_gp = GPR_QP.GPR_QP(sig_var=sig_var, length_scale=length_scale, freq=freq, noise_var=noise_var_prop*noise_var, rot_freq=rot_freq, rot_amplitude=rot_amplitude, trend_var=trend_var, c=0.0)
        t_test = np.linspace(min(t), max(t), 500)
        (f_mean, pred_var, loglik) = gpr_gp.fit(t, y-m, t_test)
        (f_t, _, _) = gpr_gp.fit(t, y-m, t)
        f_mean += m
        fvu = np.sum((f_t + m - y)**2) / n / var
        print "FVU", fvu
        print "loglik", loglik #(loglik + 0.5 * n * np.log(2.0 * np.pi))

        gpr_gp_only_exp = GPR_QP.GPR_QP(sig_var=sig_var, length_scale=length_scale, freq=0.0, noise_var=noise_var_prop*noise_var, rot_freq=0.0, rot_amplitude=0.0, trend_var=trend_var, c=0.0)
        (f_mean_only_exp, _, _) = gpr_gp_only_exp.fit(t, y-m, t_test)
        f_mean_only_exp += m
        
        
        if rot_freq > 0:
            y_residue = y - f_t
        
            gpr_gp_wo_period = GPR_QP.GPR_QP(sig_var=sig_var, length_scale=length_scale, freq=freq, noise_var=noise_var_prop*noise_var, rot_freq=0.0, rot_amplitude=0.0, trend_var=trend_var, c=0.0)
            (f_mean_wo_rot, pred_var, _) = gpr_gp_wo_period.fit(t, y-m, t_test)
            f_mean_wo_rot += m
            (f_t_wo_rot, _, _) = gpr_gp_wo_period.fit(t, y-m, t)
            y_wo_rot = y_residue + f_t_wo_rot
            
        else:
            y_wo_rot = y
            f_mean_wo_rot = f_mean
    
    
        gpr_gp_null = GPR_QP.GPR_QP(sig_var=0.0, length_scale=0.0, freq=0.0, noise_var=noise_var_prop*noise_var, rot_freq=0.0, rot_amplitude=0.0, trend_var=trend_var, c=0.0)
        t_test_null = np.linspace(min(t), max(t), 500) # not important
        (f_mean_null, _, loglik_null) = gpr_gp_null.fit(t, y_wo_rot-m, t_test_null)
        (f_t_null, _, _) = gpr_gp_null.fit(t, y-m, t)
        print "FVU_null", np.sum((f_t_null + m - y)**2) / n / var
        print "loglik_null", loglik_null #(loglik + 0.5 * n * np.log(2.0 * np.pi))
        
        ax2.plot(t, y, 'b+')
        #ax2.plot(t, y_wo_rot, 'r+')
        ax2.plot(t_test, f_mean, 'k-')
        ax2.plot(t_test, f_mean_only_exp, 'y-')
        ax2.plot(t_test, f_mean_wo_rot, 'g-')
        ax2.fill_between(t_test, f_mean + 2.0 * np.sqrt(pred_var), f_mean - 2.0 * np.sqrt(pred_var), alpha=0.1, facecolor='lightgray', interpolate=True)
    
        ax2.plot(t_test_null, f_mean_null+m, 'r-')
    
    
        ###########################################################################
        # For model comparison we use seasonal means    
        seasonal_means = mw_utils.get_seasonal_means(t, y_wo_rot - m)
        seasonal_noise_var = mw_utils.get_seasonal_noise_var(t, y_wo_rot, False)
        t_seasons = seasonal_means[:,0]    
        y_seasons = seasonal_means[:,1]    
        n_seasons = len(t_seasons)
    
        gpr_gp_seasons = GPR_QP.GPR_QP(sig_var=sig_var, length_scale=length_scale, freq=freq, noise_var=seasonal_noise_var*noise_var, rot_freq=0.0, rot_amplitude=0.0, trend_var=trend_var, c=0.0)
        (f_mean_seasons, _, loglik_seasons) = gpr_gp_seasons.fit(t_seasons, y_seasons, t_seasons)
        se_seasons = np.sum((f_mean_seasons - y_seasons)**2) / n
        
        
        gpr_gp_seasons_null = GPR_QP.GPR_QP(sig_var=0.0, length_scale=length_scale, freq=0.0, noise_var=seasonal_noise_var*noise_var, rot_freq=0.0, rot_amplitude=0.0, trend_var=trend_var, c=0.0)
        (f_mean_seasons_null, _, loglik_seasons_null) = gpr_gp_seasons_null.fit(t_seasons, y_seasons, t_seasons)
        se_seasons_null = np.sum((f_mean_seasons_null - y_seasons)**2) / n
    
        print "loglik_seasons, loglik_seasons_null", loglik_seasons, loglik_seasons_null
        print "SE_seasons, SE_seasons_null", se_seasons, se_seasons_null
        ###########################################################################
    
    
        fig.savefig("GPR_stan/"+star + downsample_iter_str +  '_fit.png')
        plt.close()
    
        log_n_seasons = np.log(n_seasons)
        if rot_freq > 0:
            num_params = 6
            num_params_null = 4
        else:
            num_params = 4
            num_params_null = 2
            
        bic_seasons = log_n_seasons * num_params - 2.0*loglik_seasons
        bic_seasons_null = log_n_seasons * num_params_null - 2.0*loglik_seasons_null
        #bic = np.log(n) * num_params - 2.0*loglik
        #bic_null = np.log(n) * num_params_null - 2.0*loglik_null
    
        #bic_seasons = n_seasons * np.log(se_seasons) + num_params * log_n_seasons
        #bic_seasons_null = n_seasons * np.log(se_seasons_null) + num_params_null * log_n_seasons
    
        print "BIC_seasons, BIC_seasons_null", bic_seasons, bic_seasons_null
    
        ###########################################################################
    
        period = 1.0/freq
        period_samples = np.ones(len(freq1_samples)) / freq1_samples;
        period_se = freq_se/freq/freq
        with FileLock("GPRLock"):
            with open("GPR_stan/results.txt", "a") as output:
                #output.write(star + ' ' + str(period/duration < 2.0/3.0 and period > 2) + ' ' + str(period) + ' ' + str(np.std(period_samples)) + " " + str(length_scale) + " " + str(np.std(length_scale_samples)) + " " + str(rot_amplitude) + " " + str(rot_amplitude_std) + " " + str(bic - bic_null) + "\n")    
                output.write(star + " " + str(downsample_iter) + " " + str(period/duration < 2.0/3.0 and period > 2) + " " + str(period) + " " + str(period_se) + ' ' + str(np.std(period_samples)) + " " + str(trend_var) + " " + str(trend_var_se)+ " " + str(np.std(trend_var_samples)) + " " + str(rot_amplitude/np.var(y)) + " " + str(fvu) + " " + str(bic_seasons_null - bic_seasons) + "\n")    
