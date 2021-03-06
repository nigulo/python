# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 09:35:12 2016

@author: nigul
"""

import sys
sys.path.append('../')
import matplotlib as mpl

mpl.use('Agg')
mpl.rcParams['figure.figsize'] = (20, 30)
import pickle
import numpy as np
import pylab as plt
from filelock import FileLock
import mw_utils
import GPR_QP
import GPR_QP2
import pandas as pd

import os
import os.path
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
#import pandas as pd


#num_groups = 1
#group_no = 0
#if len(sys.argv) > 1:
#    num_groups = int(sys.argv[1])
#if len(sys.argv) > 2:
#    group_no = int(sys.argv[2])

star = sys.argv[1]
peak_no = int(sys.argv[2])

peak_no_str = ""
if peak_no > 0:
    peak_no_str = str(peak_no) + "/"
    

num_iters = 50
num_chains = 4
down_sample_factor = 1

if len(sys.argv) > 3:
    num_iters = int(sys.argv[3])
if len(sys.argv) > 4:
    num_chains = int(sys.argv[4])
if len(sys.argv) > 5:
    down_sample_factor = int(sys.argv[5])

dynamic_downsample = False
n_jobs = num_chains
n_tries = 1
downsample_iters = 1

print star, peak_no, num_iters, num_chains, down_sample_factor

data_dir = "../GP_input"
if data_dir == "../cleaned":
    skiprows = 1
else:
    skiprows = 0

files = []

data_found = False
for root, dirs, dir_files in os.walk(data_dir):
    for file in dir_files:
        if file[-4:] == ".dat":
            file_star = file[:-4]
            file_star = file_star.upper()
            if (file_star[-3:] == '.CL'):
                file_star = file_star[0:-3]
            if (file_star[0:2] == 'HD'):
                file_star = file_star[2:]
            while star[0] == '0': # remove leading zeros
                file_star = file_star[1:]
            if star == file_star:
                dat = np.loadtxt(data_dir+"/"+file, usecols=(0,1), skiprows=skiprows)
                data_found = True
                break

if not data_found:
    print "Cannot find data for " + star
    sys.exit(1)


offset = 1979.3452

model = pickle.load(open('model.pkl', 'rb'))
model_null = pickle.load(open('model_null.pkl', 'rb'))

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

    seasonal_noise = mw_utils.get_seasonal_noise_var(t, y, per_point=False)
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

    t, y, noise_var_prop = mw_utils.downsample(t, y, noise_var_prop, 15.0/365.25)
    n = len(t)

    var = np.var(y)

    ###########################################################################
    # Quasiperiodic model

    prior_freq_mean = 0.0
    prior_freq_std = 0.167
        
    print "prior_freq_mean, prior_freq_std: ", prior_freq_mean, prior_freq_std
    
    initial_param_values = []
    for i in np.arange(0, num_chains):                    
        #initial_freq = np.random.uniform(0.25*i/num_chains,0.25*(i+1)/num_chains)
        initial_freq = 0.5*float(i+0.5)/num_chains#np.random.uniform(0, 0.5)
        #initial_freq = max(0, np.random.normal(prior_freq_mean, prior_freq_std))
        initial_m = orig_mean
        initial_trend_var = var / duration
        #initial_inv_length_scale = 0.0001#abs(np.random.normal(0, prior_freq_mean))
        #initial_param_values.append(dict(freq=initial_freq, trend_var=initial_trend_var, m=initial_m, noise_var=initial_noise_var, inv_lengh_scale=initial_inv_length_scale))
        initial_param_values.append(dict(freq=initial_freq, trend_var=initial_trend_var, m=initial_m))

    fit = model.sampling(data=dict(x=t,N=n,y=y,noise_var=noise_var_prop, var_y=var,
        var_seasonal_means=seasonal_means_var, prior_freq_mean=prior_freq_mean, prior_freq_std=prior_freq_std), 
        init=initial_param_values,
        iter=num_iters, chains=num_chains, n_jobs=n_jobs)
    
    with open("results/"+peak_no_str+star + downsample_iter_str + "_results.txt", "w") as output:
        output.write(str(fit))    


    fit.plot()
    plt.savefig("results/"+peak_no_str+star + downsample_iter_str + "_results.png")
    plt.close()

    results = fit.extract()

    loglik_samples = results['lp__']
    loglik = np.mean(loglik_samples)
    
    length_scale_samples = results['length_scale'];
    (length_scale, length_scale_se) = mw_utils.mean_with_se(length_scale_samples)

    length_scale2_samples = results['length_scale2'];
    (length_scale2, length_scale2_se) = mw_utils.mean_with_se(length_scale2_samples)

    sig_var_samples = results['sig_var']
    sig_var = np.mean(sig_var_samples)

    sig_var2_samples = results['sig_var2']
    sig_var2 = np.mean(sig_var2_samples)

    m_samples = results['m'];
    m = np.mean(m_samples)

    trend_var_samples = results['trend_var'];
    (trend_var, trend_var_se) = mw_utils.mean_with_se(trend_var_samples)

    ###########################################################################    
    # Find optimum freq 1

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    fig.set_size_inches(18, 12)

    freq_samples = results['freq'];
    
    freq_freqs = gaussian_kde(freq_samples)
    freqs = np.linspace(min(freq_samples), max(freq_samples), 1000)
    (freq, freq_se) = mw_utils.mode_with_se(freq_samples)
    local_maxima_inds = mw_utils.find_local_maxima(freq_freqs(freqs))
    
    freq_kmeans = KMeans(n_clusters=len(local_maxima_inds)).fit(freq_samples.reshape((-1, 1)))
    opt_freq_label = freq_kmeans.predict(np.array([freq]).reshape((-1, 1)))
    freq_samples_ = np.sort(freq_samples[np.where(freq_kmeans.labels_ == opt_freq_label)])
    
    inds = np.searchsorted(freqs, freq_samples_)
    freqs_ = freqs[inds]

    ax1.plot(freqs, freq_freqs(freqs), "b-", freqs_, freq_freqs(freqs_), 'k--')


    ###########################################################################    
    freq2_samples = results['freq2'];
    
    freq2_freqs = gaussian_kde(freq2_samples)
    freqs2 = np.linspace(min(freq2_samples), max(freq2_samples), 1000)
    (freq2, freq2_se) = mw_utils.mode_with_se(freq2_samples)
    local_maxima_inds = mw_utils.find_local_maxima(freq2_freqs(freqs2))
    
    freq2_kmeans = KMeans(n_clusters=len(local_maxima_inds)).fit(freq2_samples.reshape((-1, 1)))
    opt_freq2_label = freq_kmeans.predict(np.array([freq2]).reshape((-1, 1)))
    freq2_samples_ = np.sort(freq_samples[np.where(freq_kmeans.labels_ == opt_freq_label)])
    
    inds = np.searchsorted(freqs2, freq2_samples_)
    freqs2_ = freqs2[inds]

    ax1.plot(freqs2, freq2_freqs(freqs2), "g-", freqs2_, freq2_freqs(freqs2_), 'k--')

    ###########################################################################    

    print "var=", var
    print "sig_var=", sig_var
    print "sig_var2=", sig_var2
    print "length_scale", length_scale
    print "length_scale2", length_scale2
    print "freq, freq_se", freq, freq_se
    print "freq2, freq2_se", freq2, freq2_se
    print "trend_var", trend_var
    print "m", m
    
    gpr_gp = GPR_QP2.GPR_QP2(sig_vars=[sig_var, sig_var2], length_scales=[length_scale, length_scale2], freqs=[freq, freq2], noise_var=noise_var_prop, trend_var=trend_var, c=0.0)
    t_test = np.linspace(min(t), max(t), 500)
    gpr_gp.init(t, y-m)
    (f_mean, pred_var, loglik) = gpr_gp.fit(t_test)
    (f_t, _, _) = gpr_gp.fit(t)
    f_mean += m
    fvu = np.sum((f_t + m - y)**2) / n / var
    print "FVU", fvu
    print "loglik", loglik #(loglik + 0.5 * n * np.log(2.0 * np.pi))
    
    ###########################################################################
    #Squared-exponential GP for model comparison
    
    initial_param_values = []
    for i in np.arange(0, num_chains):                    
        initial_m = orig_mean
        initial_trend_var = var / duration
        initial_param_values.append(dict(trend_var=initial_trend_var, m=initial_m))

    fit_null = model_null.sampling(data=dict(x=t,N=n,y=y,noise_var=noise_var_prop, var_y=var,
        var_seasonal_means=seasonal_means_var, prior_freq_mean=prior_freq_mean, prior_freq_std=prior_freq_std), 
        init=initial_param_values,
        iter=num_iters, chains=num_chains, n_jobs=n_jobs)
    
    with open("results/"+peak_no_str+star + downsample_iter_str + "_results_null.txt", "w") as output:
        output.write(str(fit))    


    fit_null.plot()
    plt.savefig("results/"+peak_no_str+star + downsample_iter_str + "_results_null.png")
    plt.close()

    results_null = fit_null.extract()

    loglik_samples_null = results_null['lp__']
    loglik_null = np.mean(loglik_samples_null)
    
    length_scale_samples_null = results_null['length_scale'];
    (length_scale_null, length_scale_se_null) = mw_utils.mean_with_se(length_scale_samples_null)

    sig_var_samples_null = results_null['sig_var']
    sig_var_null = np.mean(sig_var_samples_null)
    
    trend_var_samples_null = results_null['trend_var'];
    (trend_var_null, trend_var_se_null) = mw_utils.mean_with_se(trend_var_samples_null)
    m_samples_null = results_null['m'];
    m_null = np.mean(m_samples_null)
    
    print "length_scale_null", length_scale_null
    print "trend_var_null", trend_var_null
    print "m_null", m_null
    
    gpr_gp_null = GPR_QP.GPR_QP(sig_var=sig_var_null, length_scale=length_scale_null, freq=0.0, noise_var=noise_var_prop, rot_freq=0.0, rot_amplitude=0.0, trend_var=trend_var_null, c=0.0)
    t_test_null = np.linspace(min(t), max(t), 500)
    gpr_gp_null.init(t, y-m_null)
    (f_mean_null, pred_var_null, loglik_null) = gpr_gp_null.fit(t_test_null)
    (f_t_null, _, _) = gpr_gp_null.fit(t)
    f_mean_null += m_null
    fvu_null = np.sum((f_t_null + m_null - y)**2) / n / var
    print "FVU_null", fvu_null
    print "loglik_null", loglik #(loglik + 0.5 * n * np.log(2.0 * np.pi))

    ###########################################################################
    ax2.plot(t, y, 'b+')
    #ax2.plot(t, y_wo_rot, 'r+')
    ax2.plot(t_test, f_mean, 'k-')
    ax2.fill_between(t_test, f_mean + 2.0 * np.sqrt(pred_var), f_mean - 2.0 * np.sqrt(pred_var), alpha=0.1, facecolor='lightgray', interpolate=True)
    ax2.plot(t_test_null, f_mean_null, 'g-')

    ###########################################################################
    # LOO-CV

    seasons = mw_utils.get_seasons(zip(t, y), 1.0, True)

    l_loo = 0.0
    l_loo_null = 0.0
    dat = np.column_stack((t, y))
    season_index = 0
    for season in seasons:
        season_start = min(season[:,0])
        season_end = max(season[:,0])
        print "cv for season: ", season_index, season_start, season_end
        dat_test = seasons[season_index]
        if season_index == len(seasons) - 1:
            indices = np.where(dat[:,0] < season_start)[0]
            dat_train = dat[indices,:]
            noise_train = noise_var_prop[indices]
            #dat_test = dat[np.where(dat[:,0] >= season_start)[0],:]
        else:
            dat_season = dat[np.where(dat[:,0] < season_end)[0],:]
            indices_after = np.where(dat[:,0] >= season_end)[0]
            dat_after = dat[indices_after,:]
            indices_before = np.where(dat_season[:,0] < season_start)[0]
            dat_before = dat_season[indices_before,:]
            #dat_test = seasonal_means[season_index]# dat_season[np.where(dat_season[:,0] >= season_start)[0],:]
            dat_train = np.concatenate((dat_before, dat_after), axis=0)
            noise_before = noise_var_prop[indices_before]
            noise_after = noise_var_prop[indices_after]
            noise_train = np.concatenate((noise_before, noise_after), axis=0)
        #test_mat = np.array([[1.16490151e-08, 1.16493677e-08], [1.16493677e-08, 1.16497061e-08]])
        #test_mat = np.array([[1.16490151e-08, 1.16e-08], [1.16e-08, 1.16497061e-08]])
        #test_mat *= 1e8
        #print test_mat
        #L_test_covar = la.cholesky(test_mat)
        
        #print indices_before, indices_after, noise_train
        gpr_gp_cv = GPR_QP.GPR_QP(sig_var=sig_var, length_scale=length_scale, freq=freq, noise_var=noise_train, rot_freq=0, rot_amplitude=0, trend_var=trend_var, c=0.0)
        gpr_gp_cv_null = GPR_QP.GPR_QP(sig_var=0.0, length_scale=length_scale, freq=0.0, noise_var=noise_train, rot_freq=0.0, rot_amplitude=0.0, trend_var=trend_var_null, c=0.0)
        gpr_gp_cv.init(dat_train[:,0], dat_train[:,1]-m)
        print seasonal_noise
        print dat_test
        print m
        print m_null
        (_, _, loglik_test) = gpr_gp_cv.cv(dat_test[:,0], dat_test[:,1]-m, np.repeat(seasonal_noise[season_index], np.shape(dat_test)[0]))
        l_loo += loglik_test
        gpr_gp_cv_null.init(dat_train[:,0], dat_train[:,1]-m_null)
        (_, _, loglik_test_null) = gpr_gp_null.cv(dat_test[:,0], dat_test[:,1]-m_null, np.repeat(seasonal_noise[season_index], np.shape(dat_test)[0]))
        l_loo_null += loglik_test_null
        season_index += 1
    print "l_loo, l_loo_null", l_loo, l_loo_null

    ###########################################################################


    fig.savefig("results/"+peak_no_str+star + downsample_iter_str +  '_fit.png')
    plt.close()

    ###########################################################################

    period = 1.0/freq
    period_samples = np.ones(len(freq1_samples)) / freq1_samples;
    period_se = freq_se/freq/freq
    with FileLock("GPRLock"):
        with open("results/"+peak_no_str+"results.txt", "a") as output:
            #output.write(star + ' ' + str(period/duration < 2.0/3.0 and period > 2) + ' ' + str(period) + ' ' + str(np.std(period_samples)) + " " + str(length_scale) + " " + str(np.std(length_scale_samples)) + " " + str(rot_amplitude) + " " + str(rot_amplitude_std) + " " + str(bic - bic_null) + "\n")    
            output.write(star + " " + str(downsample_iter) + " " + str(period/duration < 2.0/3.0 and period > 2.0) + " " + str(period) + " " + str(period_se) + ' ' + str(np.std(period_samples)) + " " + str(length_scale) + " " + str(length_scale_se) + " " + str(np.std(length_scale_samples)) + " " + str(trend_var) + " " + str(trend_var_se)+ " " + str(np.std(trend_var_samples)) + " " + str(m) + " " + str(sig_var) + " " + str(fvu) + " " + str(l_loo - l_loo_null) + " " + "\n")    
