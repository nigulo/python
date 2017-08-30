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

num_iters = 50
num_chains = 4
down_sample_factor = 8

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

#data_dir = "../downsampling/results"
data_dir = "../cleaned_wo_rot2"
if data_dir == "../cleaned":
    skiprows = 1
else:
    skiprows = 0

files = []


def load_BGLST_results():
    data = pd.read_csv("BGLST_results.txt", names=['star', 'cyc', 'sigma', 'normality', 'bic'], header=0, dtype=None, sep='\s+', engine='python').as_matrix()
    bglst_cycles = dict()
    for [star, cyc, std, normality, bic] in data:
        if not bglst_cycles.has_key(star):
            bglst_cycles[star] = list()
        all_cycles = bglst_cycles[star]
        cycles = list()
        if not np.isnan(cyc):
            cycles.append(cyc)
            cycles.append(std)
        all_cycles.append(np.asarray(cycles))
    return bglst_cycles

bglst_cycles = load_BGLST_results()

stars = np.array([])
if os.path.isfile("stars.txt"):
    #stars_to_recalculate = pd.read_csv("GPR_stan_stars.txt", names=['star'], dtype=None, sep='\s+', engine='python').as_matrix()
    stars = data = np.genfromtxt("stars.txt", dtype=None, delimiter=' ')

stars_to_recalculate = np.array([])
if os.path.isfile("GPR_stan_stars.txt"):
    #stars_to_recalculate = pd.read_csv("GPR_stan_stars.txt", names=['star'], dtype=None, sep='\s+', engine='python').as_matrix()
    stars_to_recalculate = np.genfromtxt("stars_to_recalculate.txt", dtype=None, delimiter=' ')

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

rot_periods = mw_utils.load_rot_periods("../")

model = pickle.load(open('model.pkl', 'rb'))
model_rot = model#pickle.load(open('model_rot.pkl', 'rb'))

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


    var = np.var(y)

    rot_freq = 0.0
    #if rot_periods.has_key(star):
    #    rot_freq = 365.25/rot_periods[star]

    prior_freq_mean = 0.0
    prior_freq_std = 0.167
    if bglst_cycles.has_key(star):
        cycles = bglst_cycles[star]
        assert(len(cycles) > peak_no)
        prior_freq_mean = cycles[peak_no][0]
        prior_freq_std = cycles[peak_no][1]
        
    print "prior_freq_mean, prior_freq_std: ", prior_freq_mean, prior_freq_std
    
    initial_param_values = []
    for i in np.arange(0, num_chains):                    
        #initial_freq = np.random.uniform(0.25*i/num_chains,0.25*(i+1)/num_chains)
        initial_freq = max(0, np.random.normal(prior_freq_mean, prior_freq_std))
        initial_m = orig_mean
        initial_trend_var = var / duration
        initial_noise_var = 1.0
        initial_inv_length_scale = 0.0001#abs(np.random.normal(0, prior_freq_mean))
        initial_param_values.append(dict(freq=initial_freq, trend_var=initial_trend_var, m=initial_m, noise_var=initial_noise_var, inv_lengh_scale=initial_inv_length_scale))

    if rot_freq > 0: 
        fit = model_rot.sampling(data=dict(x=t,N=n,y=y,noise_var_prop=noise_var_prop, var_y=var, 
           var_seasonal_means=seasonal_means_var, rot_freq=rot_freq, prior_freq_mean=prior_freq_mean, prior_freq_std=prior_freq_std), 
           init=initial_param_values,
           iter=num_iters, chains=num_chains, n_jobs=n_jobs)
    else:
        fit = model.sampling(data=dict(x=t,N=n,y=y,noise_var=noise_var_prop, var_y=var,
            var_seasonal_means=seasonal_means_var, prior_freq_mean=prior_freq_mean, prior_freq_std=prior_freq_std), 
            init=initial_param_values,
            iter=num_iters, chains=num_chains, n_jobs=n_jobs)
    
    with open("results/"+star + downsample_iter_str + "_results.txt", "w") as output:
        output.write(str(fit))    


    fit.plot()
    plt.savefig("results/"+star + downsample_iter_str + "_results.png")
    plt.close()

    results = fit.extract()

    loglik_samples = results['lp__']
    loglik = np.mean(loglik_samples)
    
    length_scale_samples = results['length_scale'];
    (length_scale, length_scale_se) = mw_utils.mean_with_se(length_scale_samples)

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
    opt_freq_label = freq_kmeans.predict(np.array([freq]).reshape((-1, 1)))
    freq1_samples = np.sort(freq_samples[np.where(freq_kmeans.labels_ == opt_freq_label)])
    
    inds = np.searchsorted(freqs, freq1_samples)
    freqs1 = freqs[inds]

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    fig.set_size_inches(18, 12)

    ax1.plot(freqs, freq_freqs(freqs), freqs1, freq_freqs(freqs1), 'k--')

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


    fig.savefig("results/"+star + downsample_iter_str +  '_fit.png')
    plt.close()

    log_n_seasons = np.log(n_seasons)
    if rot_freq > 0:
        num_params = 7
        num_params_null = 4
    else:
        num_params = 5
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
        with open("results/results.txt", "a") as output:
            #output.write(star + ' ' + str(period/duration < 2.0/3.0 and period > 2) + ' ' + str(period) + ' ' + str(np.std(period_samples)) + " " + str(length_scale) + " " + str(np.std(length_scale_samples)) + " " + str(rot_amplitude) + " " + str(rot_amplitude_std) + " " + str(bic - bic_null) + "\n")    
            output.write(star + " " + str(downsample_iter) + " " + str(period/duration < 2.0/3.0 and period > 2.0) + " " + str(period) + " " + str(period_se) + ' ' + str(np.std(period_samples)) + " " + str(length_scale) + " " + str(length_scale_se) + " " + str(np.std(length_scale_samples)) + " " + str(trend_var) + " " + str(trend_var_se)+ " " + str(np.std(trend_var_samples)) + " " + str(rot_amplitude/np.var(y)) + " " + str(fvu) + " " + str(bic_seasons_null - bic_seasons) + "\n")    
