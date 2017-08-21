# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 09:35:12 2016

@author: nigul
"""

import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['figure.figsize'] = (20, 15)
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

num_iters = 200
num_chains = 1
down_sample_factor = 8

num_groups = 1
group_no = 0
if len(sys.argv) > 1:
    num_groups = int(sys.argv[1])
if len(sys.argv) > 2:
    group_no = int(sys.argv[2])

files = []

for root, dirs, dir_files in os.walk("cleaned"):
    for file in dir_files:
        if file[-4:] == ".dat":
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
    if star != "19787":#"103095":#"154417":#"103095":
        continue
    print star
    dat = np.loadtxt("cleaned/"+file, usecols=(0,1), skiprows=1)
    t = dat[:,0]
    t /= 365.25
    t += offset
    
    y = dat[:,1]
    

    if down_sample_factor >= 2:
        indices = np.random.choice(len(t), len(t)/down_sample_factor, replace=False, p=None)
        indices = np.sort(indices)
    
        t = t[indices]
        y = y[indices]
        n = len(t)
    
    duration = max(t) - min(t)
    orig_mean = np.mean(y)
    #y -= orig_mean
    orig_std = np.std(y)
    n = len(t)
    #t -= np.mean(t)


    harmonicity = 1.0/duration
    var = np.var(y)
    sigma_f = var/4#np.var(y) / 2

    #noise_var = np.max(mw_utils.get_seasonal_noise_var(t, y))
    noise_var = mw_utils.get_seasonal_noise_var(t, y)
    
    rot_freq = 0.0
    if rot_periods.has_key(star):
        rot_freq = 365.25/rot_periods[star]

    if rot_freq > 0: 
        fit = model_rot.sampling(data=dict(x=t,N=n,y=y,noise_var=noise_var, var_y=np.var(y), harmonicity=harmonicity, rot_freq=rot_freq), iter=num_iters, chains=num_chains)
    else:
        fit = model.sampling(data=dict(x=t,N=n,y=y,noise_var=noise_var, var_y=np.var(y), harmonicity=harmonicity), iter=num_iters, chains=num_chains)

    #with open("GPR_stan/"+star + "_results.txt", "w") as output:
    #    output.write(str(fit))    

    fit.plot()
    #plt.savefig("GPR_stan/"+star + "_results.png")
    plt.close()

    results = fit.extract()
    
    sig_var_samples = results['sig_var'];
    sig_var = np.mean(sig_var_samples)
    length_scale_samples = results['length_scale'];
    length_scale = np.mean(length_scale_samples)
    freq_samples = results['freq'];
    freq = np.mean(freq_samples)
    if rot_freq > 0:
        rot_amplitude_samples = results['rot_amplitude'];
        rot_amplitude = np.mean(rot_amplitude_samples)
    else:
        rot_amplitude = 0
    period_samples = np.ones(len(freq_samples)) / freq_samples;
    period = np.mean(period_samples)
    trend_var_samples = results['trend_var'];
    trend_var = np.mean(trend_var_samples)
    loglik_samples = results['lp__'];
    loglik = np.mean(loglik_samples)

    freq_freqs = gaussian_kde(freq_samples)
    freqs = np.linspace(min(freq_samples), max(freq_samples), 1000)
    #density.covariance_factor = lambda : .25
    #density._compute_covariance()
    freq = freqs[np.argmax(freq_freqs(freqs))]
    local_maxima_inds = mw_utils.find_local_maxima(freq_freqs(freqs))
    
    freq_kmeans = KMeans(n_clusters=len(local_maxima_inds)).fit(freq_samples.reshape((-1, 1)))
    opt_freq_label = freq_kmeans.predict([freq])
    freq1_samples = np.sort(freq_samples[np.where(freq_kmeans.labels_ == opt_freq_label)])
    
    inds = np.searchsorted(freqs, freq1_samples)
    freqs1 = freqs[inds]

    print "sig_var=", sig_var
    print "length_scale", length_scale
    print "rot_amplitude", rot_amplitude
    print "freq", freq
    print "trend_var", trend_var
    print "loglik", loglik
    
    
    gpr_gp = GPR_QP.GPR_QP(sig_var=sig_var, length_scale=length_scale, freq=freq, noise_var=noise_var, rot_freq=rot_freq, rot_amplitude=rot_amplitude, trend_var=trend_var)
    t_test = np.linspace(min(t), max(t), 500)
    (f_mean, var, loglik) = gpr_gp.fit(t, y, t_test)
    
    print "loglik", (loglik + 0.5 * n * np.log(2.0 * np.pi))
    
    fig, _ = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(18, 6)
    plt.plot(t, y, 'b+')
    plt.plot(t_test, f_mean, 'k-')
    plt.fill_between(t_test, f_mean + 2.0 * np.sqrt(var), f_mean - 2.0 * np.sqrt(var), alpha=0.1, facecolor='lightgray', interpolate=True)

    #plt.savefig("GPR_stan/"+star + '_fit.png')
    plt.close()
    

    ###########################################################################
    # For model comparison we use seasonal means    
    seasonal_means = mw_utils.get_seasonal_means(t, y)
    seasonal_noise_var = mw_utils.get_seasonal_noise_var(t, y, False)
    t_seasons = seasonal_means[:,0]    
    y_seasons = seasonal_means[:,1]    
    n_seasons = len(t_seasons)

    gpr_gp_seasons = GPR_QP.GPR_QP(sig_var=sig_var, length_scale=length_scale, freq=freq, noise_var=seasonal_noise_var, rot_freq=rot_freq, rot_amplitude=rot_amplitude, trend_var=trend_var)
    t_test = np.linspace(min(t), max(t), 20) # not important
    (_, _, loglik_seasons) = gpr_gp_seasons.fit(t_seasons, y_seasons, t_test)
    
    gpr_gp_seasons_null = GPR_QP.GPR_QP(sig_var=0.0, length_scale=length_scale, freq=0.0, noise_var=seasonal_noise_var, rot_freq=rot_freq, rot_amplitude=rot_amplitude, trend_var=trend_var)
    t_test = np.linspace(min(t), max(t), 20) # not important
    (_, _, loglik_seasons_null) = gpr_gp_seasons_null.fit(t_seasons, y_seasons, t_test)

    log_n = np.log(len(t_seasons))
    if rot_freq > 0:
        num_params = 5
        num_params_null = 2
    else:
        num_params = 4
        num_params_null = 1
        
    bic = 2.0*loglik_seasons - log_n * num_params
    bic_null = 2.0*loglik_seasons_null - log_n * num_params_null

    ###########################################################################

    #with FileLock("GPRLock"):
    #    with open("GPR_stan/results.txt", "a") as output:
    #        output.write(star + ' ' + str(period/duration < 2.0/3.0 and period > 2) + ' ' + str(period) + ' ' + str(np.std(period_samples)) + " " + str(length_scale) + " " + str(np.std(length_scale_samples)) + " " + str(rot_freq) + " " + str(bic - bic_null) + "\n")    
