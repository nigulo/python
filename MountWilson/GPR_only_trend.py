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

#num_iters = 50
#num_chains = 50
#down_sample_factor = 1
#n_jobs=16

num_iters = 50
num_chains = 4
down_sample_factor = 1
n_jobs=4

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
            star = file[:-4]
            star = star.upper()
            if (star[-3:] == '.CL'):
                star = star[0:-3]
            if (star[0:2] == 'HD'):
                star = star[2:]
            if not os.path.isfile("GPR_only_trend/" + star + "_results.txt"):
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

model = pickle.load(open('model_only_trend.pkl', 'rb'))

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
    if star != "82635":
        continue
    print star
    dat = np.loadtxt("cleaned/"+file, usecols=(0,1), skiprows=1)
    t = dat[:,0]
    y = dat[:,1]

    (t, y) = mw_utils.daily_averages(t ,y)
    #np.savetxt("GPR_stan/" + star + ".dat", np.column_stack((t_daily, y_daily)), fmt='%f')

    t /= 365.25
    t += offset
    
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
    t -= np.mean(t)
    print min(t), max(t)


    harmonicity = 1.0/duration
    var = np.var(y)
    sigma_f = var/4#np.var(y) / 2

    #noise_var = np.max(mw_utils.get_seasonal_noise_var(t, y))
    noise_var_prop = mw_utils.get_seasonal_noise_var(t, y)
    
    fit = model.sampling(data=dict(x=t,N=n,y=y,noise_var=noise_var_prop, var_y=np.var(y)), iter=num_iters, chains=num_chains, n_jobs=n_jobs)

    with open("GPR_only_trend/"+star + "_results.txt", "w") as output:
        output.write(str(fit))    

    fit.plot()
    plt.savefig("GPR_only_trend/"+star + "_results.png")
    plt.close()

    results = fit.extract()
    
    noise_var = 1.0

    trend_var_samples = results['trend_var'];
    (trend_var, trend_var_se) = mw_utils.mean_with_se(trend_var_samples)
    #c_samples = results['c'];
    #c = np.mean(c_samples)
    m_samples = results['m'];
    m = np.mean(m_samples)
    
    loglik_samples = results['lp__'];
    loglik = np.mean(loglik_samples)

    fig, (ax2) = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(18, 12)

    print "noise_var=", noise_var
    #print "trend_var", trend_var
    #print "c", c
    print "m", m
    print "loglik", loglik
    
    
    gpr_gp = GPR_QP.GPR_QP(sig_var=0, length_scale=0, freq=0, noise_var=noise_var_prop*noise_var, rot_freq=0, rot_amplitude=0, trend_var=trend_var, c=0.0)
    t_test = np.linspace(min(t), max(t), 500)
    (f_mean, var, loglik) = gpr_gp.fit(t, y-m, t_test)
    f_mean += m    
    
    print "loglik", (loglik + 0.5 * n * np.log(2.0 * np.pi))
    
    ax2.plot(t, y, 'b+')
    ax2.plot(t_test, f_mean, 'k-')
    ax2.fill_between(t_test, f_mean + 2.0 * np.sqrt(var), f_mean - 2.0 * np.sqrt(var), alpha=0.1, facecolor='lightgray', interpolate=True)

    

    ###########################################################################
    # For model comparison we use seasonal means    
    seasonal_means = mw_utils.get_seasonal_means(t, y)
    seasonal_noise_var = mw_utils.get_seasonal_noise_var(t, y, False)
    t_seasons = seasonal_means[:,0]    
    y_seasons = seasonal_means[:,1]    
    n_seasons = len(t_seasons)

    gpr_gp_seasons = GPR_QP.GPR_QP(sig_var=0, length_scale=0, freq=0, noise_var=seasonal_noise_var*noise_var, rot_freq=0, rot_amplitude=0, trend_var=trend_var, c=0.0)
    t_test = np.linspace(min(t), max(t), 20) # not important
    (_, _, loglik_seasons) = gpr_gp_seasons.fit(t_seasons, y_seasons-m, t_test)
    
    gpr_gp_seasons_null = GPR_QP.GPR_QP(sig_var=0.0, length_scale=0, freq=0.0, noise_var=seasonal_noise_var*noise_var, rot_freq=0, rot_amplitude=0, trend_var=trend_var, c=0.0)
    t_test = np.linspace(min(t), max(t), 200) # not important
    (f_mean_null, _, loglik_seasons_null) = gpr_gp_seasons_null.fit(t_seasons, y_seasons-m, t_test)

    ax2.plot(t_test, f_mean_null+m, 'r-')

    fig.savefig("GPR_only_trend/"+star + '_fit.png')
    plt.close()

    log_n = np.log(len(t_seasons))
    num_params = 3
    num_params_null = 3
        
    bic = 2.0*loglik_seasons - log_n * num_params
    bic_null = 2.0*loglik_seasons_null - log_n * num_params_null

    ###########################################################################

    with FileLock("GPRLock"):
        with open("GPR_only_trend/results.txt", "a") as output:
            #output.write(star + ' ' + str(period/duration < 2.0/3.0 and period > 2) + ' ' + str(period) + ' ' + str(np.std(period_samples)) + " " + str(length_scale) + " " + str(np.std(length_scale_samples)) + " " + str(rot_amplitude) + " " + str(rot_amplitude_std) + " " + str(bic - bic_null) + "\n")    
            output.write(star + " " + str(trend_var) + " " + str(trend_var_se)+ " " + str(np.std(trend_var_samples)) + " " + str(bic - bic_null) + "\n")    
