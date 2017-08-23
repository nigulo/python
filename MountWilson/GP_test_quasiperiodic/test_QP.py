# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 20:20:51 2017

@author: nigul
"""

import sys
sys.path.append('../')
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
import pickle
import GPR_QP
from filelock import FileLock
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
import mw_utils
from scipy import stats
from scipy.misc import logsumexp

offset = 1979.3452
down_sample_factor = 8

model = pickle.load(open('model_test.pkl', 'rb'))

group_no = 0
if len(sys.argv) > 1:
    group_no = int(sys.argv[1])

num_experiments = 40
num_iters = 100
num_chains = 4
n_jobs = 4

def calc_cov(t, f, length_scale, sig_var, trend_var, c):
    k = np.zeros((len(t), len(t)))
    inv_l2 = 1.0/length_scale/length_scale
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            k[i, j] = sig_var*np.exp(-inv_l2*(t[i]-t[j])**2)*(np.cos(2 * np.pi*f*(t[i] - t[j]))) + trend_var * (t[i] - c) * (t[j] - c)
            k[j, i] = k[i, j]
    return k


files = []

for root, dirs, dir_files in os.walk("../cleaned_wo_rot"):
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
    
    dat = np.loadtxt("../cleaned_wo_rot/"+file, usecols=(0,1), skiprows=0)

    t_orig = dat[:,0]
    t_orig /= 365.25
    t_orig += offset


    if down_sample_factor >= 2:
        indices = np.random.choice(len(t_orig), len(t_orig)/down_sample_factor, replace=False, p=None)
        indices = np.sort(indices)
    
        t = t_orig[indices]

    t -= (max(t) + min(t)) / 2

    return t

for experiment_index in np.arange(0, num_experiments):

    duration = 0
    while duration < 30:
        t = select_dataset()
        min_t = min(t)
        duration = max(t) - min_t

        
    n = len(t)
    
    var = 1.0
    sig_var = np.random.uniform(0.2, 0.8)
    noise_var = np.ones(n) * (var - sig_var)
    trend_var = np.random.uniform(0.0, 1.0) * var / duration
    mean = 0.5
    
    p = np.random.uniform(2.0, duration/1.5)
    f = 1.0/p
    length_scale = np.random.uniform(p/2.0, 4.0*p)
    mean = 0.5

    k = calc_cov(t, f, length_scale, sig_var, trend_var, 0.0) + np.diag(noise_var)
    l = la.cholesky(k)
    s = np.random.normal(0, 1, n)
    
    y = np.repeat(mean, n) + np.dot(l, s)
    y += mean
    
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, ncols=1)
    fig.set_size_inches(10, 12)
    
    ax1.scatter(t, y, s=1)
    min_y = min(y)
    max_y = max(y)
    for i in np.arange(1, duration*f):
        ax1.plot((min_t + p * i, min_t + p * i), (min_y, max_y), 'k--')    
    ax1.set_xlim([min_t, max(t)])
    
    #data = np.column_stack((t, y))
    #print data
    #np.savetxt("cyclic.txt", data, fmt='%f')
    noise_var_prop = mw_utils.get_seasonal_noise_var(t, y)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(t, y)
    fit_trend = t * slope + intercept
    y_detrended = y - fit_trend
    
    freqs = np.linspace(0.001, 0.5, 1000)
    power = LombScargle(t, y_detrended, np.sqrt(noise_var_prop)).power(freqs, normalization='psd')
    
    ls_local_maxima_inds = mw_utils.find_local_maxima(power)
    f_opt_ls_ind = np.argmax(power[ls_local_maxima_inds])
    f_opt_ls = freqs[ls_local_maxima_inds][f_opt_ls_ind]
    #f_opt_ls = freqs[np.argmax(power[1:])+1]
    ax2.plot(freqs, power)
    ax2.plot([f_opt_ls, f_opt_ls], [min(power), power[ls_local_maxima_inds][f_opt_ls_ind]], 'k--')
    
    
    w = np.ones(n) / noise_var_prop

    bglst = BGLST.BGLST(t, y, w)
    (freqs, probs) = bglst.calc_all(min(freqs), max(freqs), len(freqs))
    bglst_local_maxima_inds = mw_utils.find_local_maxima(probs)
    f_opt_bglst_ind = np.argmax(probs[bglst_local_maxima_inds])
    f_opt_bglst = freqs[bglst_local_maxima_inds][f_opt_bglst_ind]
    #f_opt_bglst = freqs[np.argmax(probs[1:])+1]
    ax3.plot(freqs, probs)
    ax3.plot([f_opt_bglst, f_opt_bglst], [min(probs), probs[bglst_local_maxima_inds][f_opt_bglst_ind]], 'k--')
    
    #############################################
    # Calculate empirical sigma for f_opt_bglst
    tau, (A, B, alpha, beta), _, y_model, loglik = bglst.model(f_opt_bglst)
    bglst_m = BGLST.BGLST(t, y_model, np.ones(len(t))/np.var(y))
    (freqs_m, log_probs_m) = bglst_m.calc_all(min(freqs), max(freqs), len(freqs))
    log_probs_m -= logsumexp(log_probs_m)
    probs_m = np.exp(log_probs_m)
    f_bglst_sigma = np.sqrt(sum((freqs_m-f_opt_bglst)**2 * probs_m))
    #############################################
    
    
    seasonal_means = mw_utils.get_seasonal_means(t, y)
    #op = model.optimizing(data=dict(x=t,N=n,y=y,noise_var=noise_var, var_y=np.var(y), harmonicity=1.0))
    #freq_gp_opt = op['freq'];
    
    initial_param_values = []
    for i in np.arange(0, num_chains):                    
        initial_freq = np.random.uniform(0, 0.5)
        initial_param_values.append(dict(freq=initial_freq))
    fit = model.sampling(data=dict(x=t,N=n,y=y,noise_var=noise_var_prop, var_y=np.var(y), var_seasonal_means=np.var(seasonal_means), freq_mu=f_opt_bglst, freq_sigma=f_bglst_sigma), 
                         init=initial_param_values,
                         iter=num_iters, chains=num_chains, n_jobs=n_jobs)

    results = fit.extract()
    
    sig_var_samples = results['sig_var'];
    sig_var_gp = np.mean(sig_var_samples)
    length_scale_samples = results['length_scale'];
    length_scale_gp = np.mean(length_scale_samples)
    freq_samples = results['freq'];
    (_, freq_se) = mw_utils.mean_with_se(freq_samples, num_bootstrap=100)

    trend_var_samples = results['trend_var'];
    trend_var_gp = np.mean(trend_var_samples)
    m_samples = results['m'];
    m = np.mean(m_samples)
    
    loglik_samples = results['lp__'];
    loglik = np.mean(loglik_samples)

    freq_freqs = gaussian_kde(freq_samples)
    freqs = np.linspace(min(freq_samples), max(freq_samples),1000)
    #density.covariance_factor = lambda : .25
    #density._compute_covariance()
    freq_gp_ind = np.argmax(freq_freqs(freqs))
    freq_gp = freqs[freq_gp_ind]
    local_maxima_inds = mw_utils.find_local_maxima(freq_freqs(freqs))
    
    print "Num_clusters:", len(local_maxima_inds)
    freq_kmeans = KMeans(n_clusters=len(local_maxima_inds)).fit(freq_samples.reshape((-1, 1)))
    opt_freq_label = freq_kmeans.predict(np.array([freq_gp]).reshape((-1, 1)))
    freq1_samples = np.sort(freq_samples[np.where(freq_kmeans.labels_ == opt_freq_label)])
    
    inds = np.searchsorted(freqs, freq1_samples)
    freqs1 = freqs[inds]

    ax4.plot(freqs, freq_freqs(freqs), freqs1, freq_freqs(freqs1), 'k--')
    ax4.plot([freq_gp, freq_gp], [min(freq_freqs(freqs)), freq_freqs(freqs)[freq_gp_ind]], 'k--')

    gpr_gp = GPR_QP.GPR_QP(sig_var=sig_var_gp, length_scale=length_scale_gp, freq=freq_gp, noise_var=noise_var,rot_freq=0.0, rot_amplitude=0.0, trend_var=trend_var_gp, c=0.0)
    t_test = np.linspace(min(t), max(t), 500)
    (f_mean, var_pred, _) = gpr_gp.fit(t, y-m, t_test)
    f_mean += m

    ax5.plot(t, y, 'b+')
    ax5.plot(t_test, f_mean, 'k-')
    ax5.fill_between(t_test, f_mean + 2.0 * np.sqrt(var_pred), f_mean - 2.0 * np.sqrt(var_pred), alpha=0.1, facecolor='lightgray', interpolate=True)
    ax5.set_xlim([min(t), max(t)])

    index = group_no * num_experiments + experiment_index
    with FileLock("GPRLock"):
        with open("test/results.txt", "a") as output:
            output.write("%s %s %s %s %s %s %s %s %s %s %s %s %s\n" % (index, f, f_opt_ls, f_opt_bglst, freq_gp, freq_se, length_scale, length_scale_gp, sig_var, sig_var_gp, trend_var, trend_var_gp, duration))  
            #output.write("%s %s %s %s %s %s %s %s %s %s\n" % (index, f, f_opt_ls, f_opt_bglst, freq_gp, np.std(freq1_samples), length_scale, sig_var, sig_var_gp, np.std(sig_var_samples)))  

    with open("test/detailed_results_" + str(index) + ".txt", "w") as output:
        output.write(str(fit))    

    fig.savefig("test/cyclic_" + str(index) + ".png")

    fit.plot()
    plt.savefig("test/cyclic_samples_" + str(index) + ".png")
    plt.close()
    
