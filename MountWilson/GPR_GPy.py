# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 09:35:12 2016

@author: nigul
"""


import GPy
import numpy as np
from scipy import optimize
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy.linalg as la
import sys
from filelock import FileLock
import mw_utils

import os
import os.path

epsilon = 0.0001

def get_noise_cov(t, y):
    seasons = mw_utils.get_seasons(zip(t, y), 1.0, True)
    cov = np.identity(len(t))
    i = 0
    for season in seasons:
        var = np.var(season[:,1])
        season_len = np.shape(season)[0]
        for j in np.arange(i, i + season_len):
            cov[j, j] = var
        i += season_len
    return cov


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

output = open("GPR_GPy/results.txt", 'w')
output.close()
output = open("GPR_GPy/all_results.txt", 'w')
output.close()

offset = 1979.3452

rot_periods = mw_utils.load_rot_periods()

def calc_R_N_eff(samples, bins = 10):
    M = bins
    N = len(samples) / M
    samples = samples[:N*M]
    bins = samples.reshape((M, N))
    bin_means = np.mean(bins, axis=1)
    avg_bin_means = np.mean(bin_means)
    b = np.sum(bin_means - avg_bin_means)**2 * N / (M-1)
    bin_means = bin_means.reshape((M, 1))
    bin_squares = (bins - bin_means)**2
    bin_vars = np.sum(bin_squares, axis=1) / (N-1)
    w = np.mean(bin_vars)
    var_theta = (w * (N - 1) + b) / N
    r = np.sqrt(var_theta/w)
    ###########################################################################
    tau_max = N
    v_t = np.zeros(tau_max)
    for tau in np.arange(0, tau_max):
        for m in np.arange(0, M):
            corr = np.sum((bins[m, tau:N] - bins[m, 0:N-tau])**2)
            v_t[tau] += corr / (N - tau)
    v_t /= M
    rho_t = -v_t/ (2.0 * var_theta) + 1.0
    rho_t_negs_indices = np.where(rho_t < 0)
    if np.shape(rho_t_negs_indices[0])[0] > 0:
        T = rho_t_negs_indices[0][0]
    else:
        T = len(rho_t)
    N_eff = M * N / (2.0 * (1.0 + np.sum(rho_t[:T])))
    return (r, N_eff)

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
    if star != "103095":
        continue
    print star
    dat = np.loadtxt("cleaned/"+file, usecols=(0,1), skiprows=1)
    t = dat[:,0]
    t /= 365.25
    t += offset
    duration = max(t) - min(t)
    
    y = dat[:,1]
    orig_mean = np.mean(y)
    y -= orig_mean
    orif_std = np.std(y)
    y /= np.std(y)
    
    t = t.reshape((len(t), 1))    
    y = y.reshape((len(y), 1))    
    
    indices1 = np.random.choice(len(t), len(t)/2, replace=False, p=None)
    indices1 = np.sort(indices1)

    indices2 = np.random.choice(len(indices1), len(indices1)/2, replace=False, p=None)
    indices2 = np.sort(indices2)
    
    best_loglik = None

    #Test
    #params = optimize.fmin_bfgs(lambda params, *args: np.sin(params[0])*np.cos(params[1]), [1, 1], fprime=None, args=())
    #print "Optimum: ", params

    
    var = np.var(y)
    sigma_f = var/4#np.var(y) / 2
    #sigma_n = np.max(get_noise_var(t, y))
    sigma_n = np.max(mw_utils.get_seasonal_noise_var(t, y))
    #noise = mw_utils.get_seasonal_noise_var(t, y)
    #noise = noise.reshape((len(noise), 1))    
    #sigma_n = var/4#sigma_f
    #if rot_periods.has_key(star):
    #else:
    #    kernel = GPy.kern.RBF(input_dim=1, variance=sigma_f, lengthscale=3.)
    #kernel = GPy.kern.RBF(input_dim=1, variance=sigma_f, lengthscale=3.)
    kernel = GPy.kern.Linear(input_dim=1, variances=sigma_f)
    kernel += GPy.kern.PeriodicExponential(input_dim=1, variance=sigma_f, lengthscale=1., period = 5.)

    m = GPy.models.GPRegression(t,y,kernel)
    m['.*Gaussian_noise.variance'] = sigma_n #Set the noise parameters to the error in Y
    m.Gaussian_noise.variance.fix() #We can fix the noise term, since we already know it

    m.optimize(messages=True)
    #m.optimize_restarts(num_restarts = 10)
    print m
    fig = m.plot(plot_density=False)
    #GPy.plotting.show(fig, filename='test.png')
    plt.savefig("GPR_GPy/fit.png")
    plt.close()

    m.sum.linear.variances.set_prior(GPy.priors.Gamma.from_EV(m.sum.linear.variances,(m.sum.linear.variances/3)**2))
    m.sum.periodic_exponential.lengthscale.set_prior(GPy.priors.Gamma.from_EV(m.sum.periodic_exponential.lengthscale,(m.sum.periodic_exponential.lengthscale/3)**2))
    m.sum.periodic_exponential.variance.set_prior(GPy.priors.Gamma.from_EV(m.sum.periodic_exponential.variance,(m.sum.periodic_exponential.variance/3)**2))
    m.sum.periodic_exponential.period.set_prior(GPy.priors.Gamma.from_EV(m.sum.periodic_exponential.period, (m.sum.periodic_exponential.period/3)**2))#GPy.priors.Uniform(2.,100.))
    
    #m.sum.rbf.variance.fix()
    #m.sum.rbf.lengthscale.fix()

    #m.sum.periodic_exponential.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
    #m.sum.periodic_exponential.variance.set_prior(GPy.priors.Gamma.from_EV(1.,1.))
    #m.sum.periodic_exponential.period.set_prior(GPy.priors.Gamma.from_EV(1.,100.))#GPy.priors.Uniform(2.,100.))
    #m.sum.rbf.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
    #m.sum.rbf.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
    #m.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
    
    a = np.array([(m.sum.linear.variances[0]/3)**2, (m.sum.periodic_exponential.lengthscale[0]/3)**2, (m.sum.periodic_exponential.variance[0]/3)**2, (m.sum.periodic_exponential.period[0]/3)**2])
    print a
    cov=np.diag(a)
    print cov
    mcmc = GPy.inference.mcmc.Metropolis_Hastings(m, cov=cov)
    mcmc.sample(Ntotal=200, Nburn=50, Nthin=1, tune=True, tune_throughout=False, tune_interval=10) # Burnin
    s = np.asarray(mcmc.chains)[0]
    #hmc = GPy.inference.mcmc.HMC(m,stepsize=0.01)
    #s = hmc.sample(100) # Burnin
    #s = hmc.sample(num_samples=5)
    print s
    labels = ['lin_variance', 'per_lengthscale', 'per_variance', 'per_period']
    for i in xrange(s.shape[1]):
        plt.plot(s[:,i])
        print labels[i], calc_R_N_eff(s[:,i], bins = 5)
        plt.savefig("GPR_GPy/"+labels[i]+"_mcmc.png")
        plt.close()
    
    samples = s#s[300:] # cut out the burn-in period
    from scipy import stats
    xmin = samples.min()
    xmax = samples.max()
    xs = np.linspace(xmin,xmax,100)
    for i in xrange(samples.shape[1]):
        kernel = stats.gaussian_kde(samples[:,i])
        plt.plot(xs,kernel(xs),label=labels[i])
        plt.savefig("GPR_GPy/"+labels[i]+"_hist.png")
        plt.close()


    
#    with FileLock("GPRLock"):
#        with open("GPR_GPy/all_results.txt", "a") as output:
#            output.write(star + " " + str(m.loglik) + ' ' + (' '.join(['%s' % (param) for param in gpr.params])) + "\n")    

