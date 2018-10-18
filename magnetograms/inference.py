import sys
sys.path.append('../')
import matplotlib as mpl

mpl.use('Agg')
mpl.rcParams['figure.figsize'] = (20, 30)
import pickle
import numpy as np
import pylab as plt
from filelock import FileLock
import pandas as pd

import os
import os.path
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
    
num_iters = 50
num_chains = 4

if len(sys.argv) > 3:
    num_iters = int(sys.argv[3])
if len(sys.argv) > 4:
    num_chains = int(sys.argv[4])

n_jobs = num_chains
n_tries = 1
downsample_iters = 1

model = pickle.load(open('model.pkl', 'rb'))

n = 100
x = np.zeros((n, 2))
y = np.array((n, 2))

thetas = np.zeros(n)

m = 0
length_scale = 0.01
noise_var = 0.1

loglik = 0
last_loglik = None
eps = 0.1

while last_loglik is None or (abs(loglik - loglik) > eps):

    initial_param_values = []
    for i in np.arange(0, num_chains):                    
        initial_m = m
        initial_length_scale = length_scale
        initial_param_values.append(dict(m=initial_m))
    
    fit = model.sampling(data=dict(x=x,N=n,y=y,noise_var=noise_var), init=initial_param_values, iter=num_iters, chains=num_chains, n_jobs=n_jobs)
    
    results = fit.extract()
    
    loglik_samples = results['lp__']
    loglik = np.mean(loglik_samples)
    
    length_scale_samples = results['length_scale'];
    length_scale = np.mean(length_scale_samples)
    
    sig_var_samples = results['sig_var']
    sig_var = np.mean(sig_var_samples)
    
    m_samples = results['m'];
    m = np.mean(m_samples)
    
    print "loglik=", loglik, "last_loglik=", last_loglik
    print "sig_var=", sig_var
    print "length_scale", length_scale
    print "m", m
    
    last_loglik = loglik
