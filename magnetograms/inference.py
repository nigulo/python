import sys
sys.path.append('../')
import matplotlib as mpl

mpl.use('Agg')
mpl.rcParams['figure.figsize'] = (20, 30)
import pickle
import numpy as np
#import pylab as plt
#import pandas as pd
import GPR_div_free as GPR_div_free
import scipy.misc
import numpy.random as random

#import os
#import os.path
#from scipy.stats import gaussian_kde
#from sklearn.cluster import KMeans
import numpy.linalg as la

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

n1 = 10
n2 = 10
n = n1*n2
x1_range = 10.0
x2_range = 10.0
x = np.dstack(np.meshgrid(np.linspace(0, x1_range, n1), np.linspace(0, x2_range, n2))).reshape(-1, 2)

print x

sig_var_train = 0.2
length_scale_train = 0.01
noise_var_train = 0.1
m_train = 0.0

gp_train = GPR_div_free.GPR_div_free(sig_var_train, length_scale_train, noise_var_train)
K = gp_train.calc_cov(x, x, True)

L = la.cholesky(K)
s = np.random.normal(0.0, 1.0, 2*n)

y = np.repeat(m_train, 2*n) + np.dot(L, s)
y = np.reshape(y, (n, 2))

print y
thetas = random.uniform(size=n)

m = 0.0
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
    
    gp = GPR_div_free.GPR_div_free(sig_var, length_scale, noise_var)
    for i in np.arange(0, n):
        loglik1 = gp.init(x, y)
        y[i][0] = -y[i][0]
        y[i][1] = -y[i][1]
        loglik2 = gp.init(x, y)
        y[i][0] = -y[i][0]
        y[i][1] = -y[i][1]
        
        thetas[i] = thetas[i] + loglik1 - scipy.misc.logsumexp([loglik1, loglik2], b=[thetas[i], 1.0-thetas[i]])
        
    for i in np.arange(0, n):
        if thetas[i] < 0.5:
            y[i][0] = -y[i][0]
            y[i][1] = -y[i][1]
