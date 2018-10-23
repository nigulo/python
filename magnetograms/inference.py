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
num_chains = 1
inference = False


if len(sys.argv) > 3:
    num_iters = int(sys.argv[3])
if len(sys.argv) > 4:
    num_chains = int(sys.argv[4])

n_jobs = num_chains

model = pickle.load(open('model.pkl', 'rb'))

n1 = 10
n2 = 10
n = n1*n2
x1_range = 10.0
x2_range = 10.0
x = np.dstack(np.meshgrid(np.linspace(0, x1_range, n1), np.linspace(0, x2_range, n2))).reshape(-1, 2)

print x

sig_var_train = 0.2
length_scale_train = 1.0
noise_var_train = 0.1
m_train = 0.0

gp_train = GPR_div_free.GPR_div_free(sig_var_train, length_scale_train, noise_var_train)
K = gp_train.calc_cov(x, x, True)

for i in np.arange(0, n1):
    for j in np.arange(0, n2):
        assert(K[i, j]==K[j, i])

L = la.cholesky(K)
s = np.random.normal(0.0, 1.0, 2*n)

y = np.repeat(m_train, 2*n) + np.dot(L, s)
y = np.reshape(y, (n, 2))

y_orig = np.array(y)

print y_orig

for i in np.arange(0, n):
    if np.random.uniform() < 0.5:
        y[i][0] = -y[i][0]
        y[i][1] = -y[i][1]


#thetas = random.uniform(size=n)
thetas = np.ones(n)/2
thetas = np.log(thetas)
print thetas

m = 0.0
length_scale = 0.01
noise_var = 0.1

loglik = None
last_loglik = None
eps = 0.001


print np.shape(x)
print np.shape(y)
print n

num_tries = 0

while (last_loglik is None or (loglik > last_loglik - eps)) and num_tries < n1 * n2:
    print "num_tries", num_tries
    num_tries += 1

    initial_param_values = []
    
    last_loglik = loglik
    if inference:
        for i in np.arange(0, num_chains):
            initial_m = m
            initial_length_scale = length_scale
            initial_param_values.append(dict(m=initial_m))
        
        fit = model.sampling(data=dict(x=x,N=n,y=y,noise_var=noise_var), init=initial_param_values, iter=num_iters, chains=num_chains, n_jobs=n_jobs)
        
        results = fit.extract()
        loglik_samples = results['lp__']
        print loglik_samples 
        loglik = np.mean(loglik_samples)
        
        length_scale_samples = results['length_scale'];
        length_scale = np.mean(length_scale_samples)
        
        sig_var_samples = results['sig_var']
        sig_var = np.mean(sig_var_samples)
        
        m_samples = results['m'];
        m = np.mean(m_samples)
    else:
        sig_var=sig_var_train
        m=m_train
        length_scale=length_scale_train
        gp = GPR_div_free.GPR_div_free(sig_var, length_scale, noise_var)
        loglik = gp.init(x, y)
    
    print "sig_var=", sig_var
    print "length_scale", length_scale
    print "m", m
    print "loglik=", loglik, "last_loglik=", last_loglik
    
    
    gp = GPR_div_free.GPR_div_free(sig_var, length_scale, noise_var)
    flipped_count = 0
    for i in np.random.choice(n, size=1, replace=False):
        loglik1 = gp.init(x, y)
        y[i][0] = -y[i][0]
        y[i][1] = -y[i][1]
        loglik2 = gp.init(x, y)
        #print loglik1, loglik2
        
        if loglik1 > loglik2:        
            y[i][0] = -y[i][0]
            y[i][1] = -y[i][1]
        else:
            flipped_count += 1
        
        #exp_theta = np.exp(thetas[i])
        ##print "Enne", thetas[i]
        ##print loglik1, np.log(exp_theta * np.exp(loglik1) + (1.0-exp_theta) * np.exp(loglik2))
        ##print scipy.special.logsumexp(np.array([loglik1, loglik2]), b=np.array([exp_theta, 1.0-exp_theta]))
        #thetas[i] = thetas[i] + loglik1 - scipy.special.logsumexp(np.array([loglik1, loglik2]), b=np.array([exp_theta, 1.0-exp_theta]))
        ##print "Parast", thetas[i]
        #if thetas[i] >= np.log(0.5):
        #    y[i][0] = -y[i][0]
        #    y[i][1] = -y[i][1]
        #else:
        #    flipped_count += 1
    
    print "Flipped", flipped_count

mse = 0.0
sq_sum = 0.0
for i in np.arange(0, n):
    diff = y[i] - y_orig[i]
    mse += np.dot(diff, diff)
    sq_sum += np.dot(y_orig[i], y_orig[i])
mse /= sq_sum
print mse
print y-y_orig
