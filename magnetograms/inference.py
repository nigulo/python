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
x1_range = 3.0
x2_range = 3.0
x = np.dstack(np.meshgrid(np.linspace(0, x1_range, n1), np.linspace(0, x2_range, n2))).reshape(-1, 2)

print x

sig_var_train = 0.2
length_scale_train = 1.0
noise_var_train = 0.0001
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



m = 0.0
length_scale = 0.01
noise_var = 0.1

eps = 0.001
learning_rate = 0.5


print np.shape(x)
print np.shape(y)
print n

def algorithm_a(x, y, y_orig):
    loglik = None
    max_loglik = None

    #thetas = random.uniform(size=n)
    thetas = np.ones(n)/2
    thetas = np.log(thetas)
    print thetas
    num_tries = 0
    
    while (max_loglik is None or num_tries % (n1*n2) != 0 or (loglik < max_loglik) or (loglik > max_loglik + eps)):
        print "num_tries", num_tries
    
        num_tries += 1
    
        initial_param_values = []
        
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
        print "loglik=", loglik, "max_loglik=", max_loglik
        
        if max_loglik is None or loglik > max_loglik:
            max_loglik = loglik
        
        gp = GPR_div_free.GPR_div_free(sig_var, length_scale, noise_var)
        flipped_count = 0
        for i in np.random.choice(n, size=1, replace=False):
            loglik1 = gp.init(x, y)
            js = []
            for j in np.arange(0, n):
                x_diff = x[j] - x[i]
                if (np.dot(x_diff, x_diff) < length_scale**2/4):
                    y[j][0] = -y[j][0]
                    y[j][1] = -y[j][1]
                    js.append(j)
            loglik2 = gp.init(x, y)
            #print loglik1, loglik2
            print js
    
            exp_theta = np.exp(thetas[i])
            new_theta = thetas[i] + loglik1 - scipy.special.logsumexp(np.array([loglik1, loglik2]), b=np.array([exp_theta, 1.0-exp_theta]))
            thetas[i] += learning_rate * (new_theta - thetas[i])
    
            r = np.log(random.uniform())
            thetas_i = thetas[i]
            if r < thetas[i]:
                for j in js:
                    thetas[j] = thetas_i
                    y[j][0] = -y[j][0]
                    y[j][1] = -y[j][1]
            else:
                for j in js:
                    thetas[j] = np.log(1.0 - np.exp(thetas_i))
                
            
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
    print y_orig
    print np.exp(thetas)

    return mse, np.exp(thetas)
    
def algorithm_b(x, y, y_orig):
    loglik = None
    max_loglik = None

    num_tries = 0
    
    while (max_loglik is None or num_tries % (n1*n2) != 0 or (loglik > max_loglik + eps)):
        print "num_tries", num_tries
    
        num_tries += 1
    
        initial_param_values = []
        
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
        print "loglik=", loglik, "max_loglik=", max_loglik
        
        if max_loglik is None or loglik > max_loglik:
            max_loglik = loglik
        
        gp = GPR_div_free.GPR_div_free(sig_var, length_scale, noise_var)
        flipped_count = 0
        for i in np.random.choice(n, size=1, replace=False):
            loglik1 = gp.init(x, y)
            js = []
            for j in np.arange(0, n):
                x_diff = x[j] - x[i]
                if (np.dot(x_diff, x_diff) < length_scale**2/4):
                    y[j][0] = -y[j][0]
                    y[j][1] = -y[j][1]
                    js.append(j)
            loglik2 = gp.init(x, y)
            #print loglik1, loglik2
            print js
    
            for j in js:
                if loglik1 > loglik2:        
                    y[j][0] = -y[j][0]
                    y[j][1] = -y[j][1]
                else:
                    flipped_count += 1
    
        
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
    print y_orig
    
    return mse


res_a = algorithm_a(x, np.array(y), y_orig)
res_b = algorithm_b(x, np.array(y), y_orig)


print "Results:"
print res_a
print res_b