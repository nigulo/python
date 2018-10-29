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
import matplotlib.pyplot as plt

num_iters = 50
num_chains = 1
inference = False

eps = 0.001
learning_rate = 1.0
max_num_tries = 20


if len(sys.argv) > 3:
    num_iters = int(sys.argv[3])
if len(sys.argv) > 4:
    num_chains = int(sys.argv[4])

n_jobs = num_chains

model = pickle.load(open('model.pkl', 'rb'))

n1 = 2
n2 = 2
n = n1*n2
x1_range = 1.0
x2_range = 1.0
x_mesh = np.meshgrid(np.linspace(0, x1_range, n1), np.linspace(0, x2_range, n2))
x = np.dstack(x_mesh).reshape(-1, 2)

print x

sig_var_train = 0.2
length_scale_train = 0.3
noise_var_train = 0.000001
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

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
fig.set_size_inches(6, 8)

ax1.set_title('Probabilistic')
ax2.set_title('Non-probabilistic')

y1 = np.cos(x_mesh[0])
y2 = np.sin(x_mesh[1])

ax1.quiver(x_mesh[0], x_mesh[1], y[:,0], y[:,1], units='width', color = 'k')
ax2.quiver(x_mesh[0], x_mesh[1], y[:,0], y[:,1], units='width', color = 'k')
#qk = ax.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
#                   coordinates='figure')

perf_null = 0.0
for i in np.arange(0, n):
    if np.random.uniform() < 0.5:
        y[i] = y[i]*-1
        perf_null += 1.0
perf_null /=n

m = 0.0
length_scale = 1.0
noise_var = 0.0001

print np.shape(x)
print np.shape(y)
print n

def algorithm_a(x, y, y_orig):
    y_in = np.array(y)
    loglik = None
    max_loglik = None

    #thetas = random.uniform(size=n)
    thetas = np.ones(n)/2
    thetas = np.log(thetas)
    num_tries = 0
    
    while (max_loglik is None or num_tries % max_num_tries != 0 or (loglik < max_loglik) or (loglik > max_loglik + eps)):
        print "num_tries", num_tries
    
        num_tries += 1
    
        initial_param_values = []
        
        if inference:
            #for i in np.arange(0, num_chains):
            #    initial_m = m
            #    initial_length_scale = length_scale
            #    initial_param_values.append(dict(m=initial_m))
            
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
            gp = GPR_div_free.GPR_div_free(sig_var, length_scale, noise_var, toeplitz=True)
            loglik = gp.init(x, y)
        
        print "sig_var=", sig_var
        print "length_scale", length_scale
        print "m", m
        print "loglik=", loglik, "max_loglik=", max_loglik
        
        if max_loglik is None or loglik > max_loglik:
            num_tries = 1
            max_loglik = loglik
        
        gp = GPR_div_free.GPR_div_free(sig_var, length_scale, noise_var, toeplitz=True)

        y_last = np.array(y)
        for i in np.random.choice(n, size=1, replace=False):
            js = []
            for j in np.arange(0, n):
                x_diff = x[j] - x[i]
                if (np.dot(x_diff, x_diff) < length_scale**2/4):
                    js.append(j)
            for j in js:
                y[j] = np.array(y_in[j])
            loglik1 = gp.init(x, y)
            for j in js:
                y[j] = np.array(y_in[j])*-1
            loglik2 = gp.init(x, y)

            if (loglik1 > loglik or loglik2 > loglik):    
                thetas_i = thetas[i]
                exp_theta = np.exp(thetas_i)
                #new_theta = loglik1 - scipy.special.logsumexp(np.array([loglik1, loglik2]))
                #thetas[i] = new_theta
                new_theta = thetas_i + loglik1 - scipy.special.logsumexp(np.array([loglik1, loglik2]), b=np.array([exp_theta, 1.0-exp_theta]))
                thetas[i] += learning_rate * (new_theta - thetas_i)
    
                thetas_i = thetas[i]
        
                print thetas_i, np.exp(thetas_i), loglik1, loglik2
                #r = np.log(random.uniform())
                for j in js:
                    thetas[j] = thetas_i
                    if thetas_i > np.log(0.5):#r < thetas_i
                        #assert(loglik1 >= loglik2)
                        y[j] = np.array(y_in[j])
                    else:
                        #assert(loglik2 >= loglik1)
                        y[j] = np.array(y_in[j])*-1
                        
                #if thetas_i > np.log(0.5):#r < thetas_i
                #    for j in js:
                #        thetas[j] = thetas_i
                #        y[j][0] = -y[j][0]
                #        y[j][1] = -y[j][1]
                #else:
                #    for j in js:
                #        thetas[j] = np.log(1.0 - np.exp(thetas_i))
            else:
                y = y_last
            
    
    num_guessed = 0.0
    for i in np.arange(0, n):
        if np.array_equal(y[i], y_orig[i]):
            num_guessed += 1.0

    exp_thetas = np.exp(thetas)
    if num_guessed < n/2:
        num_guessed = n - num_guessed
        y *= -1
        exp_thetas = np.ones(n) - exp_thetas

    return num_guessed/n, exp_thetas, y
    
def algorithm_b(x, y, y_orig):
    loglik = None
    max_loglik = None

    num_tries = 0
    
    while (max_loglik is None or num_tries % max_num_tries != 0 or (loglik > max_loglik + eps)):
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
            gp = GPR_div_free.GPR_div_free(sig_var, length_scale, noise_var, toeplitz=True)
            loglik = gp.init(x, y)
        
        print "sig_var=", sig_var
        print "length_scale", length_scale
        print "m", m
        print "loglik=", loglik, "max_loglik=", max_loglik
        
        if max_loglik is None or loglik > max_loglik:
            num_tries = 1
            max_loglik = loglik
        
        gp = GPR_div_free.GPR_div_free(sig_var, length_scale, noise_var, toeplitz=True)
        for i in np.random.choice(n, size=1, replace=False):
            loglik1 = loglik#gp.init(x, y)
            js = []
            for j in np.arange(0, n):
                x_diff = x[j] - x[i]
                if (np.dot(x_diff, x_diff) < length_scale**2/4):
                    y[j] = y[j]*-1
                    js.append(j)
            loglik2 = gp.init(x, y)
            for j in js:
                if loglik1 > loglik2:        
                    y[j] = y[j]*-1
    
        
    
    num_guessed = 0.0
    for i in np.arange(0, n):
        if np.array_equal(y[i], y_orig[i]):
            num_guessed += 1.0

    if num_guessed < n/2:
        num_guessed = n - num_guessed
        y *= -1
   
    return num_guessed/n, y


print "******************** Algorithm a ********************"
perf_a, prob_a, field_a = algorithm_a(x, np.array(y), y_orig)
print "******************** Algorithm b ********************"
perf_b, field_b = algorithm_b(x, np.array(y), y_orig)

print "Results:"
print perf_null
print perf_a, prob_a#np.mean(np.abs(np.ones(n)/2 - prob_a)*2)
print perf_b

Q_a = ax1.quiver(x_mesh[0], x_mesh[1], field_a[:,0], field_a[:,1], units='width', color='r', linestyle=':')
Q_b = ax2.quiver(x_mesh[0], x_mesh[1], field_b[:,0], field_b[:,1], units='width', color='r', linestyle=':')

fig.savefig("field.png")
