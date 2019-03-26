import os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6

import sys
sys.path.append('../utils')
import matplotlib as mpl

mpl.use('Agg')
mpl.rcParams['figure.figsize'] = (20, 30)
#import pickle
import numpy as np
#import pylab as plt
#import pandas as pd
import GPR_div_free as GPR_div_free
import scipy.misc
import numpy.random as random
import scipy.sparse.linalg as sparse
import scipy.stats as stats
import utils
import plot
import pymc3 as pm
#import os
#import os.path
#from scipy.stats import gaussian_kde
#from sklearn.cluster import KMeans
import numpy.linalg as la
import matplotlib.pyplot as plt
import sampling
import kiss_gp

import os.path
from astropy.io import fits

#hdul = fits.open('pi-ambiguity-test/amb_turb.fits')
hdul = fits.open('pi-ambiguity-test/amb_spot.fits')
dat = hdul[0].data[:,::2,::2]#[:,200:300,200:300]
b = dat[0]
theta = dat[1]
phi = dat[2]

inference = True
sample_or_optimize = False
num_samples = 1
num_chains = 4
inference_after_iter = 20

MODE = 0

eps = 0.001
learning_rate = 0.1
max_num_tries = 20
initial_temp = 0.5
temp_delta = 0.01

if len(sys.argv) > 3:
    num_iters = int(sys.argv[3])
if len(sys.argv) > 4:
    num_chains = int(sys.argv[4])

n_jobs = num_chains

data_loaded = False
n1 = b.shape[0]
n2 = b.shape[1]
x1_range = 1.0
x2_range = 1.0

m1 = 10
m2 = 10

n = n1*n2

x1 = np.linspace(0, x1_range, n1)
x2 = np.linspace(0, x2_range, n2)
x_mesh = np.meshgrid(x1, x2)
x = np.dstack(x_mesh).reshape(-1, 2)
x_flat = np.reshape(x, (2*n, -1))

m = m1 * m2
u1 = np.linspace(0, x1_range, m1)
u2 = np.linspace(0, x2_range, m2)
u_mesh = np.meshgrid(u1, u2)
u = np.dstack(u_mesh).reshape(-1, 2)
    
print("u_mesh=", u_mesh)
print("u=", u)
print(x_mesh)



bz = b*np.cos(theta)
bxy = b*np.sin(theta)
bx = bxy*np.cos(phi)
by = bxy*np.sin(phi)
bx = np.reshape(bx, n)
by = np.reshape(by, n)
bz = np.reshape(bz, n)
bxy = np.reshape(bz, n)
bx_offset = bxy + bz
by_offset = bxy + bz
bx -= bx_offset
by -= by_offset
bx_mean = np.mean(bx)
by_mean = np.mean(by)
bx -= np.mean(bx)
by -= np.mean(by)
bx_offset += bx_mean
by_offset += by_mean
norm = np.std(np.sqrt(bx**2+by**2))
bx /= norm
by /= norm
y = np.column_stack((bx,by))
#y = np.stack((bx,by), axis=2)


y_orig = np.array(y)
print(y_orig)
print(y.shape)

#for i in np.arange(0, n):
#    if np.random.uniform() < 0.5:
#        y[i] = y[i]*-1


components_plot = plot.plot_map(nrows=2, ncols=2)
components_plot.set_color_map('bwr')
components_plot.plot(np.reshape(y[:,0]*norm+bx_offset, (n1, n2)), [0, 0])
components_plot.plot(np.reshape(y[:,1]*norm+by_offset, (n1, n2)), [0, 1])
components_plot.save("components.png")


fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
fig.set_size_inches(6, 12)

ax1.set_title('Input field')
ax2.set_title('Inferred field')

#ax1.quiver(x_mesh[0], x_mesh[1], y_orig[:,0], y_orig[:,1], units='width', color = 'k')
ax1.quiver(x_mesh[0], x_mesh[1], y[:,0], y[:,1], units='width', color = 'k')
fig.savefig("field.png")

noise_var = 0.1*np.var(bx) + 0.1*np.var(by)

print(np.shape(x))
print(np.shape(y))
print(n)

# Using KISS-GP
# https://arxiv.org/pdf/1503.01057.pdf

# define the parameters with their associated priors

data_var = np.var(y[:,0]) + np.var(y[:,1])
print("data_var:", data_var)

def sample(x, y):

    def cov_func(sig_var, ell, noise_var, u):
        gp = GPR_div_free.GPR_div_free(sig_var, ell, noise_var)
        U, U_grads = gp.calc_cov(u, u, data_or_test=True, calc_grad = True)
        return  U, U_grads

    kgp = kiss_gp.kiss_gp(x, u_mesh, u, cov_func)

    if sample_or_optimize:
        s = sampling.Sampling()
        with s.get_model():
            #ell = pm.Uniform('ell', 0.0, 1.0)
            #sig_var = pm.Uniform('sig_var', 0.0, 1.0)
            ell = pm.HalfNormal('ell', sd=1.0)
            sig_var = pm.HalfNormal('sig_var', sd=1.0)
            
        
        
        trace = s.sample(kgp.likelihood, [ell, sig_var], [noise_var, y], num_samples, num_chains, kgp.likelihood_grad)
    
        #print(trace['model_logp'])
        m_ell = np.mean(trace['ell'])
        m_sig_var = np.mean(trace['sig_var'])
    else:
        def lik_fn(params):
            return -kgp.likelihood(params, [noise_var, y])

        def grad_fn(params):
            return -kgp.likelihood_grad(params, [noise_var, y])

        min_loglik = None
        min_res = None
        for trial_no in np.arange(0, num_samples):
            #res = scipy.optimize.minimize(lik_fn, np.zeros(jmax*2), method='BFGS', jac=grad_fn, options={'disp': True, 'gtol':1e-7})
            #res = scipy.optimize.minimize(lik_fn, np.random.uniform(low=0.01, high=1., size=2), method='BFGS', jac=grad_fn, options={'disp': True, 'gtol':1e-7})
            res = scipy.optimize.minimize(lik_fn, [0.2, data_var], method='L-BFGS-B', jac=grad_fn, bounds = [(.1, 1.), (.1, data_var*2.)], options={'disp': True, 'gtol':1e-7})
            loglik = res['fun']
            #assert(loglik == lik_fn(res['x']))
            if min_loglik is None or loglik < min_loglik:
                min_loglik = loglik
                min_res = res
        m_ell = min_res['x'][0]
        m_sig_var = min_res['x'][1]
    return m_ell, m_sig_var
        

def calc_loglik_approx(U, W, y):
    #print np.shape(W), np.shape(y)
    (x, istop, itn, normr) = sparse.lsqr(W, y)[:4]#, x0=None, tol=1e-05, maxiter=None, M=None, callback=None)
    #print x
    L = la.cholesky(U)
    #print L
    v = la.solve(L, x)
    return -0.5 * np.dot(v.T, v) - sum(np.log(np.diag(L))) - 0.5 * n * np.log(2.0 * np.pi)
    
def calc_loglik(K, y):
    L = la.cholesky(K)
    v = la.solve(L, y)
    return -0.5 * np.dot(v.T, v) - sum(np.log(np.diag(L))) - 0.5 * n * np.log(2.0 * np.pi)


def align(x, y, y_sign, indices, n, length_scale, thetas):
    inv_ell_sq_two = -1./(2.*length_scale**2)
    #normal_dist = stats.norm(0.0, length_scale)
    
    include_idx = set(indices)  #Set is more efficient, but doesn't reorder your elements if that is desireable
    mask = np.array([(i in include_idx) for i in np.arange(0, len(x))])    
    x1 = x[~mask]
    y1 = y[~mask]
    y_sign1 = y[~mask]
    if thetas is not None:
        thetas1 = thetas[~mask]
    for i in indices:
        r = random.uniform()

        ### Vecorised ###
        #x_diff = x1 - np.repeat(np.array([x[i]]), x1.shape[0], axis=0)
        #x_diff = np.sqrt(np.sum(x_diff, axis=1))
        #p = np.exp(x_diff**2*inv_ell_sq_two)
        #np.testing.assert_almost_equal(p, p1)
        #close_by_inds = np.where(p > r)[0]
        #thetas1[close_by_inds] = thetas[i]

        #sim = np.sum(y1*np.repeat(np.array([y[i]]), y1.shape[0], axis=0), axis=1)
        #sim_indices = np.where(sim < 0)
        #y1[sim_indices] *= -1
        #y_sign1 *= -1
        
        ### Non-Vecorized ###
        
        for j in np.arange(0, n):
            if any(np.where(indices == j)[0]):
                continue
            
            x_diff = x[j] - x[i]
            x_diff = np.sqrt(np.dot(x_diff, x_diff))
            if (x_diff < 3.0*length_scale):
                #p = normal_dist.pdf(x_diff)*np.sqrt(2*np.pi)*length_scale
                # TODO: It makes no sense to compare to pdf value
                p = np.exp(x_diff**2*inv_ell_sq_two)
                #np.testing.assert_almost_equal(p, p1)
                if (r < p):
                    if thetas is not None:
                        thetas[j] = thetas[i]
                    if y_sign is not None:
                        if np.dot(y[i], y[j]) < 0:
                            y[j] = np.array(y[j])*-1
                            y_sign[j] *= -1
        
    
def get_random_indices(x, n, length_scale):
    #random_indices = np.random.choice(n, size=int(n/2), replace=False)
    random_indices = np.random.choice(n, size=n, replace=False)
    i = 0
    while i < len(random_indices):
        random_index_filter = np.ones_like(random_indices, dtype=bool)
        ri = random_indices[i]
        for j in np.arange(i + 1, len(random_indices)):
            rj = random_indices[j]
            x_diff = x[rj] - x[ri]
            if (np.dot(x_diff, x_diff) < length_scale**2):
                random_index_filter[j] = False
        random_indices = random_indices[random_index_filter]
        i += 1
    return random_indices

def algorithm_a(x, y, sig_var=None, length_scale=None):
    print(sig_var)
    y_in = np.array(y)
    y_sign = np.ones(n)
    num_positive = np.zeros(n)
    num_negative = np.zeros(n)
    loglik = None
    max_loglik = None
    y_best = None
    
    iteration = -1

    #thetas = random.uniform(size=n)
    thetas = np.ones(n)/2
    thetas = np.log(thetas)
    num_tries = 0
    
    temp = initial_temp
    
    while temp < 1.0 or max_loglik is None or num_tries % max_num_tries != 0:# or (loglik < max_loglik):# or (loglik > max_loglik + eps):
        iteration += 1
        print("num_tries", num_tries)
    
        num_tries += 1
    
        initial_param_values = []
        
        if inference and (iteration % inference_after_iter == 0):
            if temp <= 1.0:
                temp += temp_delta*temp    
#for i in np.arange(0, num_chains):
            #    #initial_m = m
            #    #initial_length_scale = length_scale
            #    #initial_param_values.append(dict(m=initial_m))
            #    initial_param_values.append(dict())
            #
            #fit = model.sampling(data=dict(x=x,N=n,y=y,noise_var=noise_var), init=initial_param_values, iter=num_iters, chains=num_chains, n_jobs=n_jobs)
            #
            #results = fit.extract()
            #loglik_samples = results['lp__']
            #print loglik_samples 
            #loglik = np.mean(loglik_samples)
            #
            #length_scale_samples = results['length_scale'];
            #length_scale = np.mean(length_scale_samples)
            #
            #sig_var_samples = results['sig_var']
            #sig_var = np.mean(sig_var_samples)
            #
            #mean_samples = results['m'];
            #mean = np.mean(mean_samples)
            
            length_scale, sig_var = sample(x, np.reshape(y, (2*n, -1)))
        else:
            if temp <= 1.0:
                temp += temp_delta*temp    
            

        if loglik is None:
            gp = GPR_div_free.GPR_div_free(sig_var, length_scale, noise_var)
            #loglik = gp.init(x, y)
            U = gp.calc_cov(u, u, data_or_test=True)
            W = utils.calc_W(u_mesh, u, x)#np.zeros((len(x1)*len(x2)*2, len(u1)*len(u2)*2))
            loglik = calc_loglik_approx(U, W, np.reshape(y, (2*n, -1)))
            
        print("sig_var=", sig_var)
        print("length_scale", length_scale)
        #print("mean", mean)
        print("loglik=", loglik, "max_loglik=", max_loglik)
        
        if max_loglik is None or loglik > max_loglik:
            num_tries = 1
            max_loglik = loglik
        
        gp = GPR_div_free.GPR_div_free(sig_var, length_scale, noise_var)
        U = gp.calc_cov(u, u, data_or_test=True)
        W = utils.calc_W(u_mesh, u, x)#np.zeros((len(x1)*len(x2)*2, len(u1)*len(u2)*2))

        y_last = np.array(y)
        y_sign_last = np.array(y_sign)
        
        if MODE == 0:
            random_indices = get_random_indices(x, n, temp*length_scale)
                
            r = random.uniform()
            for ri in random_indices:
                #r = np.log(random.uniform())
                if num_positive[ri] + num_negative[ri] <= 10:
                    theta = 0.5
                else:
                    theta = float(num_positive[ri])/(num_positive[ri] + num_negative[ri])
                thetas[ri] = np.log(theta)
                th = theta
                if th < 0.2:
                    th = 0.2
                if th > 0.8:
                    th = 0.8
                if r < th:
                    if y_sign[ri] < 0:
                        y[ri] = np.array(y[ri])*-1
                        y_sign[ri] = 1
                        #sign_change[ri] = True
                else:
                    if y_sign[ri] > 0:
                        y[ri] = np.array(y[ri])*-1
                        y_sign[ri] = -1
                        #sign_change[ri] = True
                #print(np.exp(thetas[ri]))

            align(x, y, y_sign, random_indices, n, temp*length_scale, thetas)

            bx_dis = y[:,0]*norm + bx_offset
            by_dis = y[:,1]*norm + by_offset
        
            components_plot.plot(np.reshape(bx_dis, (n1, n2)), [1, 0])
            components_plot.plot(np.reshape(by_dis, (n1, n2)), [1, 1])
            components_plot.save("components.png")

            loglik1 = calc_loglik_approx(U, W, np.reshape(y, (2*n, -1)))
    
            #for ri in random_indices:
            #    thetas_ri = thetas[ri]
            #    exp_theta = np.exp(thetas_ri)
            #    #new_theta = loglik1 - scipy.special.logsumexp(np.array([loglik1, loglik2]))
            #    #thetas[i] = new_theta
            #    #print(thetas[ri], np.exp(thetas[ri]), loglik1, loglik2)
            #    
            #    #new_theta = thetas_ri + loglik1 - scipy.special.logsumexp(np.array([loglik1, loglik2]), b=np.array([exp_theta, 1.0-exp_theta]))
            #    #thetas[ri] += learning_rate * np.sign(new_theta - thetas_ri) * thetas[ri]

            #    change = learning_rate * np.sign(loglik1 - loglik) * abs(thetas[ri])
            #    if sign_change[ri]:
            #        if y_sign[ri] > 0:
            #            thetas[ri] += change
            #        else:
            #            thetas[ri] -= change
            #            
            #    #thetas[ri] = new_theta
    
            if loglik1 > loglik:
                loglik = loglik1
            else:
                y = y_last
                y_sign = y_sign_last
                
            for ri in np.arange(0, n):
                if y_sign[ri] > 0:
                    num_positive[ri] += 1.0
                else:
                    num_negative[ri] += 1.0

        elif MODE == 1:
            random_indices = get_random_indices(x, n, temp*length_scale)

            for ri in random_indices:
                y[ri] = y_in[ri]
            loglik1 = calc_loglik_approx(U, W, np.reshape(y, (2*n, -1)))
    
            for ri in random_indices:
                y[ri] = y_in[ri]*-1
            loglik2 = calc_loglik_approx(U, W, np.reshape(y, (2*n, -1)))

            if (loglik1 > loglik or loglik2 > loglik):    
                for ri in random_indices:
                    thetas_ri = thetas[ri]
                    exp_theta = np.exp(thetas_ri)
                    #new_theta = loglik1 - scipy.special.logsumexp(np.array([loglik1, loglik2]))
                    #thetas[i] = new_theta
                    #print(thetas[ri], np.exp(thetas[ri]), loglik1, loglik2)
                    
                    #new_theta = thetas_ri + loglik1 - scipy.special.logsumexp(np.array([loglik1, loglik2]), b=np.array([exp_theta, 1.0-exp_theta]))
                    #thetas[ri] += learning_rate * np.sign(new_theta - thetas_ri) * thetas[ri]

                    thetas[ri] += learning_rate * np.sign(loglik1 - loglik2) * abs(thetas[ri])
                    #thetas[ri] = new_theta
        
                    print(thetas[ri], np.exp(thetas_ri), np.exp(thetas[ri]), loglik1-loglik2)
                    #r = np.log(random.uniform())
                    if thetas[ri] > np.log(0.5):#r < thetas[ri]
                        y[ri] = np.array(y_in[ri])
                    else:
                        y[ri] = np.array(y_in[ri])*-1
            else:
                y = y_last

            loglik = None
        else:
            for i in np.random.choice(n, size=1, replace=False):
                js = []
                for j in np.arange(0, n):
                    x_diff = x[j] - x[i]
                    if (np.dot(x_diff, x_diff) < length_scale**2):
                        js.append(j)
                for j in js:
                    y[j] = np.array(y_in[j])
                #loglik1 = gp.init(x, y)
                loglik1 = calc_loglik_approx(U, W, np.reshape(y, (2*n, -1)))
                for j in js:
                    y[j] = np.array(y_in[j])*-1
                #loglik2 = gp.init(x, y)
                loglik2 = calc_loglik_approx(U, W, np.reshape(y, (2*n, -1)))
    
                if (loglik1 > loglik or loglik2 > loglik):    
                    thetas_i = thetas[i]
                    exp_theta = np.exp(thetas_i)
                    #new_theta = loglik1 - scipy.special.logsumexp(np.array([loglik1, loglik2]))
                    #thetas[i] = new_theta
                    new_theta = thetas_i + loglik1 - scipy.special.logsumexp(np.array([loglik1, loglik2]), b=np.array([exp_theta, 1.0-exp_theta]))
                    thetas[i] += learning_rate * (new_theta - thetas_i)

                    thetas_i = thetas[i]
            
                    print(thetas_i, np.exp(thetas_i), loglik1, loglik2)
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
            loglik = None
    
        bx_dis = y[:,0]*norm + bx_offset
        by_dis = y[:,1]*norm + by_offset
    
        components_plot.plot(np.reshape(bx_dis, (n1, n2)), [1, 0])
        components_plot.plot(np.reshape(by_dis, (n1, n2)), [1, 1])
        components_plot.save("components.png")

    exp_thetas = np.exp(thetas)


    return exp_thetas, bx_dis, by_dis

    

def algorithm_b(x, y, sig_var=None, length_scale=None):
    loglik = None
    max_loglik = None

    num_tries = 0
    temp = initial_temp
    
    while (temp < 1.0 or max_loglik is None or num_tries % max_num_tries != 0 or (loglik > max_loglik + eps)):
    #while (max_loglik is None or num_tries % max_num_tries != 0 or (loglik > max_loglik + eps)):
        print("num_tries", num_tries)
        num_tries += 1
    
        #initial_param_values = []
        
        if inference:
            #for i in np.arange(0, num_chains):
            #    initial_m = mean
            #    initial_length_scale = length_scale
            #    initial_param_values.append(dict(m=initial_m))
            #
            #fit = model.sampling(data=dict(x=x,N=n,y=y,noise_var=noise_var), init=initial_param_values, iter=num_iters, chains=num_chains, n_jobs=n_jobs)
            #
            #results = fit.extract()
            #loglik_samples = results['lp__']
            #print loglik_samples 
            #loglik = np.mean(loglik_samples)
            #
            #length_scale_samples = results['length_scale'];
            #length_scale = np.mean(length_scale_samples)
            #
            #sig_var_samples = results['sig_var']
            #sig_var = np.mean(sig_var_samples)
            #
            #mean_samples = results['m'];
            #mean = np.mean(mean_samples)
            
            length_scale, sig_var = sample(x, np.reshape(y, (2*n, -1)))
            

        gp = GPR_div_free.GPR_div_free(sig_var, length_scale, noise_var)
        #loglik = gp.init(x, y)
        U = gp.calc_cov(u, u, data_or_test=True)
        W = utils.calc_W(u_mesh, u, x)#np.zeros((len(x1)*len(x2)*2, len(u1)*len(u2)*2))
        loglik = calc_loglik_approx(U, W, np.reshape(y, (2*n, -1)))
        
        print("sig_var=", sig_var)
        print("length_scale", length_scale)
        #print("mean", mean)
        print("loglik=", loglik, "max_loglik=", max_loglik)
        
        if max_loglik is None or loglik > max_loglik:
            num_tries = 1
            max_loglik = loglik
        
        gp = GPR_div_free.GPR_div_free(sig_var, length_scale, noise_var)
        U = gp.calc_cov(u, u, data_or_test=True)
        W = utils.calc_W(u_mesh, u, x)#np.zeros((len(x1)*len(x2)*2, len(u1)*len(u2)*2))
        
        
        if MODE == 0:
            # Select n/2 random indices and filter out those too close to each other
            random_indices = get_random_indices(x, n, temp*length_scale)
    
            align(x, y, None, random_indices, n, temp*length_scale, None)

            loglik1 = loglik
            y_saved = np.array(y)
            for ri in random_indices:
                y[ri] = y[ri]*-1
            loglik2 = calc_loglik_approx(U, W, np.reshape(y, (2*n, -1)))
            if loglik1 > loglik2:
                y = y_saved
        else:
            for i in np.random.choice(n, size=1, replace=False):
                loglik1 = loglik#gp.init(x, y)
                js = []
                for j in np.arange(0, n):
                    x_diff = x[j] - x[i]
                    if (np.dot(x_diff, x_diff) < length_scale**2):
                        y[j] = y[j]*-1
                        js.append(j)
                #loglik2 = gp.init(x, y)
                loglik2 = calc_loglik_approx(U, W, np.reshape(y, (2*n, -1)))
                for j in js:
                    if loglik1 > loglik2:        
                        y[j] = y[j]*-1
    
        if temp <= 1.0:
            temp += temp_delta*temp    
        
    
    return y

sig_var = None
length_scale = None
if not inference:

    sig_var=0.9*np.var(bx) + 0.9*np.var(by)
    length_scale=0.2


print("******************** Algorithm a ********************")
prob_a, field_a_x, field_a_y = algorithm_a(x, np.array(y), sig_var, length_scale)
print("******************** Algorithm b ********************")
#field_b = algorithm_b(x, np.array(y), sig_var, length_scale)
#field_b = field_a_x


Q_a_1 = ax2.quiver(x[:,0], x[:,1], field_a_x, field_a_y, units='width', color='g', linestyle=':')

#Q_a_1 = ax3.quiver(x[:,0], x[:,1], field_b[:,0], field_b[:,1], units='width', color='g', linestyle=':')

fig.savefig("field.png")
