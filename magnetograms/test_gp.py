import os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6

import sys
sys.path.append('../')
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
import scipy.interpolate as interp
import scipy.sparse.linalg as sparse
import utils
import pymc3 as pm
#import os
#import os.path
#from scipy.stats import gaussian_kde
#from sklearn.cluster import KMeans
import numpy.linalg as la
import matplotlib.pyplot as plt
import warnings
import pyhdust.triangle as triangle
import sampling
import kiss_gp

num_samples = 100
num_chains = 4

eps = 0.001
learning_rate = 1.0
max_num_tries = 20

if len(sys.argv) > 3:
    num_iters = int(sys.argv[3])
if len(sys.argv) > 4:
    num_chains = int(sys.argv[4])

n_jobs = num_chains

#model = pickle.load(open('model.pkl', 'rb'))

n1 = 10
n2 = 10
n = n1*n2
x1_range = 1.0
x2_range = 1.0
x1 = np.linspace(0, x1_range, n1)
x2 = np.linspace(0, x2_range, n2)
x_mesh = np.meshgrid(x1, x2)
x = np.dstack(x_mesh).reshape(-1, 2)
x_flat = np.reshape(x, (2*n, -1))

m1 = 5
m2 = 5
m = m1 * m2
u1 = np.linspace(0, x1_range, m1)
u2 = np.linspace(0, x2_range, m2)
u_mesh = np.meshgrid(u1, u2)
u = np.dstack(u_mesh).reshape(-1, 2)

print("u_mesh=", u_mesh)
print("u=", u)
#Optimeerida
#def get_W(u_mesh, us, xys):
#    W = np.zeros((np.shape(xys)[0], np.shape(us)[0]))
#    i = 0
#    for (x, y) in xys:
#        (u1s, u2s), (indices_x, indices_y) = get_closest(u_mesh[0][0,:], u_mesh[1][:,0], x, y)
#        coefs = bilinear_interp(u1s, u2s, x, y)
#        for j in np.arange(0, len(us)):
#            found = False
#            coef_ind = 0
#            for u1 in u1s:
#                if us[j][0] == u1:
#                    for u2 in u2s:
#                        if us[j][1] == u2:
#                            found = True
#                            break
#                        coef_ind += 1
#                    if found:
#                        break
#                else:
#                    coef_ind += len(u2s)
#            if found:print
#                W[i, j] = coefs[coef_ind]
#                #print "W=", W
#        i += 1
#    return W

#def calc_W(u_mesh, u, x):
#    W = np.zeros((np.shape(x)[0]*2, np.shape(u)[0]*2))
#    for i in np.arange(0, np.shape(x)[0]):
#        W1 = get_W(u_mesh, u, x)#, np.reshape(U[i,0::2], (len(u1), len(u2))))
#        #print np.shape(W), np.shape(W1), np.shape(W1)
#        for j in np.arange(0, np.shape(W1)[1]):
#            W[2*i,2*j] = W1[i, j]
#            W[2*i,2*j+1] = W1[i, j]
#            W[2*i+1,2*j] = W1[i, j]
#            W[2*i+1,2*j+1] = W1[i, j]
#    return W

print(x_mesh)

sig_var_train = 0.2
length_scale_train = 0.2
noise_var_train = 0.000001
mean_train = 0.0

gp_train = GPR_div_free.GPR_div_free(sig_var_train, length_scale_train, noise_var_train)
K = gp_train.calc_cov(x, x, True)
#U = gp_train.calc_cov(u, u, data_or_test=True)
#W = calc_W(u_mesh, u, x)#np.zeros((len(x1)*len(x2)*2, len(u1)*len(u2)*2))
#K = np.dot(W.T, np.dot(U, W))

for i in np.arange(0, n1):
    for j in np.arange(0, n2):
        assert(K[i, j]==K[j, i])

L = la.cholesky(K)
s = np.random.normal(0.0, 1.0, 2*n)

y = np.repeat(mean_train, 2*n) + np.dot(L, s)

y = np.reshape(y, (n, 2))
y_orig = np.array(y)
print(y_orig)


y1 = np.cos(x_mesh[0])
y2 = np.sin(x_mesh[1])

y_flat = np.reshape(y, (2*n, -1))

mean = 0.0
length_scale = 1.0
noise_var = 0.0001

print(np.shape(x))
print(np.shape(y))
print(n)

# Using KISS-GP
# https://arxiv.org/pdf/1503.01057.pdf

# define the parameters with their associated priors


def sample(x, y):

    s = sampling.Sampling()
    with s.get_model():
        ell = pm.Uniform('ell', 0.0, 1.0)
        sig_var = pm.Uniform('sig_var', 0.0, 1.0)
        
    def cov_func(sig_var, ell, noise_var, u):
        gp = GPR_div_free.GPR_div_free(sig_var, ell, noise_var)
        U, U_grads = gp.calc_cov(u, u, data_or_test=True, calc_grad = True)
        return  U, U_grads
    
    kgp = kiss_gp.kiss_gp(x, u_mesh, u, cov_func)
    
    trace = s.sample(kgp.likelihood, [ell, sig_var], [noise_var_train, y], num_samples, num_chains, kgp.likelihood_grad)

    return trace

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


loglik = None
max_loglik = None



trace = sample(x, np.reshape(y, (2*n, -1)))

length_scale = np.mean(trace['ell'])
sig_var = np.mean(trace['sig_var'])

gp = GPR_div_free.GPR_div_free(sig_var, length_scale, noise_var)
#loglik = gp.init(x, y)
U = gp.calc_cov(u, u, data_or_test=True)
W = utils.calc_W(u_mesh, u, x)#np.zeros((len(x1)*len(x2)*2, len(u1)*len(u2)*2))
loglik = calc_loglik_approx(U, W, np.reshape(y, (2*n, -1)))
    

print("sig_var", sig_var, sig_var_train)
print("length_scale", length_scale, length_scale_train)
#print("mean", mean)
print("loglik=", loglik, "max_loglik=", max_loglik)

samples = np.array([trace['ell'], trace['sig_var']]).T

fig, ax = plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(6, 6)
tmp = triangle.corner(samples[:,:], labels=['ell','sig_var'], 
                truths=[length_scale_train, sig_var_train], fig=fig)
fig.savefig("fit.png")
