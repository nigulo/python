import os
os.environ["OMP_NUM_THREADS"] = "16" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "16" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "16" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "16" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "16" # export NUMEXPR_NUM_THREADS=6

import sys
sys.path.append('../utils')
import matplotlib as mpl

mpl.use('Agg')
mpl.rcParams['figure.figsize'] = (20, 30)
#import pickle
import numpy as np
#import pylab as plt
#import pandas as pd
import cov_div_free as cov_div_free
import scipy.misc
import numpy.random as random
import scipy.sparse.linalg as sparse
import scipy.stats as stats
import scipy.special as special
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
from scipy.integrate import simps

import time
import os.path
from astropy.io import fits
from scipy.io import readsav
import scipy.signal as signal
import pickle
import depths


state_file = None#state.pkl"
if len(sys.argv) > 1:
    state_file = sys.argv[1]

mode = 2

if mode == 0:
    if state_file is None:
        state_file = 'data/IVM_AR9026.sav'
    idl_dict = readsav(state_file)
    #idl_dict = readsav('data/fan_simu_ts56.sav')
    
    lat = idl_dict['b'][0][1]
    long = idl_dict['b'][0][2]
    b_long = idl_dict['b'][0][3]
    b_trans = idl_dict['b'][0][4]
    b_azim = idl_dict['b'][0][5]
    
    #print(lat)
    #print(long)
    #print(b_long)
    #print(b_trans)
    #print(b_azim)
    
    b = np.sqrt(b_long**2 + b_trans**2)
    phi = b_azim*np.pi/180
    theta = np.arccos((b_long+1e-10)/(b+1e-10))
    
    print(b)
    print(phi)
    print(theta)

elif mode == 1:
    if state_file is None:
        state_file = 'pi-ambiguity-test/amb_turb.fits'
    hdul = fits.open(state_file)
    #hdul = fits.open('pi-ambiguity-test/amb_spot.fits')
    dat = hdul[0].data[:,::4,::4]
    b = dat[0]
    theta = dat[1]
    phi = dat[2]
else:
    
    if state_file is None:
        state_file = 'data3d.pkl'
    if os.path.isfile(state_file):
        y = pickle.load(open(state_file, 'rb'))
#    if os.path.isfile('data3d50x50x10.pkl'):
#        y = pickle.load(open('data3d50x50x10.pkl', 'rb'))
    else:
        n1, n2, n3 = 30, 30, 10
        n = n1 * n2 * n3
        x1_range, x2_range, x3_range = 1., 1., 1.
        
        x1 = np.linspace(0, x1_range, n1)
        x2 = np.linspace(0, x2_range, n2)
        x3 = np.linspace(0, x3_range, n3)
        x1_mesh, x2_mesh, x3_mesh = np.meshgrid(x1, x2, x3, indexing='ij')
        assert np.all(x1_mesh[:,0,0] == x1)
        assert np.all(x2_mesh[0,:,0] == x2)
        assert np.all(x3_mesh[0,0,:] == x3)
        x = np.stack((x1_mesh, x2_mesh, x3_mesh), axis=3)
        print("x1_mesh", x1_mesh)
        print("x2_mesh", x2_mesh)
        print("x3_mesh", x3_mesh)
        print("x", x)
        x = x.reshape(-1, 3)
        print("x", x)
        
        sig_var_train = 1.0
        length_scale_train = .1
        noise_var_train = 0.01
        mean_train = 0.

        gp_train = cov_div_free.cov_div_free(sig_var_train, length_scale_train, noise_var_train)
        K = gp_train.calc_cov(x, x, True)

        print("SIIN")
        for i in np.arange(0, n1):
            for j in np.arange(0, n2):
                assert(K[i, j]==K[j, i])
        
        L = la.cholesky(K)
        s = np.random.normal(0.0, 1.0, 3*n)
        
        y = np.repeat(mean_train, 3*n) + np.dot(L, s)
        
        y = np.reshape(y, (n1, n2, n3, 3))
        
        with open(state_file, 'wb') as f:
            pickle.dump(y, f)    
    
    ###########################################################################
    # Plotting the whole qube
    x1_range, x2_range = 1., 1.
    x1 = np.linspace(0, x1_range, y.shape[0])
    x2 = np.linspace(0, x2_range, y.shape[1])
    x_mesh = np.meshgrid(x2, x1)

    for i in np.arange(0, y.shape[3]):
        test_plot = plot.plot(nrows=1, ncols=1)
        test_plot.set_color_map('bwr')
        
        test_plot.colormap(y[:, :, i, 2])
        test_plot.vectors(x_mesh[0], x_mesh[1], y[:, :, i, 0], y[:, :, i, 1], [], units='width', color = 'k')
        test_plot.save("test_field" + str(i) +".png")
        test_plot.close()
    ###########################################################################

    bx = y[:, :, 4, 0]
    by = y[:, :, 4, 1]
    bz = y[:, :, 4, 2]
    
    ###########################################################################
    # Overwrite some of the vector for depth testing purposes
    for i in np.arange(0, y.shape[0]):
        for j in np.arange(0, y.shape[1]):
            if i == y.shape[0]//2 or j == y.shape[1]//2:
                bx[i, j] = y[i, j, 3, 0]
                by[i, j] = y[i, j, 3, 1]
                bz[i, j] = y[i, j, 3, 2]
            
    ###########################################################################
    b = np.sqrt(bx**2 + by**2 + bz**2)
    phi = np.arctan2(by, bx)
    theta = np.arccos((bz+1e-10)/(b+1e-10))


    truth_plot = plot.plot(nrows=1, ncols=3)
    truth_plot.set_color_map('bwr')
    
    truth_plot.colormap(bx, [0])
    truth_plot.colormap(by, [1])
    truth_plot.colormap(phi, [2])
    
    truth_plot.save("truth.png")
    truth_plot.close()



inference = True
sample_or_optimize = False
num_samples = 1
num_chains = 4
inference_after_iter = 20

eps = 0.001
learning_rate = 0.1
max_num_tries = 10000
initial_temp = 0.1
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


bz = b*np.cos(theta)
bz_smooth = signal.convolve2d(bz, np.ones((5,5)), mode = 'same') #Smooth it a little
dbzy = bz_smooth[1:,:-1]-bz_smooth[:-1,:-1]
dbzx = bz_smooth[:-1,1:]-bz_smooth[:-1,:-1]


b = b[:-1,:-1]
bz = bz[:-1,:-1]
theta = theta[:-1,:-1]
phi = phi[:-1,:-1]
bxy = b*np.sin(theta)
bx = bxy*np.cos(phi)
by = bxy*np.sin(phi)

n1 -= 1
n2 -= 1
n = n1*n2

x1 = np.linspace(0, x1_range, n1)
x2 = np.linspace(0, x2_range, n2)
x_mesh = np.meshgrid(x2, x1)
x_grid = np.dstack(x_mesh)
x = x_grid.reshape(-1, 2)
x_flat = np.reshape(x, (2*n, -1))

d = depths.depths(x_grid, bx, by, bz)
d.estimate()


if mode == 2:
    
    truth_plot = plot.plot(nrows=1, ncols=1)
    truth_plot.set_color_map('bwr')
    
    truth_plot.colormap(bz)
    truth_plot.vectors(x_mesh[0], x_mesh[1], bx, by, [], units='width', color = 'k')
    truth_plot.save("truth_field.png")
    truth_plot.close()


def calc_p(x, y):
    xs = np.dstack(x_mesh)
    #print("xs:",xs)
    r = xs - np.array([x, y])
    r = np.sqrt(np.sum(r*r, axis=2))
    indices = np.where(r == 0.)
    bz1 = np.array(bz)
    bz1[indices] = 0.
    r[indices] = 1.
    z = bz/r
    p = simps(simps(z, x2), x1)
    return -p/2./np.pi

p = np.zeros_like(bz)
for i in np.arange(0, len(x1)):
    for j in np.arange(0, len(x2)):
        p[i, j] = calc_p(x1[i], x2[j])
print("p:", p)

bx = np.reshape(bx, n)
by = np.reshape(by, n)
bz = np.reshape(bz, n)
dbzx = np.reshape(dbzx, n)
dbzy = np.reshape(dbzy, n)
bxy = np.reshape(bz, n)

m = m1 * m2
u1 = np.linspace(0, x1_range, m1)
u2 = np.linspace(0, x2_range, m2)
u_mesh = np.meshgrid(u1, u2)
u = np.dstack(u_mesh).reshape(-1, 2)
    
print("u_mesh=", u_mesh)
print("u=", u)
print(x_mesh)

#for i in np.arange(0, n):
    #if np.random.uniform() < 0.5:
    #    bx[i] *= -1
    #    by[i] *= -1

#    bx[i] = abs(bx[i])
#    by[i] = abs(by[i])


bx_temp = np.reshape(bx, (n1, n2))
by_temp = np.reshape(by, (n1, n2))
bx_smooth = signal.convolve2d(bx_temp, np.ones((5,5)), mode = 'same') #Smooth it a little
by_smooth = signal.convolve2d(by_temp, np.ones((5,5)), mode = 'same') #Smooth it a little
bx_smooth = np.reshape(bx_smooth, n)
by_smooth = np.reshape(by_smooth, n)

alpha = 0.0
bx_offset = np.zeros_like(bx)#bxy + bz
by_offset = np.zeros_like(by)#bxy + bz
#bx_offset = dbzx*np.std(bx)/np.std(dbzx)
#by_offset = dbzy*np.std(by)/np.std(dbzy)

#bx_offset1 = -alpha*by_smooth
#by_offset1 = -alpha*bx_smooth
bx_offset1 = -alpha*dbzy*np.std(by)/np.std(dbzy)
by_offset1 = -alpha*dbzx*np.std(bx)/np.std(dbzx)


bx1 = bx - bx_offset - bx_offset1
by1 = by - by_offset - by_offset1
#bx_mean = np.mean(bx)
#by_mean = np.mean(by)
#bx -= bx_mean
#by -= by_mean
#bx_offset += bx_mean
#by_offset += by_mean
norm = 1.#np.std(np.sqrt(bx**2+by**2))
bx1 /= norm
by1 /= norm
y = np.column_stack((bx1, by1))


y_orig = np.array(y)
print(y_orig)
print(y.shape)

def do_plots(y):
    components_plot = plot.plot(nrows=2, ncols=3)
    components_plot.set_color_map('bwr')
    
    bx_norm = y_orig[:,0]*norm+bx_offset+bx_offset1
    by_norm = y_orig[:,1]*norm+by_offset+by_offset1
    
    components_plot.colormap(np.reshape(bx_norm, (n1, n2)), [0, 0])
    components_plot.colormap(np.reshape(by_norm, (n1, n2)), [0, 1])
    components_plot.colormap(np.reshape(np.arctan2(by_norm, bx_norm), (n1, n2)), [0, 2])
    
    components_plot2 = plot.plot(nrows=2, ncols=2)
    components_plot2.set_color_map('bwr')
    components_plot2.colormap(np.reshape(dbzx, (n1, n2)), [0, 0])
    components_plot2.colormap(np.reshape(dbzy, (n1, n2)), [0, 1])
    

    if y is not None:    
        bx_dis = y[:,0]*norm + bx_offset+bx_offset1
        by_dis = y[:,1]*norm + by_offset+by_offset1
       
        components_plot.colormap(np.reshape(bx_dis, (n1, n2)), [1, 0])
        components_plot.colormap(np.reshape(by_dis, (n1, n2)), [1, 1])
        components_plot.colormap(np.reshape(np.arctan2(by_dis, bx_dis), (n1, n2)), [1, 2])
    
        components_plot2.colormap(np.reshape(y[:,0], (n1, n2)), [1, 0])
        components_plot2.colormap(np.reshape(y[:,1], (n1, n2)), [1, 1])

    components_plot.save("components.png")
    components_plot2.save("components2.png")
    
    components_plot.close()
    components_plot2.close()


bx_norm = y[:,0]*norm+bx_offset+bx_offset1
by_norm = y[:,1]*norm+by_offset+by_offset1

energy = np.sum(bx**2 + by**2)
np.testing.assert_array_almost_equal(np.sum(bx_norm**2 + by_norm**2), energy)

do_plots(None)

print(np.shape(x))
print(np.shape(y))
print(n)

# Using KISS-GP
# https://arxiv.org/pdf/1503.01057.pdf

# define the parameters with their associated priors

data_var = (np.var(y[:,0]) + np.var(y[:,1]))
#noise_var = .01*data_var
#forced_noise_var = data_var - (np.var(np.abs(y[:,0])) + np.var(np.abs(y[:,1])))/2
#data_var -= forced_noise_var
print("data_var:", data_var)



def sample(x, y):

    def cov_func(sig_var, ell, noise_var, u):
        gp = cov_div_free.cov_div_free(sig_var, ell, noise_var)
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
            noise_var = pm.HalfNormal('noise_var', sd=1.0)
            
        
        
        trace = s.sample(kgp.likelihood2, [ell, sig_var], [noise_var, y], num_samples, num_chains, kgp.likelihood_grad2)
    
        #print(trace['model_logp'])
        m_ell = np.mean(trace['ell'])
        m_sig_var = np.mean(trace['sig_var'])
        m_noise_var = np.mean(trace['noise_var'])
    else:
        def lik_fn(params):
            return -kgp.likelihood2(params, [y])

        def grad_fn(params):
            return -kgp.likelihood_grad2(params, [y])

        min_loglik = None
        min_res = None
        for trial_no in np.arange(0, num_samples):
            ell_min = 0.2#.05
            ell_max = 0.5#1.
            sig_var_min = data_var*.1
            sig_var_max = data_var*2.
            noise_var_min = data_var*0.0001#*.001
            noise_var_max = data_var*0.01#*.5
            ell_init = random.uniform(ell_min, ell_max)
            sig_var_init = random.uniform(sig_var_min, sig_var_max)
            noise_var_init = random.uniform(noise_var_min, noise_var_max)
            #res = scipy.optimize.minimize(lik_fn, [.5, data_var, data_var*.015], method='L-BFGS-B', jac=grad_fn, bounds = [(.1, 1.), (data_var*.1, data_var*2.), (data_var*.01, data_var*.02)], options={'disp': True, 'gtol':1e-7})
            res = scipy.optimize.minimize(lik_fn, [ell_init, sig_var_init, noise_var_init], method='L-BFGS-B', jac=grad_fn, bounds = [(ell_min, ell_max), (sig_var_min, sig_var_max), (noise_var_min, noise_var_max)], options={'disp': True, 'gtol':1e-7})
            loglik = res['fun']
            #assert(loglik == lik_fn(res['x']))
            if min_loglik is None or loglik < min_loglik:
                min_loglik = loglik
                min_res = res
        m_ell = min_res['x'][0]
        m_sig_var = min_res['x'][1]
        m_noise_var = min_res['x'][2]
    return m_ell, m_sig_var, m_noise_var
        

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


def recalc_offsets(y, bx_offset1, by_offset1):
    y = y*norm
    y[:,0] += bx_offset
    y[:,1] += by_offset

    y[:,0] += bx_offset1
    y[:,1] += by_offset1

    bx = np.reshape(y[:,0], (n1, n2))
    by = np.reshape(y[:,1], (n1, n2))
    bx_smooth = signal.convolve2d(bx, np.ones((5,5)), mode = 'same') #Smooth it a little
    by_smooth = signal.convolve2d(by, np.ones((5,5)), mode = 'same') #Smooth it a little
    bx_smooth = np.reshape(bx_smooth, n)
    by_smooth = np.reshape(by_smooth, n)

    bx_offset1 = -alpha*by_smooth
    by_offset1 = -alpha*bx_smooth

    y[:,0] -= bx_offset1
    y[:,1] -= by_offset1

    y[:,0] -= bx_offset
    y[:,1] -= by_offset
    y/=norm

    return (y, bx_offset1, by_offset1)

def reverse(y, y_sign, ii, bx_offset1, by_offset1):
    y[ii]*=norm
    y[ii,0] += bx_offset[ii]
    y[ii,1] += by_offset[ii]

    y[ii,0] += bx_offset1[ii]
    y[ii,1] += by_offset1[ii]
    
    y[ii,:] *= -1

    y[ii,0] -= bx_offset1[ii]
    y[ii,1] -= by_offset1[ii]

    y[ii,0] -= bx_offset[ii]
    y[ii,1] -= by_offset[ii]
    y[ii]/=norm
    y_sign[ii] *= -1

    #np.testing.assert_almost_equal(yc_sign, y_sign)
    #np.testing.assert_almost_equal(yc, y)

    
def get_b(y, ii):
    y1 = y * norm
    y1[:,0] += bx_offset[ii]+bx_offset1[ii]
    y1[:,1] += by_offset[ii]+by_offset1[ii]
    return y1

def get_probs(thetas, y):
    b = np.sqrt(np.sum(y*y, axis=1))
    return b/np.sum(b)
    
    #p = np.exp(thetas)
    #p -= 0.5
    #p = np.abs(p)*2.
    #p[p > 0.8] = 0.8
    #p[p < 0.2] = 0.2
    #p /= np.sum(p)
    #return p    

def align2(x, y, y_sign, indices, n, length_scale, sig_var, noise_var, thetas, num_positive, num_negative, bx_offset1, by_offset1):
    #inv_ell_sq_two = -1./(2.*length_scale**2)
    inv_ell_sq_two = 1./(2.*length_scale)
    #normal_dist = stats.norm(0.0, length_scale)
    
    include_idx = set(indices)  #Set is more efficient, but doesn't reorder your elements if that is desireable
    mask = np.array([(i in include_idx) for i in np.arange(0, len(x))])
    #used_js = set()
    mask = np.where(~mask)[0]
    affected_indices = set()
    gp = cov_div_free.cov_div_free(sig_var, length_scale, noise_var)
    for i in indices:

        #######################################################################
        # Determine the points wich lie in the vicinity of the point i
        r = random.uniform()
        x_diff = x[mask] - np.repeat(np.array([x[i]]), x[mask].shape[0], axis=0)
        x_diff = np.sum(x_diff**2, axis=1)
        p = .5*(1+special.erf(np.sqrt(x_diff)*inv_ell_sq_two))
        i1 = np.where(p > 0.5)[0]
        p[i1] = 1. - p[i1]
        p *= 2.
        inds1 = np.where(p >= r)[0]
        #######################################################################
        
        affected_indices.add(i)
        affected_indices.update(inds1)
        
        p1 = get_probs(thetas[inds1], y[inds1])

        inds_train = np.array([i])
        
        if len(p1) > 0:
            inds_train = np.concatenate((inds_train, np.random.choice(inds1, min(2, len(inds1)), p=p1)))
        inds1 = np.setdiff1d(inds1, inds_train)
        
        #######################################################################
        # Do aligning of the chosen training vectors
        # based on most likely direction
        r = random.uniform()
        for ri in inds_train:
            #r = np.log(random.uniform())
            if num_positive[ri] + num_negative[ri] < 1:
                theta = 0.5
            else:
                theta = float(num_positive[ri])/(num_positive[ri] + num_negative[ri])
            th = theta
            if th < 0.2:
                th = 0.2
            if th > 0.8:
                th = 0.8
            if r < th:
                if y_sign[ri] < 0:
                    reverse(y, y_sign, ri, bx_offset1, by_offset1)
                    y_sign[ri] = 1 # overwrite sign
                    #sign_change[ri] = True
            else:
                if y_sign[ri] > 0:
                    reverse(y, y_sign, ri, bx_offset1, by_offset1)
                    y_sign[ri] = -1 # overwrite sign
                    #sign_change[ri] = True
            #print(np.exp(thetas[ri]))
        #######################################################################
        
        while len(inds1) > 0:
            inds_test = inds1[:min(1000, len(inds1))]
            inds1 = inds1[min(1000, len(inds1)):]
            #while len(inds) > 1000:
            #    inds = np.random.choice(inds, 1000) 
            #inds_train = inds[:int(len(inds)/10)]
            
            #inds_test = inds[int(len(inds)/10):]
            #print(len(inds_train), len(inds_test))
    
            x_train = x[inds_train]
            y_train = y[inds_train]
    
            x_test = x[mask][inds_test]
            y_test_obs = y[mask][inds_test]
    
            y_train_flat = np.reshape(y_train, (2*len(y_train), -1))
            #loglik = gp.init(x, y)
            #print(x_train.shape, y_train_flat.shape)
            gp.init(x_train, y_train_flat)
            
            y_test_mean = gp.fit(x_test, calc_var = False)
            y_test_mean = np.reshape(y_test_mean, y_test_obs.shape)
            b_test_mean = get_b(y_test_mean, mask[inds_test])
            b_test_obs = get_b(y_test_obs, mask[inds_test])

            sim = np.sum(b_test_obs*b_test_mean, axis=1)
            #sim = np.sum(y_test_obs*np.reshape(y_test_mean, y_test_obs.shape), axis=1)
    
            sim_indices = np.where(sim < 0.)[0]
            #y_copy=np.array(y)
            #y_sign_copy=np.array(y_sign)
            reverse(y, y_sign, mask[inds_test][sim_indices], bx_offset1, by_offset1)
            #y_copy1=np.array(y)
            #y_sign_copy1=np.array(y_sign)
            #reverse(y_copy1, y_sign_copy1, mask[inds_test][sim_indices])
            #np.testing.assert_almost_equal(y_sign_copy, y_sign_copy1)
            #np.testing.assert_almost_equal(y_copy, y_copy1)
    return affected_indices
'''
#################
# Simple aligning
#################
def align(x, y, y_sign, indices, n, length_scale, thetas):
    inv_ell_sq_two = -1./(2.*length_scale**2)
    #normal_dist = stats.norm(0.0, length_scale)
    
    include_idx = set(indices)  #Set is more efficient, but doesn't reorder your elements if that is desireable
    mask = np.array([(i in include_idx) for i in np.arange(0, len(x))])
    #used_js = set()
    mask = np.where(~mask)[0]
    
    for i in indices:
        r = random.uniform()

        ### Vectorized ###
        x_diff = x[mask] - np.repeat(np.array([x[i]]), x[mask].shape[0], axis=0)
        x_diff = np.sum(x_diff**2, axis=1)
        p = np.exp(x_diff*inv_ell_sq_two)
        #np.testing.assert_almost_equal(p, p1)
        close_by_inds = np.where(p >= r)[0]
        thetas[close_by_inds] = thetas[i]

        y_close_by = y[mask][close_by_inds]
        
        b = get_b(y)
        b_close_by = b[mask][close_by_inds]

        sim = np.sum(b_close_by*np.repeat(np.array([b[i]]), b_close_by.shape[0], axis=0), axis=1)
        #sim = np.sum(y_close_by*np.repeat(np.array([y[i]]), y_close_by.shape[0], axis=0), axis=1)
        #sim = np.sum(y_close_by*np.repeat(np.array([y[i]]), y_close_by.shape[0], axis=0), axis=1)
        sim_indices = np.where(sim < 0)[0]
        reverse(y, y_sign, mask[close_by_inds][sim_indices])
'''

def get_random_indices(x, n, length_scale, thetas, y):
    p = get_probs(thetas, y)
    #random_indices = np.random.choice(n, size=int(n/2), replace=False)
    random_indices = np.random.choice(n, min(max(1, int(1./(np.pi*length_scale**2))), 20), replace=False, p=p)
    i = 0
    while i < len(random_indices):
        random_index_filter = np.ones_like(random_indices, dtype=bool)
        ri = random_indices[i]
        for rj in np.arange(0, n):
            x_diff = x[rj] - x[ri]
            if (np.dot(x_diff, x_diff) < (3.*length_scale)**2):
                for j in np.arange(i + 1, len(random_indices)):
                    if random_indices[j] == rj:
                        random_index_filter[j] = False
        random_indices = random_indices[random_index_filter]
        i += 1
    return random_indices

def algorithm_a(x, y, sig_var, length_scale, noise_var, bx_offset1, by_offset1):
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
    thetas = np.ones(n)/2.
    thetas = np.log(thetas)
    num_tries = 0
    
    temp = initial_temp
    
    while  max_loglik is None or num_tries % max_num_tries != 0:# or (loglik < max_loglik):# or (loglik > max_loglik + eps):
        iteration += 1
        print("num_tries", num_tries)
    
        num_tries += 1
    
        temp = random.uniform(initial_temp, 0.5)
        if inference and (iteration % inference_after_iter == 0):
            #if temp <= 1.0:
            #    temp += temp_delta*temp    
            
            length_scale, sig_var, noise_var = sample(x, np.reshape(y, (2*n, -1)))
        #else:
            #if temp <= 1.0:
            #    temp += temp_delta*temp    
            

        if loglik is None:
            gp = cov_div_free.cov_div_free(sig_var, length_scale, noise_var)
            #loglik = gp.init(x, y)
            U = gp.calc_cov(u, u, data_or_test=True)
            W = utils.calc_W(u_mesh, u, x)#np.zeros((len(x1)*len(x2)*2, len(u1)*len(u2)*2))
            loglik = calc_loglik_approx(U, W, np.reshape(y, (2*n, -1)))
            
        print("sig_var=", sig_var)
        print("noise_var=", noise_var)
        print("length_scale", length_scale)
        #print("mean", mean)
        print("loglik=", loglik, "max_loglik=", max_loglik)
        
        if max_loglik is None or loglik > max_loglik:
            num_tries = 1
            max_loglik = loglik
        
        gp = cov_div_free.cov_div_free(sig_var, length_scale, noise_var)
        U = gp.calc_cov(u, u, data_or_test=True)
        W = utils.calc_W(u_mesh, u, x)#np.zeros((len(x1)*len(x2)*2, len(u1)*len(u2)*2))

        y_last = np.array(y)
        y_sign_last = np.array(y_sign)
        
        random_indices = get_random_indices(x, n, temp*length_scale, thetas, y)


        start = time.time()

        affected_indices = align2(x, y, y_sign, random_indices, n, temp*length_scale, sig_var, noise_var, thetas, num_positive, num_negative, bx_offset1, by_offset1)
        #align(x, y, y_sign, random_indices, n, temp*length_scale, thetas)
        end = time.time()
        print("Align took: " + str(end - start))


        do_plots(y)    

        start = time.time()
        loglik1 = calc_loglik_approx(U, W, np.reshape(y, (2*n, -1)))
        end = time.time()
        print("Inference took: " + str(end - start))

        print("loglik1=", loglik1)

        if loglik1 > loglik:
            loglik = loglik1
            bx_dis = y[:,0]*norm + bx_offset+bx_offset1
            by_dis = y[:,1]*norm + by_offset+by_offset1
        
            energy1 = np.sum(bx_dis**2 + by_dis**2)
            print("Energy diff before:", (energy1-energy)/energy)
            #y, bx_offset1, by_offset1 = recalc_offsets(y, bx_offset1, by_offset1)


            for ri in np.arange(0, n):
                if ri in affected_indices:
                    if y_sign[ri] > 0:
                        num_positive[ri] += 1.0
                    else:
                        num_negative[ri] += 1.0
                    if num_positive[ri] + num_negative[ri] >= 10:
                        theta = float(num_positive[ri])/(num_positive[ri] + num_negative[ri])
                        thetas[ri] = np.log(theta)
                    print("num_positive, num_negative:", num_positive[ri], num_negative[ri])
        else:
            y = y_last
            y_sign = y_sign_last
            

        bx_dis = y[:,0]*norm + bx_offset+bx_offset1
        by_dis = y[:,1]*norm + by_offset+by_offset1
    
        energy1 = np.sum(bx_dis**2 + by_dis**2)
        print("Energy diff:", (energy1-energy)/energy)
        #np.testing.assert_almost_equal(energy1, energy)

        do_plots(y)


    do_plots(y)

    exp_thetas = np.exp(thetas)


    return exp_thetas, bx_dis, by_dis

    
sig_var = None
length_scale = None
noise_var = None
if not inference:

    sig_var=0.9*np.var(bx) + 0.9*np.var(by)
    length_scale=0.2
    noise_var=0.1*sig_var


prob_a, field_a_x, field_a_y = algorithm_a(x, np.array(y), sig_var, length_scale, noise_var, bx_offset1, by_offset1)


