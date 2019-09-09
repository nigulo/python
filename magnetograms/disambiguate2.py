import sys
sys.path.append('../utils')
sys.path.append('..')
import config
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
import misc
import pymc3 as pm
#import os
#import os.path
#from scipy.stats import gaussian_kde
#from sklearn.cluster import KMeans
import numpy.linalg as la
import matplotlib.pyplot as plt
import sampling
from scipy.integrate import simps

import time
import os.path
from astropy.io import fits
from scipy.io import readsav
import scipy.signal as signal
import pickle


state_file = 'data3d.pkl'
#state_file = 'data/IVM_AR9026.sav'
#state_file = 'pi-ambiguity-test/amb_turb.fits'

num_x = 1
num_y = 1
x_no = 0
y_no = 0

if len(sys.argv) > 1:
    state_file = sys.argv[1]
if len(sys.argv) > 2:
    num_x = int(sys.argv[2])
if len(sys.argv) > 3:
    num_y = int(sys.argv[3])
if len(sys.argv) > 4:
    x_no = int(sys.argv[4])
if len(sys.argv) > 5:
    y_no = int(sys.argv[5])

mode = 2

subsample = 1000000
num_reps = 1
num_layers = 3

if state_file[-4:] == '.sav':
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

elif state_file[-5:] == '.fits':
    if state_file is None:
        state_file = 'pi-ambiguity-test/amb_turb.fits'
    hdul = fits.open(state_file)
    #hdul = fits.open('pi-ambiguity-test/amb_spot.fits')
    dat = hdul[0].data[:,::4,::4]
    b = dat[0]
    theta = dat[1]
    phi = dat[2]
elif state_file[-4:] == '.pkl':
    
    if state_file is None:
        state_file = 'data3d.pkl'
    if os.path.isfile(state_file):
        y = pickle.load(open(state_file, 'rb'))
#    if os.path.isfile('data3d50x50x10.pkl'):
#        y = pickle.load(open('data3d50x50x10.pkl', 'rb'))
    else:
        n1, n2, n3 = 30, 30, 10
        n = n1 * n2 * n3
        x1_range, x2_range, x3_range = 1., 1., .33
        
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

    for i in np.arange(0, y.shape[2]):
        test_plot = plot.plot(nrows=1, ncols=1)
        test_plot.set_color_map('bwr')
        
        test_plot.colormap(y[:, :, i, 2])
        test_plot.vectors(x_mesh[0], x_mesh[1], y[:, :, i, 0], y[:, :, i, 1], [], units='width', color = 'k')
        test_plot.save("test_field" + str(i) +".png")
        test_plot.close()
    ###########################################################################

    n1 = y.shape[0]//num_x
    n2 = y.shape[1]//num_y
    x_start = n1*x_no
    x_end = min(x_start + n1, y.shape[0])
    y_start = n2*y_no
    y_end = min(y_start + n1, y.shape[1])
    

    bx = y[x_start:x_end, y_start:y_end, :num_layers, 0]
    by = y[x_start:x_end, y_start:y_end, :num_layers, 1]
    bz = y[x_start:x_end, y_start:y_end, :num_layers, 2]
    
    ###########################################################################
    # Overwrite some of the vector for depth testing purposes
    #for i in np.arange(0, y.shape[0]):
    #    for j in np.arange(0, y.shape[1]):
    #        #if i == y.shape[0]//2 or j == y.shape[1]//2:
    #        if i == j or y.shape[0] - i == j:
    #            bx[i, j] = y[i, j, 1, 0]
    #            by[i, j] = y[i, j, 1, 1]
    #            bz[i, j] = y[i, j, 1, 2]
    ###########################################################################
    b = np.sqrt(bx**2 + by**2 + bz**2)
    phi = np.arctan2(by, bx)
    theta = np.arccos((bz+1e-10)/(b+1e-10))


    num_layers = bx.shape[-1]

    truth_plot = plot.plot(nrows=num_layers, ncols=3)
    for layer in np.arange(0, num_layers):
        truth_plot.set_color_map('bwr')
        
        truth_plot.colormap(bx[:, :, layer], [layer, 0])
        truth_plot.colormap(by[:, :, layer], [layer, 1])
        truth_plot.colormap(phi[:, :, layer], [layer, 2])
    
    truth_plot.save("truth.png")
    truth_plot.close()

else:
    print("Unknown input file type")
    sys.exit(1)

inference = False
sample_or_optimize = False
num_samples = 1
num_chains = 4
inference_after_iter = 20

eps = 0.001
learning_rate = 0.1
max_num_tries = 20
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
n3 = b.shape[2]
x1_range = 1.0
x2_range = 1.0
x3_range = x1_range*n3/n1


bz = b*np.cos(theta)
bxy = b*np.sin(theta)
bx = bxy*np.cos(phi)
by = bxy*np.sin(phi)

n = n1*n2*n3

x1 = np.linspace(0, x1_range, n1)
x2 = np.linspace(0, x2_range, n2)
x3 = np.linspace(0, x3_range, n3)

x1_mesh, x2_mesh, x3_mesh = np.meshgrid(x1, x2, x3, indexing='ij')
x_grid = np.stack((x1_mesh, x2_mesh, x3_mesh), axis=3)
x = x_grid.reshape(-1, 3)
x_flat = np.reshape(x, (3*n, -1))

###############################################################################


if mode == 2:
    
    truth_plot = plot.plot(nrows=n3, ncols=1)
    truth_plot.set_color_map('bwr')
    for layer in np.arange(0, n3):
        truth_plot.colormap(bz[:, :, layer], ax_index = [layer])
        truth_plot.vectors(x1_mesh, x2_mesh, bx[:, :, layer], by[:, :, layer], ax_index = [layer], units='width', color = 'k')
    truth_plot.save("truth_field_" + str(x_no) + "_" + str(y_no) + ".png")
    truth_plot.close()


bx = np.reshape(bx, n)
by = np.reshape(by, n)
bz = np.reshape(bz, n)


y = np.column_stack((bx, by, bz))


y_orig = np.array(y)
print(y_orig)
print(y.shape)


# Align all the transverse components either randomly or identically
for i in np.arange(0, n):
    if np.random.uniform() < 0.5:
        y[i, :2] *= -1
    #y[i, :2] = np.abs(y[i, :2])


def do_plots(y, title = None):
    bx_orig = y_orig[:, 0]    
    by_orig = y_orig[:, 1]
    bx_orig = np.reshape(bx_orig, (n1, n2, n3))
    by_orig = np.reshape(by_orig, (n1, n2, n3))

    if y is not None:
        bx_dis = y[:, 0]
        by_dis = y[:, 1]
        bx_dis = np.reshape(bx_dis, (n1, n2, n3))
        by_dis = np.reshape(by_dis, (n1, n2, n3))

    for layer in np.arange(0, n3):
        components_plot = plot.plot(nrows=2, ncols=3, title = title)
        components_plot.set_color_map('bwr')
        
        components_plot.colormap(bx_orig[:, :, layer], [0, 0])
        components_plot.colormap(by_orig[:, :, layer], [0, 1])
        components_plot.colormap(np.reshape(np.arctan2(by_orig[:, :, layer], bx_orig[:, :, layer]), (n1, n2)), [0, 2])
        
        if y is not None:    
            components_plot.colormap(bx_dis[:, :, layer], [1, 0])
            components_plot.colormap(by_dis[:, :, layer], [1, 1])
            components_plot.colormap(np.reshape(np.arctan2(by_dis[:, :, layer], bx_dis[:, :, layer]), (n1, n2)), [1, 2])
        
        components_plot.save("components_" + str(x_no) + "_" + str(y_no) + "_" + str(layer) +".png")
        components_plot.close()


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

    def lik_fn(theta):
        sig_var = theta[0]
        ell = theta[1]
        noise_var = theta[2]
        gp = cov_div_free.cov_div_free(sig_var, ell, noise_var)
        return gp.loglik_approx(x, y, subsample=subsample)

    lik_grad = None

    if sample_or_optimize:
        s = sampling.Sampling()
        with s.get_model():
            #ell = pm.Uniform('ell', 0.0, 1.0)
            #sig_var = pm.Uniform('sig_var', 0.0, 1.0)
            ell = pm.HalfNormal('ell', sd=1.0)
            sig_var = pm.HalfNormal('sig_var', sd=1.0)
            noise_var = pm.HalfNormal('noise_var', sd=1.0)
        
        trace = s.sample(lik_fn, [sig_var, ell, noise_var], [], num_samples, num_chains, lik_grad)
    
        #print(trace['model_logp'])
        m_ell = np.mean(trace['ell'])
        m_sig_var = np.mean(trace['sig_var'])
        m_noise_var = np.mean(trace['noise_var'])

    else:

        min_loglik = None
        min_res = None
        for trial_no in np.arange(0, num_samples):
            ell_min = 0.5#.05
            ell_max = 2.#1.
            sig_var_min = data_var*.1
            sig_var_max = data_var*2.
            noise_var_min = data_var*0.0001#*.001
            noise_var_max = data_var*0.01#*.5


            ell_init = random.uniform(ell_min, ell_max)
            sig_var_init = random.uniform(sig_var_min, sig_var_max)
            noise_var_init = random.uniform(noise_var_min, noise_var_max)

            #res = scipy.optimize.minimize(lik_fn, [.5, data_var, data_var*.015], method='L-BFGS-B', jac=grad_fn, bounds = [(.1, 1.), (data_var*.1, data_var*2.), (data_var*.01, data_var*.02)], options={'disp': True, 'gtol':1e-7})
            res = scipy.optimize.minimize(lik_fn, [sig_var_init, ell_init, noise_var_init], method='L-BFGS-B', jac=lik_grad, bounds = [(sig_var_min, sig_var_max), (ell_min, ell_max), (noise_var_min, noise_var_max)], options={'disp': True, 'gtol':1e-7})
            loglik = res['fun']
            #assert(loglik == lik_fn(res['x']))
            if min_loglik is None or loglik < min_loglik:
                min_loglik = loglik
                min_res = res
        m_sig_var = min_res['x'][0]
        m_ell = min_res['x'][1]
        m_noise_var = min_res['x'][2]
    return m_ell, m_sig_var, m_noise_var
        



def reverse(y, y_sign, ii):
    #y_old = np.array(y)
    y[ii,:2] *= -1
    y_sign[ii] *= -1
    
    #np.testing.assert_almost_equal(np.sum(y_old[ii]**2), np.sum(y[ii]**2))
    #np.testing.assert_almost_equal(y_old[ii,:2] + y[ii,:2], np.zeros_like(y_old[ii,:2]))


def get_probs(thetas, y):
    b = np.sqrt(np.sum(y[:,:2]*y[:,:2], axis=1))
    return b/np.sum(b)
    
'''
def align2(x, y, y_sign, indices, n, length_scale, sig_var, noise_var, thetas, num_positive, num_negative):
    print("len(indices)", len(indices))
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
        # Determine the points which lie in the vicinity of the point i
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
                    reverse(y, y_sign, ri)
                    y_sign[ri] = 1 # overwrite sign
                    #sign_change[ri] = True
            else:
                if y_sign[ri] > 0:
                    reverse(y, y_sign, ri)
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
    
            y_train_flat = np.reshape(y_train, (3*len(y_train), -1))
            #loglik = gp.init(x, y)
            #print(x_train.shape, y_train_flat.shape)
            print("x_train", x_train.shape)
            gp.init(x_train, y_train_flat)
            
            y_test_mean = gp.fit(x_test, calc_var = False)
            y_test_mean = np.reshape(y_test_mean, y_test_obs.shape)

            sim = np.sum(y_test_obs*y_test_mean, axis=1)
            #sim = np.sum(y_test_obs*np.reshape(y_test_mean, y_test_obs.shape), axis=1)
    
            sim_indices = np.where(sim < 0.)[0]
            #y_copy=np.array(y)
            #y_sign_copy=np.array(y_sign)
            reverse(y, y_sign, mask[inds_test][sim_indices])
            #y_copy1=np.array(y)
            #y_sign_copy1=np.array(y_sign)
            #reverse(y_copy1, y_sign_copy1, mask[inds_test][sim_indices])
            #np.testing.assert_almost_equal(y_sign_copy, y_sign_copy1)
            #np.testing.assert_almost_equal(y_copy, y_copy1)
    return affected_indices
'''

class disambiguator():
    
    def __init__(self, x, y, sig_var, length_scale, noise_var, approx=True):
        self.x = x
        self.y = y
        self.sig_var = sig_var
        self.length_scale = length_scale
        self.noise_var = noise_var
        self.n = len(x)
        assert(self.n == n1*n2*n3)
        
        self.y_sign = np.ones(n)
        thetas = np.ones(self.n)/2.
        self.thetas = np.log(thetas)

        self.num_positive = np.zeros(self.n)
        self.num_negative = np.zeros(self.n)
        
        self.approx = approx
        
        gp = cov_div_free.cov_div_free(self.sig_var, self.length_scale, self.noise_var)
        if self.approx:
            self.true_loglik = 0.
            for i in np.arange(0, num_reps):
                self.true_loglik += gp.loglik_approx(self.x, np.reshape(y_orig, (3*self.n, -1)), subsample=subsample)
                #if (self.true_loglik is None or true_loglik > self.true_loglik):
                #    self.true_loglik = true_loglik
            self.true_loglik /= num_reps
        else:
            self.true_loglik = gp.loglik(self.x, np.reshape(y_orig, (3*self.n, -1)))

    def loglik(self):
        gp = cov_div_free.cov_div_free(self.sig_var, self.length_scale, self.noise_var)
        if (self.approx):
            loglik = 0.
            for i in np.arange(0, num_reps):
                loglik += gp.loglik_approx(self.x, np.reshape(self.y, (3*self.n, -1)), subsample=subsample)
                #if (best_loglik is None or loglik > best_loglik):
                #    best_loglik = loglik
            return loglik/num_reps
        else:
            return gp.loglik(self.x, np.reshape(self.y, (3*self.n, -1)))
        

    def reverse(self):
        num_positive = np.zeros(self.n)
        num_negative = np.zeros(self.n)
        gp = cov_div_free.cov_div_free(self.sig_var, self.length_scale, self.noise_var)
        for round_no in np.arange(0, 10):
            #i = self.get_random_indices()[0]
            #inds_train = np.array([i])
            inds_train = self.get_random_indices()
        
            x_train = self.x[inds_train]
            y_train = np.array(self.y[inds_train])
            #reverse(y_train, np.ones(len(y_train)), np.arange(0, len(y_train)))
        
            # Determine the points which lie in the vicinity of the point i
            
            #######################################################################
            # Determine the points wich lie in the vicinity of the point i
            #r = random.uniform()
            #x_diff = self.x - np.repeat(np.array([self.x[i]]), self.x.shape[0], axis=0)
            #x_diff = np.sum(x_diff**2, axis=1)
            #p = .5*(1+special.erf(np.sqrt(x_diff)/(self.length_scale*self.length_scale)))
            #i1 = np.where(p > 0.5)[0]
            #p[i1] = 1. - p[i1]
            #p *= 2.
            #inds1 = np.where(p >= r)[0]
            #######################################################################
            inds1 = np.random.choice(self.n, size=10000)
            inds1 = np.setdiff1d(inds1, inds_train)
            
            #print("inds_test", inds1)
            while len(inds1) > 0:
                inds_test = inds1[:min(1000, len(inds1))]
                inds1 = inds1[min(1000, len(inds1)):]
            
                x_test = self.x[inds_test]
                y_test_obs = self.y[inds_test]
            
                y_train_flat = np.reshape(y_train, (3*len(y_train), -1))
                gp.init(x_train, y_train_flat)
            
                y_test_mean = gp.fit(x_test, calc_var = False)
                y_test_mean = np.reshape(y_test_mean, y_test_obs.shape)
            
                sim = np.sum(y_test_obs[:,:2]*y_test_mean[:,:2], axis=1)
            
                pos_indices = np.where(sim >= 0.)[0]
                neg_indices = np.where(sim < 0.)[0]
                
                num_positive[inds_test[pos_indices]] += 1.0
                num_negative[inds_test[neg_indices]] += 1.0
        
        indices_to_reverse = np.where(num_negative > num_positive)
        print("indices_to_reverse", indices_to_reverse[0])
        
        if len(indices_to_reverse[0]) == 0:
            return False
        else:
            #print("num_positive, num_negative:", num_positive, num_negative)
            reverse(self.y, self.y_sign, indices_to_reverse)
            return True
                
        
    
    def align2(self, indices, length_scale):
        print("len(indices)", len(indices))
        
        #include_idx = set(indices)  #Set is more efficient, but doesn't reorder your elements if that is desireable
        #mask = np.array([(i in include_idx) for i in np.arange(0, len(x))])
        #mask = np.where(~mask)[0]
        affected_indices = set()
        gp = cov_div_free.cov_div_free(self.sig_var, length_scale, self.noise_var)
    
        inds_train = np.array(indices)
        
        # Determine the points which lie in the vicinity of the point i
        inds1 = np.random.choice(self.n, size=10000)
        inds1 = np.setdiff1d(inds1, inds_train)
        affected_indices.update(inds1)
        
        while len(inds1) > 0:
            inds_test = inds1[:min(1000, len(inds1))]
            inds1 = inds1[min(1000, len(inds1)):]
        
            x_train = self.x[inds_train]
            y_train = self.y[inds_train]
        
            x_test = self.x[inds_test]
            y_test_obs = self.y[inds_test]
        
            y_train_flat = np.reshape(y_train, (3*len(y_train), -1))
            print("x_train", x_train.shape)
            gp.init(x_train, y_train_flat)
        
            y_test_mean = gp.fit(x_test, calc_var = False)
            y_test_mean = np.reshape(y_test_mean, y_test_obs.shape)
        
            sim = np.sum(y_test_obs*y_test_mean, axis=1)
        
            sim_indices = np.where(sim < 0.)[0]
            reverse(self.y, self.y_sign, inds_test[sim_indices])
        
        return affected_indices
    
    
    def get_random_indices(self, length_scale=None):
        
        #random_indices = np.random.choice(n, size=int(n/2), replace=False)
        #num_indices = np.random.randint(low=1, high=min(max(2, int(1./(np.pi*length_scale**2))), 100))
        #num_indices = np.random.randint(low=1, high=min(max(2, int(100)), 10))
        num_indices = np.random.randint(low=1, high=min(max(2, int(100)), 4))

        assert(num_layers == 3)
        # Take support points from all three layers
        p1 = get_probs(self.thetas, self.y[::3])
        indices1 = np.random.choice(self.n//3, num_indices, replace=False, p=p1)*3
        p2 = get_probs(self.thetas, self.y[1::3])
        indices2 = np.random.choice(self.n//3, num_indices, replace=False, p=p2)*3
        p3 = get_probs(self.thetas, self.y[2::3])
        indices3 = np.random.choice(self.n//3, num_indices, replace=False, p=p3)*3

        #random_indices = np.random.choice(self.n, num_indices, replace=False, p=p)
        random_indices = np.concatenate((indices1, indices2, indices3))
        if length_scale is not None:
            i = 0
            while i < len(random_indices):
                random_index_filter = np.ones_like(random_indices, dtype=bool)
                ri = random_indices[i]
                for rj in np.arange(0, self.n):
                    x_diff = self.x[rj] - self.x[ri]
                    if (np.dot(x_diff, x_diff) < (3.*length_scale)**2):
                        for j in np.arange(i + 1, len(random_indices)):
                            if random_indices[j] == rj:
                                random_index_filter[j] = False
                random_indices = random_indices[random_index_filter]
                i += 1
        #print("random_indices", random_indices)
        return random_indices
    
    def algorithm_a(self):
        print(self.sig_var)
        loglik = None
        max_loglik = None
        
        iteration = -1
    
        #thetas = random.uniform(size=n)
        num_tries = 0
        
        temp = initial_temp
        changed = True
        while max_loglik is None or num_tries % max_num_tries != 0:# or (loglik < max_loglik):# or (loglik > max_loglik + eps):
            iteration += 1
            print("num_tries", num_tries)
        
            num_tries += 1
        
            temp = 1.#random.uniform(initial_temp, 2.)
            if inference and (iteration % inference_after_iter == 0):
                #if temp <= 1.0:
                #    temp += temp_delta*temp    
                
                length_scale, sig_var, noise_var = sample(self.x, np.reshape(self.y, (3*self.n, -1)))
                changed = True
                self.length_scale = length_scale
                self.sig_var = sig_var
                self.noise_var = noise_var
            #else:
                #if temp <= 1.0:
                #    temp += temp_delta*temp    
                
    
            if changed:
                loglik = self.loglik()
                changed = False
                
            print("sig_var=", self.sig_var)
            print("length_scale", self.length_scale)
            print("noise_var=", self.noise_var)
            #print("mean", mean)
            print("loglik=", loglik, "max_loglik=", max_loglik)
            
            if max_loglik is None or loglik > max_loglik:
                num_tries = 1
                max_loglik = loglik
            
            #y_last = np.array(self.y)
            #y_sign_last = np.array(self.y_sign)
            
            random_indices = self.get_random_indices(length_scale = temp*self.length_scale)
    
            
            #######################################################################
            # Do aligning of the chosen training vectors
            # based on most likely direction
            #r = random.uniform()
            #for ri in random_indices:
            #    if num_positive[ri] + num_negative[ri] < 10:
            #        theta = 0.5
            #    else:
            #        theta = float(num_positive[ri])/(num_positive[ri] + num_negative[ri])
            #    th = theta
            #    if th < 0.2:
            #        th = 0.2
            #    if th > 0.8:
            #        th = 0.8
            #    if r < th:
            #        if y_sign[ri] < 0:
            #            reverse(y, y_sign, ri)
            #            y_sign[ri] = 1 # overwrite sign
            #    else:
            #        if y_sign[ri] > 0:
            #            reverse(y, y_sign, ri)
            #            y_sign[ri] = -1 # overwrite sign
            
            
            #gp = cov_div_free.cov_div_free(sig_var, length_scale, noise_var)
            ##loglik1 = gp.loglik_approx(x, np.reshape(y, (3*n, -1)), subsample=subsample)
            #loglik1 = gp.loglik(x, np.reshape(y, (3*n, -1)))
            #print("loglik1=", loglik1, "max_loglik=", max_loglik)
    
            #if loglik1 > loglik:
            #    loglik = loglik1
            #    changed = True
                
            #    for ri in random_indices:
            #        if y_sign[ri] > 0:
            #            num_positive[ri] += 1.0
            #        else:
            #            num_negative[ri] += 1.0
            #        if num_positive[ri] + num_negative[ri] >= 10:
            #            theta = float(num_positive[ri])/(num_positive[ri] + num_negative[ri])
            #            thetas[ri] = np.log(theta)
            #        print("num_positive, num_negative:", num_positive[ri], num_negative[ri])
                
            y_last = np.array(self.y)
            y_sign_last = np.array(self.y_sign)
    
            start = time.time()
            affected_indices = self.align2(random_indices, temp*self.length_scale)
            #align(x, y, y_sign, random_indices, n, temp*length_scale, thetas)
            end = time.time()
            print("Align took: " + str(end - start))
    
            do_plots(self.y, "Guess")    
    
            start = time.time()
    
            loglik1 = self.loglik()
            end = time.time()
            print("Inference took: " + str(end - start))
    
            #do_plots(self.y)
    
            print("loglik1=", loglik1)
            print("loglik1=", loglik1, "max_loglik=", max_loglik)
    
            if loglik1 > loglik:
                loglik = loglik1
                changed = True
    
    
                for ri in affected_indices:
                    if self.y_sign[ri] > 0:
                        self.num_positive[ri] += 1.0
                    else:
                        self.num_negative[ri] += 1.0
                    if self.num_positive[ri] + self.num_negative[ri] >= 10:
                        theta = float(self.num_positive[ri])/(self.num_positive[ri] + self.num_negative[ri])
                        self.thetas[ri] = np.log(theta)
                    print("num_positive, num_negative:", self.num_positive[ri], self.num_negative[ri])
            else:
                self.y = y_last
                self.y_sign = y_sign_last
            #else:
            #    y = y_last
            #    y_sign = y_sign_last
                
    
            do_plots(self.y, "Current best")
    
    
        do_plots(self.y, "Result")
    
        exp_thetas = np.exp(self.thetas)
        bx_dis = self.y[:,0]
        by_dis = self.y[:,1]
        return exp_thetas, bx_dis, by_dis


    def algorithm_b(self):
        print(self.sig_var)
        loglik = None
        max_loglik = None
        
        iteration = -1
    
        #thetas = random.uniform(size=n)
        num_tries = 0
        
        changed = True
        while max_loglik is None or num_tries % max_num_tries != 0:# or (loglik < max_loglik):# or (loglik > max_loglik + eps):
            iteration += 1
            print("num_tries", num_tries)
        
            num_tries += 1
        
            if inference and (iteration % inference_after_iter == 0):
                #if temp <= 1.0:
                #    temp += temp_delta*temp    
                
                length_scale, sig_var, noise_var = sample(self.x, np.reshape(self.y, (3*self.n, -1)))
                changed = True
                self.length_scale = length_scale
                self.sig_var = sig_var
                self.noise_var = noise_var
            #else:
                #if temp <= 1.0:
                #    temp += temp_delta*temp    
                
    
            if changed:
                loglik = self.loglik()
                #loglik = gp.loglik(self.x, np.reshape(self.y, (3*self.n, -1)))
                changed = False
                
            print("sig_var=", self.sig_var)
            print("length_scale", self.length_scale)
            print("noise_var=", self.noise_var)
            #print("mean", mean)
            print("loglik=", loglik, "max_loglik=", max_loglik, "true_loglik=", self.true_loglik)
            
            if max_loglik is None or loglik > max_loglik:
                num_tries = 1
                max_loglik = loglik
            
            
            y_last = np.array(self.y)
            y_sign_last = np.array(self.y_sign)
            
            start = time.time()
            did_reverse = self.reverse()
            end = time.time()
            print("Reverse took: " + str(end - start))
    
            if did_reverse:
                do_plots(self.y, "Guess")
        
                start = time.time()
        
                loglik1 = self.loglik()
                end = time.time()
                print("Inference took: " + str(end - start))
        
                #do_plots(self.y)
        
                print("loglik1=", loglik1)
                print("loglik1=", loglik1, "max_loglik=", max_loglik)
        
                if loglik1 > loglik:
                    loglik = loglik1
                    changed = True
        
                    for ri in np.arange(0, n):
                        if self.y_sign[ri] > 0:
                            self.num_positive[ri] += 1.0
                        else:
                            self.num_negative[ri] += 1.0
                        if self.num_positive[ri] + self.num_negative[ri] >= 10:
                            theta = float(self.num_positive[ri])/(self.num_positive[ri] + self.num_negative[ri])
                            self.thetas[ri] = np.log(theta)
                        print("num_positive, num_negative:", self.num_positive[ri], self.num_negative[ri])
                else:
                    self.y = y_last
                    self.y_sign = y_sign_last
            else:
                self.y = y_last
                self.y_sign = y_sign_last
                 
            do_plots(self.y, "Current best")
    
    
        do_plots(self.y, "Result")
    
        exp_thetas = np.exp(self.thetas)
        bx_dis = self.y[:,0]
        by_dis = self.y[:,1]
        return exp_thetas, bx_dis, by_dis
    
sig_var = None
length_scale = None
noise_var = None
if not inference:

    sig_var=1.
    length_scale=.1*30/20
    noise_var=0.01
    

d = disambiguator(x, np.array(y), sig_var, length_scale, noise_var)
prob_a, field_x, field_y = d.algorithm_b()

misc.save("result_" + str(x_no) + "_" + str(y_no) + ".pkl", (n1, n2, n3, num_x, num_y, x_no, y_no, field_y))


