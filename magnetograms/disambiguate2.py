'''
    Disambiguation based on multiple layers of data
'''

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
import cov_sq_exp as cov_sq_exp
import scipy.misc
import numpy.random as random
import scipy.sparse.linalg as sparse
import scipy.stats as stats
import scipy.special as special
import utils
import plot
import misc
import pymc3 as pm
import kiss_gp
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
true_input = None # If False, then already processed input and no shuffling needed

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
if len(sys.argv) > 6:
    true_input = sys.argv[6]

mode = 2

subsample = 1000000 # For D2 approximation
num_subsample_reps = 1

num_layers = 3
inference = True
infer_z_scale = True
sample_or_optimize = False
num_samples = 1
num_chains = 4
inference_after_iter = 20

total_num_tries = 100
num_tries_without_progress = 100


def load(file_name):
    
    if file_name[-4:] == '.sav':
        print("Use disambiguate.py for single layer data")
        sys.exit(0)
        idl_dict = readsav(file_name)
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
    
    elif file_name[-5:] == '.fits':
        hdul = fits.open(file_name)
        if state_file == 'hinode_data/inverted_atmos.fits':
            #hdul = fits.open('pi-ambiguity-test/amb_spot.fits')
            # Actually x and y axis are swapped because of how the grid is defined later
            # Also swap y-axis to plot identically to XFITS Analyzer
            b = hdul[0].data[9:12, ::-1, :]
            theta = hdul[0].data[12:15, ::-1, :]*np.pi/180
            phi = hdul[0].data[15:18, ::-1, :]*np.pi/180
            
            b = np.transpose(b, (1, 2, 0))
            theta = np.transpose(theta, (1, 2, 0))
            phi = np.transpose(phi, (1, 2, 0))
            
        else:
            print("Use disambiguate.py for single layer data")
            sys.exit(0)
            dat = hdul[0].data[:,::4,::4]
            b = dat[0]
            theta = dat[1]
            phi = dat[2]
        hdul.close()
    elif file_name[-4:] == '.pkl':
        if os.path.isfile(file_name):
            y = pickle.load(open(file_name, 'rb'))
    #    if os.path.isfile('data3d50x50x10.pkl'):
    #        y = pickle.load(open('data3d50x50x10.pkl', 'rb'))
        else: 
            # Data not present, generate new
            n1, n2, n3 = 100, 100, 3
            n = n1 * n2 * n3
            x1_range, x2_range, x3_range = 1., 1., .03
            
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
            length_scale_train = .03
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
            
            with open(file_name, 'wb') as f:
                pickle.dump(y, f)    
        
        ###########################################################################
        # Plotting the whole qube
        x1_range, x2_range = 1., 1.
        x1 = np.linspace(0, x1_range, y.shape[0])
        x2 = np.linspace(0, x2_range, y.shape[1])
        x_mesh = np.meshgrid(x2, x1)
    
        if true_input is None:
            for i in np.arange(0, y.shape[2]):
                test_plot = plot.plot(nrows=1, ncols=1)
                test_plot.set_color_map('bwr')
                
                test_plot.colormap(y[:, :, i, 2])
                test_plot.vectors(x_mesh[0], x_mesh[1], y[:, :, i, 0], y[:, :, i, 1], [], units='width', color = 'k')
                test_plot.save("test_field" + str(i) +".png")
                test_plot.close()
        ###########################################################################
    
        bx = y[:, :, :, 0]
        by = y[:, :, :, 1]
        bz = y[:, :, :, 2]
        
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
    
        if true_input is None or file_name == true_input:
    
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
    return b, phi, theta

def convert(b, phi, theta):
    
    ###########################################################################
    # Cut out the patch as specified by program arguments
    
    n1 = b.shape[0]//num_x
    n2 = b.shape[1]//num_y
    x_start = n1*x_no
    x_end = min(x_start + n1, b.shape[0])
    y_start = n2*y_no
    y_end = min(y_start + n1, b.shape[1])
    
    b = b[x_start:x_end, y_start:y_end, :num_layers]
    phi = phi[x_start:x_end, y_start:y_end, :num_layers]
    theta = theta[x_start:x_end, y_start:y_end, :num_layers]
    ###########################################################################
    
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
    print("x1_mesh", x1_mesh[:,0,0])
    print("x2_mesh", x2_mesh[0,:,0])
    print("x3_mesh", x3_mesh[0,0,:])
    x_grid = np.stack((x1_mesh, x2_mesh, x3_mesh), axis=3)
    x = x_grid.reshape(-1, 3)
    x_flat = np.reshape(x, (3*n, -1))

    if mode == 2:
        
        truth_plot = plot.plot(nrows=n3, ncols=4)
        truth_plot.set_color_map('bwr')
        for layer in np.arange(0, n3):
            truth_plot.colormap(bx[:, :, layer], ax_index = [layer, 0])
            truth_plot.colormap(by[:, :, layer], ax_index = [layer, 1])
            truth_plot.colormap(bz[:, :, layer], ax_index = [layer, 2])
            truth_plot.colormap(np.sqrt(bx[:, :, layer]**2 + by[:, :, layer]**2 + bz[:, :, layer]**2), ax_index = [layer, 3])
            #truth_plot.vectors(x1_mesh, x2_mesh, bx[:, :, layer], by[:, :, layer], ax_index = [layer], units='width', color = 'k')
        truth_plot.save("truth_field_" + str(x_no) + "_" + str(y_no) + ".png")
        truth_plot.close()

    bx = np.reshape(bx, n)
    by = np.reshape(by, n)
    bz = np.reshape(bz, n)
    
    y = np.column_stack((bx, by, bz))
    
    return x, y, n1, n2, n3
    ###########################################################################

b, phi, theta = load(state_file)

n1_orig = b.shape[0]
n2_orig = b.shape[1]
n3_orig = num_layers

x, y, n1, n2, n3 = convert(b, phi, theta)
n = n1*n2*n3

y_orig = np.array(y)

m1 = max(10, n1//10)
m2 = max(10, n2//10)
m3 = n3
m = m1 * m2 * m3
u1_range = np.max(x[:,0])
u2_range = np.max(x[:,1])
u3_range = np.max(x[:,2])

u1 = np.linspace(0, u1_range, m1)
u2 = np.linspace(0, u2_range, m2)
u3 = np.linspace(0, u3_range, m3)
u_mesh = np.meshgrid(u1, u2, u3, indexing='ij')


if true_input is None:
    y_true = y_orig
else:
    assert(os.path.isfile(true_input))
    b_true, phi_true, theta_true = load(true_input)
    _, y_true, _, _, _ = convert(b_true, phi_true, theta_true)
    


def do_plots(y, thetas, title = None, file_name=None):
    bx_true = y_true[:, 0]    
    by_true = y_true[:, 1]
    bx_true = np.reshape(bx_true, (n1, n2, n3))
    by_true = np.reshape(by_true, (n1, n2, n3))
    
    if thetas is not None:
        thetas = np.exp(thetas)
        ids = np.where(thetas > 0.5)
        thetas[ids] = 1. - thetas[ids]
        thetas = np.reshape(thetas*2., (n1, n2, n3))

    if y is not None:
        bx_dis = y[:, 0]
        by_dis = y[:, 1]
        bx_dis = np.reshape(bx_dis, (n1, n2, n3))
        by_dis = np.reshape(by_dis, (n1, n2, n3))

    for layer in np.arange(0, n3):
        components_plot = plot.plot(nrows=3, ncols=3, title = title)
        components_plot.set_color_map('bwr')
        
        components_plot.colormap(bx_true[:, :, layer], [0, 0])
        components_plot.colormap(by_true[:, :, layer], [0, 1])
        components_plot.colormap(np.reshape(np.arctan2(by_true[:, :, layer], bx_true[:, :, layer]), (n1, n2)), [0, 2])
        
        
        if y is not None:    
            components_plot.colormap(bx_dis[:, :, layer], [1, 0])
            components_plot.colormap(by_dis[:, :, layer], [1, 1])
            components_plot.colormap(np.reshape(np.arctan2(by_dis[:, :, layer], bx_dis[:, :, layer]), (n1, n2)), [1, 2])

            components_plot.set_color_map('Greys')
            
            bx_diff = bx_true[:, :, layer] - bx_dis[:, :, layer]
            by_diff = by_true[:, :, layer] - by_dis[:, :, layer]
            components_plot.colormap(np.array(bx_diff == 0., dtype='float'), [2, 0])
            components_plot.colormap(np.array(by_diff == 0., dtype='float'), [2, 1])

        if thetas is not None:
            components_plot.set_color_map('Greys')
            components_plot.colormap(thetas[:, :, layer], [2, 2], vmin=0., vmax=1.)
        
        if file_name is None:
            file_name = "components"
        components_plot.save(file_name + "_" + str(x_no) + "_" + str(y_no) + "_" + str(layer) +".png")
        components_plot.close()


do_plots(None, None)

# Using KISS-GP
# https://arxiv.org/pdf/1503.01057.pdf

# define the parameters with their associated priors

data_var = (np.var(y[:,0]) + np.var(y[:,1]))
#noise_var = .01*data_var
#forced_noise_var = data_var - (np.var(np.abs(y[:,0])) + np.var(np.abs(y[:,1])))/2
#data_var -= forced_noise_var
print("data_var:", data_var)

grid_density = (np.max(x[:,0])-np.min(x[:,0]))/n1 +(np.max(x[:,1])-np.min(x[:,1]))/n2 + (np.max(x[:,2])-np.min(x[:,2]))/n3
grid_density /= 3
max_grid_size = max((np.max(x[:,0])-np.min(x[:,0])), (np.max(x[:,1])-np.min(x[:,1])) + (np.max(x[:,2])-np.min(x[:,2])))
print("grid_density:", grid_density)
print("max_grid_size:", max_grid_size)


def sample(x, y, infer_z_scale=False, known_params=dict()):

    
    def decode_params(theta):
        i = 0
        if "sig_var" in known_params:
            sig_var = known_params["sig_var"]
        else:
            sig_var = theta[i]
            i += 1
        if "ell" in known_params:
            ell = known_params["ell"]
        else:
            ell = theta[i]
            i += 1
        if "noise_var" in known_params:
            noise_var = known_params["noise_var"]
        else:
            noise_var = theta[i]
            i += 1
            
        if "z_scale" in known_params:
            z_scale = known_params["z_scale"]
        else:
            z_scale = None
            if len(theta) > i:
                z_scale = theta[i]
                i += 1
                
        return sig_var, ell, noise_var, z_scale
        
    
    def lik_fn_div_free(theta):
        sig_var = theta[0]
        ell = theta[1]
        noise_var = theta[2]
        gp = cov_div_free.cov_div_free(sig_var, ell, noise_var)
        return gp.calc_loglik_approx(x, y, subsample=subsample)

    def lik_fn_sq_exp(theta):
        sig_var = theta[0]
        ell = theta[1]
        noise_var = theta[2]
        gp = cov_sq_exp.cov_sq_exp(sig_var, ell, noise_var)
        return gp.calc_loglik_approx(x, y, subsample=subsample)

    def cov_func_div_free(theta, data, u1, u2, data_or_test):
        np.testing.assert_array_almost_equal(u1, u2)
        assert(data_or_test)
        sig_var = theta[0]
        ell = theta[1]
        noise_var = theta[2]
        gp = cov_div_free.cov_div_free(sig_var, ell, noise_var)
        U, U_grads = gp.calc_cov(u1, u2, data_or_test=data_or_test, calc_grad = True)
        return  U, U_grads

    def cov_func_sq_exp(theta, data, u1, u2, data_or_test):
        np.testing.assert_array_almost_equal(u1, u2)
        assert(data_or_test)
            
        sig_var, ell, noise_var, z_scale = decode_params(theta)
        if z_scale is None:
            gp = cov_sq_exp.cov_sq_exp(sig_var, ell, noise_var, dim_out=1)
        else:
            gp = cov_sq_exp.cov_sq_exp(sig_var, ell, noise_var, dim_out=1, scales=[None, None, z_scale])
        U, U_grads = gp.calc_cov(u1, u2, data_or_test=data_or_test, calc_grad = True)
        return  U, U_grads

    if len(y.shape) > 1 and y.shape[1] == 3:
        #lik_fn = lik_fn_div_free
        cov_func = cov_func_div_free
        kgp = kiss_gp.kiss_gp(x, u_mesh, cov_func, y, indexing_type=False)
    else:
        #lik_fn = lik_fn_sq_exp
        cov_func = cov_func_sq_exp
        kgp = kiss_gp.kiss_gp(x, u_mesh, cov_func, y, dim=1, indexing_type=False)


    lik_grad = None

    sampling_result = dict()
    if sample_or_optimize:
        s = sampling.Sampling()
        with s.get_model():
            params = []
            if "sig_var" not in known_params:
                sig_var = pm.HalfNormal('sig_var', sd=1.0)
                params.append(sig_var)
            if "ell" not in known_params:
                ell = pm.HalfNormal('ell', sd=1.0)
                params.append(ell)
            if "noise_var" not in known_params:
                noise_var = pm.HalfNormal('noise_var', sd=1.0)
                params.append(noise_var)
            
            if infer_z_scale:
                if "z_scale" not in known_params:
                    z_scale = pm.Gamma('z_scale', mu = 1.0, sigma = 10.)
                    params.append(z_scale)
        
        #trace = s.sample(lik_fn, [sig_var, ell, noise_var], [], num_samples, num_chains, lik_grad)
        trace = s.sample(kgp.likelihood, params, [], num_samples, num_chains, kgp.likelihood_grad)

        #print(trace['model_logp'])
        if "ell" in trace:
            sampling_result["ell"] = np.mean(trace['ell'])
        if "sig_var" in trace:
            sampling_result["sig_var"] = np.mean(trace['sig_var'])
        if "noise_var" in trace:
            sampling_result["noise_var"] = np.mean(trace['noise_var'])
        if "z_scale" in trace:
            sampling_result["z_scale"] = np.mean(trace['z_scale'])

    else:
        def lik_fn(params):
            return -kgp.likelihood(params, [])

        def grad_fn(params):
            return -kgp.likelihood_grad(params, [])

        min_loglik = None
        min_res = None
        for trial_no in np.arange(0, num_samples):
            params = []
            bounds = []
            if "sig_var" not in known_params:
                sig_var_min = data_var*.1
                sig_var_max = data_var*2.
                sig_var_init = random.uniform(sig_var_min, sig_var_max)
                params.append(sig_var_init)
                bounds.append((sig_var_min, sig_var_max))
            if "ell" not in known_params:
                ell_min = grid_density#.05
                ell_max = max_grid_size*2#1.
                ell_init = random.uniform(ell_min, ell_max)
                params.append(ell_init)
                bounds.append((ell_min, ell_max))
            if "noise_var" not in known_params:
                noise_var_min = data_var*0.0001#*.001
                noise_var_max = data_var*0.01#*.5
                noise_var_init = random.uniform(noise_var_min, noise_var_max)
                params.append(noise_var_init)
                bounds.append((noise_var_min, noise_var_max))
            
            if infer_z_scale:
                if "z_scale" not in known_params:
                    z_scale_min = 0.1
                    z_scale_max = 10
                    z_scale_init = 1.#random.uniform(z_scale_min, z_scale_max)
                    params.append(z_scale_init)
                    bounds.append((z_scale_min, z_scale_max))
            

            #res = scipy.optimize.minimize(lik_fn, [.5, data_var, data_var*.015], method='L-BFGS-B', jac=grad_fn, bounds = [(.1, 1.), (data_var*.1, data_var*2.), (data_var*.01, data_var*.02)], options={'disp': True, 'gtol':1e-7})
            res = scipy.optimize.minimize(lik_fn, params, method='L-BFGS-B', jac=lik_grad, bounds=bounds, options={'disp': True, 'gtol':1e-7})
            loglik = res['fun']
            #assert(loglik == lik_fn(res['x']))
            if min_loglik is None or loglik < min_loglik:
                min_loglik = loglik
                min_res = res

        i = 0
        if "sig_var" not in known_params:
            sampling_result["sig_var"] = min_res['x'][i]
            i += 1
        if "ell" not in known_params:
            sampling_result["ell"] = min_res['x'][i]
            i += 1
        if "noise_var" not in known_params:
            sampling_result["noise_var"] = min_res['x'][i]
            i += 1
        
        if infer_z_scale:
            if "z_scale" not in known_params:
                sampling_result["z_scale"] = min_res['x'][i]
                i += 1
    return sampling_result
        



def reverse(y, y_sign, ii):
    #y_old = np.array(y)
    y[ii,:2] *= -1
    y_sign[ii] *= -1
    
    #np.testing.assert_almost_equal(np.sum(y_old[ii]**2), np.sum(y[ii]**2))
    #np.testing.assert_almost_equal(y_old[ii,:2] + y[ii,:2], np.zeros_like(y_old[ii,:2]))


def get_probs(thetas, y):
    b = np.sqrt(np.sum(y[:,:2]*y[:,:2], axis=1))
    return b/np.sum(b)
    

class disambiguator():
    
    def __init__(self, x, y, sig_var=None, length_scale=None, noise_var=None, approx_type='kiss-gp', u_mesh=None):
        self.x = x
        self.y = y
        self.n = len(x)
        assert(self.n == n1*n2*n3)
        
        self.y_sign = np.ones(n)
        thetas = np.ones(self.n)/2.
        self.thetas = np.log(thetas)

        self.num_positive = np.zeros(self.n)
        self.num_negative = np.zeros(self.n)
        
        self.approx_type = approx_type
        
        if approx_type == 'kiss-gp':
            assert(u_mesh is not None)
            self.u_mesh = u_mesh
            self.u = np.stack(u_mesh, axis=3).reshape(-1, 3)
        
        if sig_var is not None and length_scale is not None and noise_var is not None:
            self.sig_var = sig_var
            self.length_scale = length_scale
            self.noise_var = noise_var
            self.init()
        
    def init(self):    
        gp = cov_div_free.cov_div_free(self.sig_var, self.length_scale, self.noise_var)
        if self.approx_type == 'd2':
            self.true_loglik = 0.
            for i in np.arange(0, num_subsample_reps):
                self.true_loglik += gp.calc_loglik_approx(self.x, np.reshape(y_orig, (3*self.n, -1)), subsample=subsample)
                #if (self.true_loglik is None or true_loglik > self.true_loglik):
                #    self.true_loglik = true_loglik
            self.true_loglik /= num_subsample_reps
        
        elif self.approx_type == 'kiss-gp':
            U = gp.calc_cov(self.u, self.u, data_or_test=True)
            W = utils.calc_W(self.u_mesh, self.x, us=self.u, indexing_type=False)#np.zeros((len(x1)*len(x2)*2, len(u1)*len(u2)*2))
            (x, istop, itn, normr) = sparse.lsqr(W, np.reshape(y_orig, (3*self.n, -1)))[:4]#, x0=None, tol=1e-05, maxiter=None, M=None, callback=None)
            L = la.cholesky(U)
            v = la.solve(L, x)
            self.true_loglik = -0.5 * np.dot(v.T, v) - sum(np.log(np.diag(L))) - 0.5 * self.n * np.log(2.0 * np.pi)
        else:
            self.true_loglik = gp.calc_loglik(self.x, np.reshape(y_orig, (3*self.n, -1)))

    def loglik(self):
        gp = cov_div_free.cov_div_free(self.sig_var, self.length_scale, self.noise_var)
        if (self.approx_type == 'd2'):
            loglik = 0.
            for i in np.arange(0, num_subsample_reps):
                loglik += gp.calc_loglik_approx(self.x, np.reshape(self.y, (3*self.n, -1)), subsample=subsample)
                #if (best_loglik is None or loglik > best_loglik):
                #    best_loglik = loglik
            return loglik/num_subsample_reps
        elif self.approx_type == 'kiss-gp':
            U = gp.calc_cov(self.u, self.u, data_or_test=True)
            W = utils.calc_W(self.u_mesh, self.x, us=self.u, indexing_type=False)#np.zeros((len(x1)*len(x2)*2, len(u1)*len(u2)*2))
            (x, istop, itn, normr) = sparse.lsqr(W, np.reshape(self.y, (3*self.n, -1)))[:4]#, x0=None, tol=1e-05, maxiter=None, M=None, callback=None)
            L = la.cholesky(U)
            v = la.solve(L, x)
            return -0.5 * np.dot(v.T, v) - sum(np.log(np.diag(L))) - 0.5 * self.n * np.log(2.0 * np.pi)
        else:
            return gp.calc_loglik(self.x, np.reshape(self.y, (3*self.n, -1)))
        
    def estimate_params(self):
        #print("y.shape", self.y.shape)
        #y_flat = np.reshape(self.y, (3*self.n, -1))
        #print("y_flat.shape", y_flat.shape)
        b_abs = np.sqrt(np.sum(self.y*self.y, axis=1))
        #b_sq = np.sum(y_flat*y_flat, axis=1)
        
        if infer_z_scale:
            ###################################################################
            # First infer the length-scale, sig_var and noise_var from different layers separately
            avg_length_scale = 0.
            avg_sig_var = 0.
            avg_noise_var = 0.
            for i in np.arange(n3):
                x1 = self.x[i::n3, :2]
                b_abs1 = b_abs[i::n3]
                res = sample(x1, b_abs1)
                length_scale = res["ell"]
                sig_var = res["sig_var"]
                noise_var = res["noise_var"]
                print("Layer ", i, ": length_scale, sig_var, noise_var", length_scale, sig_var, noise_var)
                avg_length_scale += length_scale
                avg_sig_var += sig_var
                avg_noise_var += noise_var
            avg_length_scale /= n3
            avg_sig_var /= n3
            avg_noise_var /= n3
            
            ###################################################################
            # Now infer the z_scale
            known_params={"ell": avg_length_scale, "sig_var": avg_sig_var, "noise_var": avg_noise_var}
            res = sample(self.x, b_abs, infer_z_scale=infer_z_scale, known_params=known_params)
            z_scale = res["z_scale"]
            print("z_scale", z_scale)
            self.x[:, 2] *= z_scale

        ###################################################################
        # Infer the length-scale, sig_var and noise_var from full data again
        res = sample(self.x, b_abs)
        self.length_scale = res["ell"]
        self.sig_var = res["sig_var"]
        self.noise_var = res["noise_var"]
            
        self.init()


    def reverse(self, try_no):
        num_positive = np.ones(self.n)
        num_negative = np.ones(self.n)
        #probs = np.zeros(self.n)
        gp = cov_div_free.cov_div_free(self.sig_var, self.length_scale, self.noise_var)
        in_or_out = np.random.choice([True, False])
        num_indices = np.random.randint(low=1, high=12)
        num_train_sets = 10
        #inds_train_all = self.get_random_indices(try_no, num_indices = num_indices*num_train_sets)#, length_scale=self.length_scale, in_or_out = in_or_out)

        r1 = random.uniform()
        for round_no in np.arange(0, num_train_sets):#max(1, min(10, try_no//4))):
            #i = self.get_random_indices()[0]
            #inds_train = np.array([i])
            inds_train = self.get_random_indices(try_no, num_indices = num_indices)#, length_scale=self.length_scale, in_or_out = in_or_out)
            #inds_train = inds_train_all[round_no*num_indices:(round_no+1)*num_indices]
        
            x_train = self.x[inds_train]
            y_train = np.array(self.y[inds_train])
            
            y_train_copy = np.array(self.y[inds_train])
            y_sign_train_copy = np.array(self.y_sign[inds_train])
            if r1 > min(0.9, 0.5+float(try_no)/num_tries_without_progress):
                reverse(self.y, self.y_sign, inds_train)
        
            # Determine the points which lie in the vicinity of the point i            
            #inds1 = np.random.choice(self.n, size=10000, replace=False)
            inds1 = np.arange(self.n)
            inds1 = np.setdiff1d(inds1, inds_train)
            #######################################################################
            # Determine the points wich lie in the vicinity of at least one training point
            close_mask = np.zeros(len(inds1), dtype='bool')
            x_test1 = self.x[inds1]
            ell = self.length_scale#max(self.length_scale, 3*self.length_scale*try_no/num_tries_without_progress)
            for x_t in self.x[inds_train]:
                r = random.uniform()
                #print("x_t", x_t)
                x_diff = x_test1 - np.repeat(np.array([x_t]), x_test1.shape[0], axis=0)
                x_diff = np.sum(x_diff**2, axis=1)
                
                p = .5*(1+special.erf(np.sqrt(.5*x_diff)/ell))
                #print(p[1:10], self.length_scale)
                i1 = np.where(p > 0.5)[0]
                p[i1] = 1. - p[i1]
                p *= 2.
                close_inds = np.where(p >= r)[0]
                close_mask[close_inds] = True
                #print("close_inds", r, p, np.where(close_inds))
            #print("close_mask", np.where(close_mask))
            inds1 = inds1[close_mask]
            #######################################################################
        
            
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
                #sim_norm = sim / np.sqrt(np.sum(y_test_obs[:,:2]*y_test_obs[:,:2], axis=1)*np.sum(y_test_mean[:,:2]*y_test_mean[:,:2], axis=1))
            
                #probs[inds_test] = (sim_norm + 1.) / 2.
                #assert(np.all(probs >= 0) and np.all(probs <= 1))
                pos_indices = np.where(sim >= 0.)[0]
                neg_indices = np.where(sim < 0.)[0]
                
                num_positive[inds_test[pos_indices]] += 1.0
                num_negative[inds_test[neg_indices]] += 1.0
        
            self.y[inds_train] = y_train_copy
            self.y_sign[inds_train] = y_sign_train_copy
            
        
        reverse_probs = num_negative / (num_negative + num_positive)
        reverse_probs[reverse_probs < .6] = 0.
        indices_to_reverse = np.where(random.uniform(size = len(reverse_probs)) < reverse_probs)
        #indices_to_reverse = np.where(random.uniform(size = len(probs)) > probs)
        
        #indices_to_reverse = np.where(num_negative > num_positive)
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
    
    
    def get_random_indices(self, try_no, num_indices = None, length_scale=None, in_or_out=False):
        
        if num_indices is None:
            #random_indices = np.random.choice(n, size=int(n/2), replace=False)
            #num_indices = np.random.randint(low=1, high=min(max(2, int(1./(np.pi*length_scale**2))), 100))
            #num_indices = np.random.randint(low=1, high=min(max(2, int(100)), 10))
            num_indices = np.random.randint(low=1, high=12)#min(10, max(2, try_no//2)))
            #num_indices = 9#min(6, max(2, try_no//2))

        assert(num_layers == 3)
        # Take support points from all three layers
        p1 = get_probs(self.thetas, self.y[::3])
        indices1 = np.random.choice(self.n//3, num_indices, replace=False, p=p1)*3
        p2 = get_probs(self.thetas, self.y[1::3])
        indices2 = np.random.choice(self.n//3, num_indices, replace=False, p=p2)*3 + 1
        p3 = get_probs(self.thetas, self.y[2::3])
        indices3 = np.random.choice(self.n//3, num_indices, replace=False, p=p3)*3 + 2

        #random_indices = np.random.choice(self.n, num_indices, replace=False, p=p)
        #random_indices = np.concatenate((indices1, indices2, indices3))
        random_indices = np.array([indices1, indices2, indices3]).T.flatten()
        if length_scale is not None:
            i = 0
            while i < len(random_indices):
                random_index_filter = np.ones_like(random_indices, dtype=bool)
                ri = random_indices[i]
                for rj in np.arange(0, self.n):
                    x_diff = self.x[rj] - self.x[ri]
                    dist = np.dot(x_diff, x_diff)
                    if ((in_or_out and dist > (3.*length_scale)**2) or (not in_or_out and dist < (length_scale)**2)):
                        for j in np.arange(i + 1, len(random_indices)):
                            if random_indices[j] == rj:
                                random_index_filter[j] = False
                random_indices = random_indices[random_index_filter]
                i += 1
        #print("random_indices", random_indices)
        return random_indices
    

    def disambiguate(self):#, best_loglik = sys.float_info.min):
        print(self.sig_var)
        loglik = None
        max_loglik = None
        
        iteration = -1
    
        #thetas = random.uniform(size=n)
        num_tries = 0
        tot_num_tries = 0
        
        #changed = True
        while max_loglik is None or num_tries % num_tries_without_progress != 0:# or (loglik < max_loglik):# or (loglik > max_loglik + eps):
            iteration += 1
            print("num_tries", num_tries)
        
            num_tries += 1
            tot_num_tries += 1 # This counter is not reset
        
            #if inference and (iteration % inference_after_iter == 0):
            #    #if temp <= 1.0:
            #    #    temp += temp_delta*temp    
            #    
            #    length_scale, sig_var, noise_var = sample(self.x, np.reshape(self.y, (3*self.n, -1)))
            #    changed = True
            #    self.length_scale = length_scale
            #    self.sig_var = sig_var
            #    self.noise_var = noise_var
            ##else:
            #    #if temp <= 1.0:
            #    #    temp += temp_delta*temp    
                
    
            #if changed:
            #    loglik = self.loglik()
            #    #loglik = gp.loglik(self.x, np.reshape(self.y, (3*self.n, -1)))
            #    changed = False
                
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
            did_reverse = self.reverse(tot_num_tries)
            end = time.time()
            print("Reverse took: " + str(end - start))
    
            if did_reverse:
                do_plots(self.y, self.thetas, "Guess")
        
                start = time.time()
        
                loglik1 = self.loglik()
                end = time.time()
                print("Inference took: " + str(end - start))
        
                #do_plots(self.y)
        
                print("loglik1=", loglik1)
                print("loglik1=", loglik1, "max_loglik=", max_loglik)
        
                if loglik is None or loglik1 > loglik:
                    loglik = loglik1
                    changed = True
        
                    for ri in np.arange(0, n):
                        print("num_positive, num_negative:", self.num_positive[ri], self.num_negative[ri])
                else:
                    self.y = y_last
                    self.y_sign = y_sign_last
            else:
                self.y = y_last
                self.y_sign = y_sign_last

            for ri in np.arange(0, n):
                if self.y_sign[ri] > 0:
                    self.num_positive[ri] += 1.0
                else:
                    self.num_negative[ri] += 1.0
                if self.num_positive[ri] + self.num_negative[ri] >= 10:
                    theta = float(self.num_positive[ri])/(self.num_positive[ri] + self.num_negative[ri])
                    self.thetas[ri] = np.log(theta)
                 
            do_plots(self.y, self.thetas, "Current best")
    
    
        do_plots(self.y, self.thetas, "Result")
    
        exp_thetas = np.exp(self.thetas)
        #bx_dis = self.y[:,0]
        #by_dis = self.y[:,1]
        return exp_thetas, self.y, loglik
    
sig_var = None
length_scale = None
noise_var = None
if not inference:

    sig_var=1.
    #length_scale=.03*n1_orig/n1
    length_scale=.1*n1_orig/n1
    noise_var=0.01
    
best_loglik = None

for i in np.arange(0, total_num_tries):
    
    y = np.array(y_orig)
    if true_input is None:
        # Align all the transverse components either randomly or identically
        for i in np.arange(0, n):
            if np.random.uniform() < 0.5:
                y[i, :2] *= -1
            #y[i, :2] = np.abs(y[i, :2])
    
    print("True length_scale", length_scale)
    d = disambiguator(x, y, sig_var, length_scale, noise_var, approx_type='kiss-gp', u_mesh=u_mesh)
    if inference:
        d.estimate_params()
    print("Estimated length_scale", d.length_scale)
    prob_a, field_y, loglik = d.disambiguate()
    if best_loglik is None or loglik > best_loglik:
        best_loglik = loglik
        best_y = field_y
        do_plots(best_y, d.thetas, title="Current best", file_name="best_result")

        misc.save("result_" + str(x_no) + "_" + str(y_no) + ".pkl", (n1_orig, n2_orig, n3_orig, num_x, num_y, x_no, y_no, best_y))


