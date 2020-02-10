import sys
sys.path.append('../utils')
sys.path.append('..')

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colorbar as cb
from matplotlib import cm
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.ticker import LogFormatterMathtext, FormatStrFormatter
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import scipy.special as special
import numpy.linalg as la

import kalman_utils as ku
import os
import os.path
import MountWilson.mw_utils as mw_utils
import utils.plot as plot


dir_name = "data"
file_name = None
if len(sys.argv) > 1:
    index = sys.argv[1].find('/')
    if index >= 0:
        dir_name = sys.argv[1][:index]
        file_name = sys.argv[1][index+1:]
    else:
        dir_name = sys.argv[1]
        
skiprows = 1

cov_types = ["quasiperiodic"]#, "linear_trend"]
matern_p = 1

files = []

ds = []
data_found = False
for root, dirs, dir_files in os.walk(dir_name):
    for file in dir_files:
        if file_name is None or file_name == file:
            dat = np.loadtxt(dir_name+"/"+file, usecols=(0,1), skiprows=skiprows)
            star = file[:file.find('.')] 
            ds.append((star, dat))

for (star, dat) in ds:

    print("Estimating rotational period for star " + star)
    t = dat[:,0]
    y = dat[:,1]
    
    duration = t[-1] - t[0]
    t -= duration/2
    
    #fig, (ax1) = plt.subplots(nrows=1, ncols=1)
    #fig.set_size_inches(6, 3)
    
    orig_var = np.var(y)
    noise_var = mw_utils.get_seasonal_noise_var(t/365.25, y, per_point=True, num_years=1.0)
    means = mw_utils.get_seasonal_means_per_point(t/365.25, y)
    y -= means
    
    #print(means)
    #print(noise_var)
    
    #noise_var = np.reshape(noise_var, (len(noise_var), 1, 1))
    
    # just for removing the duplicates
    #t, y, noise_var = mw_utils.downsample(t, y, noise_var, min_time_diff=30.0/365.25, average=True)
    myplot = plot.plot(width=6, height=3)
    myplot.plot(t, y, params='b+')
    myplot.plot([min(t), max(t)], [np.mean(y), np.mean(y)], params='k:')
    myplot.plot([min(t), max(t)], np.mean(y)+[np.sqrt(orig_var), np.sqrt(orig_var)], params='k--')
    myplot.plot([min(t), max(t)], np.mean(y)-[np.sqrt(orig_var), np.sqrt(orig_var)], params='k--')
    myplot.fill(t, np.mean(y)-np.sqrt(noise_var), np.mean(y)+np.sqrt(noise_var))
    myplot.save(star + '.png')
    
    print(np.sqrt([orig_var, np.min(noise_var), np.max(noise_var), np.mean(noise_var), np.median(noise_var)]))
    noise_var = np.mean(noise_var)#np.max(noise_var)
    
    slope_hat, intercept_hat, r_value, p_value, std_err = stats.linregress(t, y)
    print("slope_hat, intercept_hat", slope_hat, intercept_hat)
    
    sig_vars = [orig_var]#, slope_hat**2]
    
    freqs = np.linspace(1./365, 2, 1000)
    omegas = freqs*2.0*np.pi
    ellqs = np.array([2])
    
    def condition_fn(params):
        omega = params[1]
        ellq = params[3]
        if omega == 0:
            return False
        else:
            return 2.0*np.pi/omega <= ellq
        
    def ellq_param_func(params):
        omega = params[1]
        ellq = params[3]
        return 2.0*np.pi/omega * ellq
        
    qp_param_funcs = [None, None, None, ellq_param_func]
        
    
    initial_indices = [None, None, None, len(ellqs)-1, None, None, None]
    kalman_utils = ku.kalman_utils(t, y, num_iterations=3, condition_fn=condition_fn, initial_indices = initial_indices)
    for i in np.arange(0, len(cov_types)):
        cov_type = cov_types[i]
        sig_var = np.array([sig_vars[i]])
        #sig_var = np.linspace(sig_vars[i]*0.01, sig_vars[i]*5.0, 10)
        if cov_type == "linear_trend":
            slopes = np.linspace(slope_hat*0.5, slope_hat*1.5, 10)
            intercepts = np.linspace(intercept_hat*0.5, intercept_hat*1.5, 10)
            kalman_utils.add_component(cov_type, [slopes, intercepts])
        elif cov_type == "periodic":
            ells = np.array([10.0])
            kalman_utils.add_component(cov_type, [sig_var, omegas, ells], {"j_max":2})
        elif cov_type == "quasiperiodic":
            ellps = np.array([10.0])
            kalman_utils.add_component(cov_type, [sig_var, omegas, ellps, ellqs], {"j_max":2}, param_funcs=qp_param_funcs)
        elif cov_type == "exp_quad":
            kalman_utils.add_component(cov_type, [sig_var, ellqs])
        elif cov_type == "matern":
            kalman_utils.add_component(cov_type, [sig_var, ellqs], {"p":matern_p})
        else:           
            assert(True==False)
    
    kalman_utils.add_component("white_noise", [np.array([noise_var])])
    
    param_modes, param_means, param_sigmas, y_means, logliks = kalman_utils.do_inference()
    
    print("Estimated mode:", param_modes[:-1])
    print("Estimated mean:", param_means[:-1])
    print("Estimated sigma:", param_sigmas[:-1])
    myplot.plot(t[1:], y_means, params='r--')
    
    kalman_utils.plot(star + "_")
    
    #y_means = kf_max.smooth()
    #ax1.plot(t[1:-1], y_means, 'g--')
    
    myplot.save(star + '.png')
    myplot.close()
