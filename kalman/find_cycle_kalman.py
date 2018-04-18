
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
import sys
import os
import os.path
import mw_utils


star = sys.argv[1]

data_dir = "./"
if data_dir == "../cleaned":
    skiprows = 1
else:
    skiprows = 0

files = []

data_found = False
for root, dirs, dir_files in os.walk(data_dir):
    for file in dir_files:
        if file[-4:] == ".dat":
            file_star = file[:-4]
            file_star = file_star.upper()
            if (file_star[-3:] == '.CL'):
                file_star = file_star[0:-3]
            if (file_star[0:2] == 'HD'):
                file_star = file_star[2:]
            while star[0] == '0': # remove leading zeros
                file_star = file_star[1:]
            if star == file_star:
                dat = np.loadtxt(data_dir+"/"+file, usecols=(0,1), skiprows=skiprows)
                data_found = True
                break

if not data_found:
    print "Cannot find data for " + star
    sys.exit(1)

offset = 1979.3452

t = dat[:,0]
y = dat[:,1]

t /= 365.25
t += offset


duration = t[-1] - t[0]
t -= duration/2

cov_types = ["quasiperiodic", "linear_trend"]

matern_p = 1

fig, (ax1) = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(6, 3)

orig_var = np.var(y)
noise_var = mw_utils.get_seasonal_noise_var(t, y, per_point=True, num_days=1.0)
noise_var = np.reshape(noise_var, (len(noise_var), 1, 1))

# just for removing the duplicates
t, y, noise_var = mw_utils.downsample(t, y, noise_var, min_time_diff=30.0/365.25, average=True)
ax1.plot(t, y, 'b+')

#print orig_var, np.min(noise_var), np.max(noise_var), np.mean(noise_var), np.std(noise_var)
noise_var = np.mean(noise_var)#np.max(noise_var)

slope_hat, intercept_hat, r_value, p_value, std_err = stats.linregress(t, y)
print "slope_hat, intercept_hat", slope_hat, intercept_hat

sig_vars = [orig_var, slope_hat**2]
freqs = np.linspace(0.0, 0.5, 100)
omegas = freqs*2.0*np.pi
ellqs = np.linspace(2.0, duration, int(duration))

def condition_fn(params):
    omega = params[1]
    ellq = params[3]
    if omega == 0:
        return False
    else:
        return 2.0*np.pi/omega <= ellq

kalman_utils = ku.kalman_utils(t, y, num_iterations=3, condition_fn=condition_fn)
for i in np.arange(0, len(cov_types)):
    cov_type = cov_types[i]
    #sig_var = np.array([sig_vars[i], 1.0])
    sig_var = np.linspace(sig_vars[i]*0.01, sig_vars[i]*100.0, 10)
    if cov_type == "linear_trend":
        slopes = np.linspace(slope_hat*0.5, slope_hat*1.5, 10)
        intercepts = np.linspace(intercept_hat*0.5, intercept_hat*1.5, 10)
        kalman_utils.add_component(cov_type, [slopes, intercepts])
    elif cov_type == "periodic":
        ells = np.array([10.0])
        kalman_utils.add_component(cov_type, [sig_var, omegas, ells])
    elif cov_type == "quasiperiodic":
        ellps = np.array([10.0])
        kalman_utils.add_component(cov_type, [sig_var, omegas, ellps, ellqs])
    elif cov_type == "exp_quad":
        kalman_utils.add_component(cov_type, [sig_var, ellqs])
    elif cov_type == "matern":
        kalman_utils.add_component(cov_type, [sig_var, ellqs])
    else:           
        assert(True==False)

kalman_utils.add_component("white_noise", [np.array([noise_var])])

param_modes, param_means, param_sigmas, y_means, logliks = kalman_utils.do_inference()

print "Estimated mode:", param_modes[:-1]
print "Estimated mean:", param_means[:-1]
print "Estimated sigma:", param_sigmas[:-1]
ax1.plot(t[1:], y_means, 'r--')

#y_means = kf_max.smooth()
#ax1.plot(t[1:-1], y_means, 'g--')

fig.savefig(star + '.png')
plt.close(fig)