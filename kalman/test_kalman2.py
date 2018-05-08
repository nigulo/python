
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

cov_types = ["quasiperiodic"]#, "linear_trend"]
var = 200.0
sig_vars = [np.random.uniform(var*0.999, var*0.999)]#, np.random.uniform(var*0.0009, var*0.0009)]
noise_var = var - sum(sig_vars)
#cov_types = ["linear_trend"]
#cov_type = "periodic"
#cov_type = "quasiperiodic"
#cov_type = "exp_quad"
#cov_type = "matern"

matern_p = 1

n = 50
time_range = 200
t = np.random.uniform(0.0, time_range, n)
t = np.sort(t)
t -= np.mean(t)

#p = time_range/12.54321#
p = np.random.uniform(time_range/20, time_range/5)
freq = 1.0/p
mean = 0.0#np.random.uniform(-10.0, 10.0)

ellq = None
k = np.zeros((len(t), len(t)))
for i in np.arange(0, len(cov_types)):
    cov_type = cov_types[i]
    if cov_type == "linear_trend":
        k += ku.calc_cov_linear_trend(t, sig_vars[i])
    elif cov_type == "periodic":
        k += ku.calc_cov_p(t, freq, sig_vars[i])
    elif cov_type == "quasiperiodic":
        ellq = np.random.uniform(p, 4.0*p)
        k += ku.calc_cov_qp(t, freq, ellq, sig_vars[i])
    elif cov_type == "exp_quad":
        ellq = np.random.uniform(time_range/10, time_range/5)
        k += ku.calc_cov_exp_quad(t, ellq, sig_vars[i])
    elif cov_type == "matern":
        ellq = np.random.uniform(time_range/10, time_range/5)
        k += ku.calc_cov_matern(t, ellq, sig_vars[i], matern_p + 0.5)
    else:
        assert(True==False)

k += np.diag(np.ones(n) * noise_var)
l = la.cholesky(k)
s = np.random.normal(0.0, 1.0, n)

y = np.repeat(mean, n) + np.dot(l, s)
#y += mean

    
fig, (ax1) = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(6, 3)
ax1.plot(t, y, 'b+')

slope_hat, intercept_hat, r_value, p_value, std_err = stats.linregress(t, y)
print "slope_hat, intercept_hat", slope_hat, intercept_hat

kalman_utils = ku.kalman_utils(t, y, num_iterations=3)
for i in np.arange(0, len(cov_types)):
    cov_type = cov_types[i]
    #sig_var = np.array([sig_vars[i], 1.0])
    sig_var = np.linspace(sig_vars[i]/10, sig_vars[i]*5, 10)
    if cov_type == "linear_trend":
        slopes = np.linspace(slope_hat/2, slope_hat*2, 10)
        intercepts = np.linspace(mean/2, mean*2, 10)
        kalman_utils.add_component(cov_type, [slopes, intercepts])
    elif cov_type == "periodic":
        ells = np.array([10.0])
        omegas = np.linspace(2.0*np.pi*freq/3, 2.0*np.pi*freq*2, 10)
        kalman_utils.add_component(cov_type, [sig_var, omegas, ells], {"j_max":2})
    elif cov_type == "quasiperiodic":
        omegas = np.linspace(2.0*np.pi*freq/1.9, 2.0*np.pi*freq*2, 10) 
        ellps = np.array([1.0])
        ellqs = np.linspace(ellq/2, ellq*2, 10) 
        kalman_utils.add_component(cov_type, [sig_var, omegas, ellps, ellqs], {"j_max":2})
    elif cov_type == "exp_quad":
        ellqs = np.linspace(ellq/2, ellq*2, 20) 
        kalman_utils.add_component(cov_type, [sig_var, ellqs])
    elif cov_type == "matern":
        ellqs = np.linspace(ellq/2, ellq*2, 20) 
        kalman_utils.add_component(cov_type, [sig_var, ellqs], {"p":1})
    else:           
        assert(True==False)

kalman_utils.add_component("white_noise", [np.array([noise_var])])

param_modes, param_means, param_sigmas, y_means, logliks = kalman_utils.do_inference()


print "Estimated mode:", param_modes
print "Estimated mean:", param_means
print "Estimated sigma:", param_sigmas
print "True:", sig_vars[0], 2.0*np.pi*freq, 10.0, ellq, slope_hat, mean, noise_var
ax1.plot(t[1:], y_means, 'r--')

#y_means = kf_max.smooth()
#ax1.plot(t[1:-1], y_means, 'g--')

fig.savefig('test.png')
plt.close(fig)
