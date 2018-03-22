
import numpy as np
import scipy
from scipy import stats
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colorbar as cb
from matplotlib import cm
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.ticker import LogFormatterMathtext, FormatStrFormatter
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import numpy.linalg as la
import scipy.special as special

import kalman

cov_type = "periodic"
#cov_type = "quasiperiodic"

def calc_cov_qp(t, f, length_scale, sig_var):
    k = np.zeros((len(t), len(t)))
    inv_l2 = 1.0/length_scale/length_scale
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            k[i, j] = sig_var*np.exp(-inv_l2*(t[i]-t[j])**2)*(np.cos(2 * np.pi*f*(t[i] - t[j])))
            k[j, i] = k[i, j]
    return k

def calc_cov_p(t, f, sig_var):
    k = np.zeros((len(t), len(t)))
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            k[i, j] = sig_var*(np.cos(2 * np.pi*f*(t[i] - t[j])))
            k[j, i] = k[i, j]
    return k


n = 50
time_range = 200
t = np.random.uniform(0.0, time_range, n)
t = np.sort(t)
var = 1.0
sig_var = np.random.uniform(0.99999, 0.99999)
noise_var = var - sig_var
mean = 0.5

#p = time_range/12.54321#
p = time_range/5#np.random.uniform(time_range/200, time_range/5)
freq = 1.0/p
mean = 0.0

if cov_type == "periodic":
    length_scale = 1e10*p
    k = calc_cov_p(t, freq, sig_var) + np.diag(np.ones(n) * noise_var)
else:
    length_scale = np.random.uniform(p/2.0, 4.0*p)
    k = calc_cov_qp(t, freq, length_scale, sig_var) + np.diag(np.ones(n) * noise_var)
    
l = la.cholesky(k)
s = np.random.normal(0, 1, n)

y = np.repeat(mean, n) + np.dot(l, s)
#y += mean

num_freqs = 100
num_cohs = 10

if cov_type == "periodic":
    num_cohs = 1
    
fig, (ax1) = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(6, 3)
ax1.plot(t, y, 'b+')

y_means_max = None
loglik_max = None
omega_max = None

for omega_0 in np.linspace(np.pi*freq, 4.0*np.pi*freq, 100):
    j_max = 2
    ell = 10
    
    ell_inv_sq = 1.0/ell/ell
    
    F = np.zeros((2*j_max, 2*j_max))
    L = np.zeros((2*j_max, 2*j_max))
    Q_c = np.zeros((2*j_max, 2*j_max)) # process noise
    H = np.zeros(2*j_max) # ovservatioanl matrix
    
    m_0 = np.zeros(2*j_max) # zeroth state mean
    P_0 = np.zeros((2*j_max, 2*j_max)) # zeroth state covariance
    
    R = 0.0 # observational noise
    
    for j in np.arange(0, j_max):
        F[2*j, 2*j+1] = -omega_0*j
        F[2*j+1, 2*j] = omega_0*j
    
        L[2*j, 2*j] = 1.0
        L[2*j+1, 2*j+1] = 1.0
    
        if j == 0:
            q = special.iv(0, ell_inv_sq)/np.exp(ell_inv_sq)
        else:
            q = 2.0 * special.iv(j, ell_inv_sq)/np.exp(ell_inv_sq)
    
        #print q
        P_0[2*j, 2*j] = q
        P_0[2*j+1, 2*j+1] = q
    
        H[2*j] = 1.0
    
    
    #F = 1.0
    #L = 1.0
    #H = 1.0
    #R = 0.0
    #m_0 = 0.0
    #P_0 = 1.0
    #Q_c = 1.0
    
    kf = kalman.kalman(t=t, y=y, F=F, L=L, H=H, R=R, m_0=m_0, P_0=P_0, Q_c=Q_c)
    y_means, loglik = kf.filter()
    print omega_0, loglik
    if loglik_max is None or loglik > loglik_max:
       loglik_max = loglik
       y_means_max = y_means
       omega_max = omega_0

print omega_max, freq*2.0*np.pi
ax1.plot(t[1:], y_means_max, 'r--')

#y_means = kalman.smooth()
#ax1.plot(t[1:], y_means, 'g--')

fig.savefig('test.png')
plt.close(fig)

