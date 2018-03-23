
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

#cov_type = "periodic"
cov_type = "quasiperiodic"

def calc_cov_qp(t, f, length_scale, sig_var):
    k = np.zeros((len(t), len(t)))
    inv_l2 = 1.0/length_scale/length_scale
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            #k[i, j] = sig_var*np.exp(-inv_l2*(t[i]-t[j])**2)*(np.cos(2 * np.pi*f*(t[i] - t[j])))
            k[i, j] = sig_var*np.exp(-np.abs(t[i]-t[j])/length_scale)*(np.cos(2 * np.pi*f*(t[i] - t[j])))
            k[j, i] = k[i, j]
    return k

def calc_cov_p(t, f, sig_var):
    k = np.zeros((len(t), len(t)))
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            k[i, j] = sig_var*(np.cos(2 * np.pi*f*(t[i] - t[j])))
            k[j, i] = k[i, j]
    return k


def get_params_p_j(j, omega_0, ell):
    ell_inv_sq = 1.0/ell/ell
    I2 = np.diag(np.ones(2))

    Fpj = np.array([[0.0, -omega_0*j], [omega_0*j, 0.0]])

    Lpj = I2

    if j == 0:
        qj = special.iv(0, ell_inv_sq)/np.exp(ell_inv_sq)
    else:
        qj = 2.0 * special.iv(j, ell_inv_sq)/np.exp(ell_inv_sq)

    P0pj = I2 * qj
    #Qcpj = np.zeros((2, 2))
    Hpj = np.array([1.0, 0.0])
    
    return Fpj, Lpj, P0pj, Hpj, qj


def get_params_p(j_max, omega_0, ell, noise_var):
    
    F = np.zeros((2*j_max, 2*j_max))
    L = np.zeros((2*j_max, 2*j_max))
    Q_c = np.zeros((2*j_max, 2*j_max)) # process noise
    H = np.zeros(2*j_max) # ovservatioanl matrix
    
    m_0 = np.zeros(2*j_max) # zeroth state mean
    P_0 = np.zeros((2*j_max, 2*j_max)) # zeroth state covariance
    
    R = noise_var # observational noise
    
    for j in np.arange(0, j_max):
        Fpj, Lpj, P0pj, Hpj, _ = get_params_p_j(j, omega_0, ell)
        F[2*j:2*j+2, 2*j:2*j+2] = Fpj
        L[2*j:2*j+2, 2*j:2*j+2] = Lpj
        P_0[2*j:2*j+2, 2*j:2*j+2] = P0pj
        H[2*j:2*j+2] = Hpj

        #ell_inv_sq = 1.0/ell/ell
        #F[2*j, 2*j+1] = -omega_0*j
        #F[2*j+1, 2*j] = omega_0*j
        #L[2*j, 2*j] = 1.0
        #L[2*j+1, 2*j+1] = 1.0
        #if j == 0:
        #    q = special.iv(0, ell_inv_sq)/np.exp(ell_inv_sq)
        #else:
        #    q = 2.0 * special.iv(j, ell_inv_sq)/np.exp(ell_inv_sq)
        #P_0[2*j, 2*j] = q
        #P_0[2*j+1, 2*j+1] = q
        #H[2*j] = 1.0
    return F, L, H, R, m_0, P_0, Q_c

def get_params_qp(j_max, omega_0, ellp, noise_var, ellq, sig_var):
    #ell_inv_sq = 1.0/ell/ell

    lmbda = 1.0/ellq
    Fq = -np.ones(1) * lmbda
    Lq = np.ones(1)
    Qcq = np.ones(1) * 2.0*sig_var*np.sqrt(np.pi)*lmbda*special.gamma(1.0)/special.gamma(0.5)
    Hq = np.ones(1) # ovservatioanl matrix
    P0q = np.ones(1)
    
    F = np.zeros((2*j_max, 2*j_max))
    L = np.zeros((2*j_max, 2*j_max))
    Q_c = np.zeros((2*j_max, 2*j_max)) # process noise
    H = np.zeros(2*j_max) # ovservatioanl matrix
    
    m_0 = np.zeros(2*j_max) # zeroth state mean
    P_0 = np.zeros((2*j_max, 2*j_max)) # zeroth state covariance
    
    R = noise_var # observational noise
    
    I2 = np.diag(np.ones(2))
    for j in np.arange(0, j_max):
        Fpj, Lpj, P0pj, Hpj, qj = get_params_p_j(j, omega_0, ell)
        #Fpj = np.array([[0.0, -omega_0*j], [omega_0*j, 0.0]])
    
        #Lpj = I2
    
        #if j == 0:
        #    q = special.iv(0, ell_inv_sq)/np.exp(ell_inv_sq)
        #else:
        #    q = 2.0 * special.iv(j, ell_inv_sq)/np.exp(ell_inv_sq)

        #P0pj = I2 * q
        ##Qcpj = np.zeros((2, 2))
        #Hpj = np.array([1.0, 0.0])
        
        F[2*j:2*j+2, 2*j:2*j+2] = np.kron(Fq, I2) + np.kron(np.diag(np.ones(1)), Fpj)
        L[2*j:2*j+2, 2*j:2*j+2] = np.kron(Lq, Lpj)
        Q_c[2*j:2*j+2, 2*j:2*j+2] = np.kron(Qcq, I2 * qj)
        P_0[2*j:2*j+2, 2*j:2*j+2] = np.kron(P0q, P0pj)
        H[2*j:2*j+2] = np.kron(Hq, Hpj)
        
    return F, L, H, R, m_0, P_0, Q_c

n = 50
time_range = 200
t = np.random.uniform(0.0, time_range, n)
t = np.sort(t)
var = 1.0
sig_var = np.random.uniform(0.999, 0.999)
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
ellq_max = None
kf_max =None

j_max = 2
ell = 10
if cov_type == "periodic":
    ellqs = [length_scale]
else:
    ellqs = np.linspace(length_scale/2, length_scale*2, 20) 
    
for omega_0 in [2.0*np.pi*freq]:#np.linspace(np.pi*freq, 4.0*np.pi*freq, 100):
    
    for ellq in ellqs:
        print ellq
        if cov_type == "periodic":
            F, L, H, R, m_0, P_0, Q_c = get_params_p(j_max, omega_0, ell, noise_var)
        else:
            F, L, H, R, m_0, P_0, Q_c = get_params_qp(j_max, omega_0, ell, noise_var, ellq, sig_var)
            
        #F = 1.0
        #L = 1.0
        #H = 1.0
        #R = 0.0
        #m_0 = 0.0
        #P_0 = 1.0
        #Q_c = 1.0
        
        kf = kalman.kalman(t=t, y=y, F=F, L=L, H=H, R=R, m_0=m_0, P_0=P_0, Q_c=Q_c)
        y_means, loglik = kf.filter()
        #print omega_0, loglik
        if loglik_max is None or loglik > loglik_max:
           loglik_max = loglik
           y_means_max = y_means
           omega_max = omega_0
           ellq_max = ellq
           kf_max = kf

print omega_max, freq*2.0*np.pi
print ellq_max, length_scale
ax1.plot(t[1:], y_means_max, 'r--')

y_means = kf_max.smooth()
ax1.plot(t[1:-1], y_means, 'g--')

fig.savefig('test.png')
plt.close(fig)

