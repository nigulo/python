
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
from scipy.special import gamma
from scipy.special import kv

try:
    from scipy.linalg import solve_lyapunov as solve_continuous_lyapunov
except ImportError:  # pragma: no cover; github.com/scipy/scipy/pull/8082
    from scipy.linalg import solve_continuous_lyapunov

import kalman
from cov_exp_quad import cov_exp_quad
from cov_matern import cov_matern

cov_type = "linear_trend"
#cov_type = "periodic"
#cov_type = "quasiperiodic"
#cov_type = "exp_quad"
#cov_type = "matern"

matern_p = 1

def calc_cov_linear_trend(t, trend_var, c = 0.0):
    k = np.zeros((len(t), len(t)))
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            k[i, j] = trend_var * (t[i] - c)*(t[j] - c)
            k[j, i] = k[i, j]
    return k

def calc_cov_matern(t, length_scale, sig_var, nu):
    k = np.zeros((len(t), len(t)))
    sqrt_two_nu_inv_l = np.sqrt(2.0*nu)/length_scale
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            if i == j:
                k[i, j] = sig_var
            else:
                tau = np.abs(t[i]-t[j])
                k[i, j] = sig_var*2.0**(1.0-nu)/gamma(nu)*(sqrt_two_nu_inv_l*tau)**nu*kv(nu, sqrt_two_nu_inv_l*tau)
            #test = sig_var*np.exp(-np.abs(t[i]-t[j])/length_scale)
            #assert(k[i, j] - test < 1e-15)
            k[j, i] = k[i, j]
    return k

def calc_cov_exp_quad(t, length_scale, sig_var):
    k = np.zeros((len(t), len(t)))
    inv_l2 = 0.5/(length_scale*length_scale)
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            k[i, j] = sig_var*np.exp(-inv_l2*(t[i]-t[j])**2)
            k[j, i] = k[i, j]
    return k

def calc_cov_qp(t, f, length_scale, sig_var):
    k = np.zeros((len(t), len(t)))
    inv_l2 = 0.5/(length_scale*length_scale)
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            k[i, j] = sig_var*np.exp(-inv_l2*(t[i]-t[j])**2)*(np.cos(2 * np.pi*f*(t[i] - t[j])))
            #k[i, j] = sig_var*np.exp(-np.abs(t[i]-t[j])/length_scale)*(np.cos(2 * np.pi*f*(t[i] - t[j])))
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
    H = np.zeros(2*j_max) # observatioanl matrix
    
    m_0 = np.zeros(2*j_max) # zeroth state mean
    P_0 = np.zeros((2*j_max, 2*j_max)) # zeroth state covariance
    
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
    return F, L, H, m_0, P_0, Q_c

def get_params_qp(j_max, omega_0, ellp, noise_var, ellq, sig_var):
    #ell_inv_sq = 1.0/ell/ell
    Nq = 6
    Fq, Lq, Hq, _, _, P0q, Qcq = get_params_exp_quad(ellq, noise_var, sig_var, N=Nq)
    
    #lmbda = 1.0/ellq
    #Fq = -np.ones(1) * lmbda
    #Lq = np.ones(1)
    #Qcq = np.ones(1) * 2.0*sig_var*np.sqrt(np.pi)*lmbda*special.gamma(1.0)/special.gamma(0.5)
    #Hq = np.ones(1) # observatioanl matrix
    #P0q = np.ones(1)
    
    F = np.zeros((2*Nq*j_max, 2*Nq*j_max))
    L = np.zeros((2*Nq*j_max, 2*j_max))
    Q_c = np.zeros((2*j_max, 2*j_max)) # process noise
    H = np.zeros(2*Nq*j_max) # observatioanl matrix
    
    m_0 = np.zeros(2*Nq*j_max) # zeroth state mean
    P_0 = np.zeros((2*Nq*j_max, 2*Nq*j_max)) # zeroth state covariance
    
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
        
        F[2*Nq*j:2*Nq*j+2*Nq, 2*Nq*j:2*Nq*j+2*Nq] = np.kron(Fq, I2) + np.kron(np.diag(np.ones(Nq)), Fpj)
        L[2*Nq*j:2*Nq*j+2*Nq, 2*j:2*j+2] = np.kron(Lq, Lpj)
        Q_c[2*j:2*j+2, 2*j:2*j+2] = np.kron(Qcq, I2 * qj)
        P_0[2*Nq*j:2*Nq*j+2*Nq, 2*Nq*j:2*Nq*j+2*Nq] = np.kron(P0q, P0pj)
        H[2*Nq*j:2*Nq*j+2*Nq] = np.kron(Hq, Hpj)
        
    return F, L, H, m_0, P_0, Q_c

def get_params_exp_quad(ell, noise_var, sig_var, N=6):
    
    k = cov_exp_quad(N=N)
    F, q = k.get_F_q(sig_var, ell)

    n_dim = np.shape(F)[0]
    Q_c = np.ones((1, 1))*q
    L = np.zeros((n_dim, 1))
    L[n_dim - 1] = 1.0
    
    H = np.zeros(n_dim) # observatioanl matrix
    H[0] = 1.0
    
    m_0 = np.zeros(n_dim) # zeroth state mean
    #P_0 = np.diag(np.ones(n_dim))#*sig_var) # zeroth state covariance
    
    #P_0 = solve_continuous_lyapunov(F, -np.dot(L, np.dot(Q_c, L.T)))
    P_0 = solve_continuous_lyapunov(F, -np.dot(L, L.T)*Q_c[0,0])
    #print P_0
    
    return F, L, H, m_0, P_0, Q_c

def get_params_matern(ell, noise_var, sig_var):
    
    k = cov_matern(p=matern_p)
    F, q = k.get_F_q(sig_var, ell)

    n_dim = np.shape(F)[0]
    Q_c = np.ones((1, 1))*q
    L = np.zeros((n_dim, 1))
    L[n_dim - 1] = 1.0
    
    H = np.zeros(n_dim) # observatioanl matrix
    H[0] = 1.0
    
    m_0 = np.zeros(n_dim) # zeroth state mean
    #P_0 = np.diag(np.ones(n_dim))#*sig_var) # zeroth state covariance
    
    #P_0 = solve_continuous_lyapunov(F, -np.dot(L, np.dot(Q_c, L.T)))
    P_0 = solve_continuous_lyapunov(F, -np.dot(L, L.T)*Q_c[0,0])
    #print P_0
    
    #Q_c[n_dim - 1, n_dim - 1] = q

    return F, L, H, m_0, P_0, Q_c

def get_params_linear_trend(t, slope, intercept):
    
    delta_t = t[1:] - t[:-1]


    Q_c = np.zeros((1, 1))
    L = np.zeros((2, 1))
    L[1] = 1.0
    
    H = np.array([0.0, 1.0]) # observatioanl matrix
    
    F = np.zeros((len(delta_t), 2, 2))
    i = 0
    for dt in delta_t:
        F[i] = np.array([[1, 0], [slope*dt,1]])
        i += 1
    #P_0 = np.diag(np.ones(n_dim))#*sig_var) # zeroth state covariance
    
    #P_0 = solve_continuous_lyapunov(F, -np.dot(L, np.dot(Q_c, L.T)))
    m_0 = np.array([1.0, slope*t[0] + intercept]) # zeroth state mean
    P_0 = np.diag(np.array([0.0, 0.0]))#slope**2*t[0]**2 + intercept**2]))
    #print P_0
    
    #Q_c[n_dim - 1, n_dim - 1] = q

    return F, L, H, m_0, P_0, Q_c

n = 50
time_range = 200
t = np.random.uniform(0.0, time_range, n)
t = np.sort(t)
var = 2.0
sig_var = 0.0#np.random.uniform(0.99*var, 0.99*var)
trend_var = np.random.uniform(0.9999*var, 0.9999*var)
noise_var = var - sig_var - trend_var
t -= np.mean(t)

#p = time_range/12.54321#
p = time_range/5#np.random.uniform(time_range/200, time_range/5)
freq = 1.0/p
mean = np.random.uniform(-100.0, 100.0)

if cov_type == "linear_trend":
    k = calc_cov_linear_trend(t, trend_var)
elif cov_type == "periodic":
    length_scale = 1e10*p
    k = calc_cov_p(t, freq, sig_var)
elif cov_type == "quasiperiodic":
    length_scale = np.random.uniform(p/2.0, 4.0*p)
    k = calc_cov_qp(t, freq, length_scale, sig_var)
elif cov_type == "exp_quad":
    length_scale = np.random.uniform(time_range/10, time_range/5)
    k = calc_cov_exp_quad(t, length_scale, sig_var)
elif cov_type == "matern":
    length_scale = np.random.uniform(time_range/10, time_range/5)
    k = calc_cov_matern(t, length_scale, sig_var, matern_p + 0.5)
else:
    assert(True==False)

k += np.diag(np.ones(n) * noise_var)
l = la.cholesky(k)
s = np.random.normal(0.0, 1.0, n)

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
slope_max = None
intercept_max = None
kf_max =None

j_max = 2
ell = 10
true_slope = np.sqrt(trend_var)
slope_hat, intercept_hat, r_value, p_value, std_err = stats.linregress(t, y)
print "slope_hat, intercept_hat", slope_hat, intercept_hat
slopes = np.linspace(slope_hat/2, slope_hat*2, 10)#[true_slope, -true_slope]
intercepts = np.linspace(mean/2, mean*2, 10)#[true_slope, -true_slope]
if cov_type == "linear_trend":
    ellqs = [length_scale]
    omegas = [2.0*np.pi*freq]
elif cov_type == "periodic":
    ellqs = [length_scale]
    omegas = [2.0*np.pi*freq]
elif cov_type == "quasiperiodic":
    ellqs = np.linspace(length_scale/2, length_scale*2, 10) 
    omegas = np.linspace(2.0*np.pi*freq/2, 2.0*np.pi*freq*2, 10) 
    #omegas = [2.0*np.pi*freq]
elif cov_type == "exp_quad":
    #ellqs = [length_scale] 
    ellqs = np.linspace(length_scale/2, length_scale*2, 20) 
    #ellqs = [length_scale/2, length_scale] 
    #ellqs = [length_scale/2, length_scale, length_scale*2] 
    omegas = [2.0*np.pi*freq]
elif cov_type == "matern":
    #ellqs = [length_scale] 
    ellqs = np.linspace(length_scale/2, length_scale*2, 20) 
    #ellqs = [length_scale/2, length_scale, length_scale*2] 
    #ellqs = [length_scale] 
    omegas = [2.0*np.pi*freq]
else:           
    assert(True==False)

y_means_true = None
for omega_0 in omegas:
    
    for ellq in ellqs:
        for slope in slopes:
            for intercept in intercepts:
                if cov_type == "linear_trend":
                    F, L, H, m_0, P_0, Q_c = get_params_linear_trend(t, slope=slope, intercept=intercept)
                elif cov_type == "periodic":
                    F, L, H, m_0, P_0, Q_c = get_params_p(j_max, omega_0, ell, noise_var)
                elif cov_type == "quasiperiodic":
                    F, L, H, m_0, P_0, Q_c = get_params_qp(j_max, omega_0, ell, noise_var, ellq, sig_var)
                elif cov_type == "exp_quad":
                    F, L, H, m_0, P_0, Q_c = get_params_exp_quad(ellq, noise_var, sig_var)
                elif cov_type == "matern":
                    F, L, H, m_0, P_0, Q_c = get_params_matern(ellq, noise_var, sig_var)
                else:           
                    assert(True==False)
                #F = 1.0
                #L = 1.0
                #H = 1.0
                #R = 0.0
                #m_0 = 0.0
                #P_0 = 1.0
                #Q_c = 1.0
                    
                R = noise_var # observational noise
                
                kf = kalman.kalman(t=t, y=y, F=F, L=L, H=H, R=R, m_0=m_0, P_0=P_0, Q_c=Q_c, noise_int_prec=100, F_is_A=True)
                y_means, loglik = kf.filter()
                print 2.0*np.pi/omega_0, ellq, slope, intercept, loglik
                if loglik_max is None or loglik > loglik_max:
                   loglik_max = loglik
                   y_means_max = y_means
                   omega_max = omega_0
                   ellq_max = ellq
                   slope_max = slope
                   intercept_max = intercept
                   kf_max = kf
                if ellq == length_scale and omega_0 == freq*2.0*np.pi and abs(slope) == slope_hat and intercept == mean:
                    y_means_true = y_means

print "period", 2.0*np.pi/omega_max, p
print "length-scale", ellq_max, length_scale
print "slope", slope_max, slope_hat
print "intercept", intercept_max, mean
ax1.plot(t[1:], y_means_max, 'r--')
if y_means_true is not None:
    ax1.plot(t[1:], y_means_true, 'g--')

#y_means = kf_max.smooth()
#ax1.plot(t[1:-1], y_means, 'g--')

fig.savefig('test.png')
plt.close(fig)

