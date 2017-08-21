import numpy as np
import matplotlib.pyplot as plt
import time
import BGLST
import bayes_lin_reg
from scipy import stats
from astropy.stats import LombScargle
import numpy.linalg as la
import mw_utils
import os

offset = 1979.3452
down_sample_factor = 8

dataset = "counter_example_2.dat"

def calc_cov(t, f, sig_var, trend_var, c):
    k = np.zeros((len(t), len(t)))
    for i in np.arange(0, len(t)):
        for j in np.arange(i, len(t)):
            k[i, j] = sig_var*(np.cos(2 * np.pi*f*(t[i] - t[j]))) + trend_var * (t[i] - c) * (t[j] - c)
            k[j, i] = k[i, j]
    return k


files = []

for root, dirs, dir_files in os.walk("cleaned"):
    for file in dir_files:
        if file[-4:] == ".dat":
            star = file[:-4]
            star = star.upper()
            if (star[-3:] == '.CL'):
                star = star[0:-3]
            if (star[0:2] == 'HD'):
                star = star[2:]
            files.append(file)

def select_dataset():
    file = files[np.random.choice(len(files))]
    print file
    
    dat = np.loadtxt("cleaned/"+file, usecols=(0,1), skiprows=1)

    t_orig = dat[:,0]
    t_orig /= 365.25
    t_orig += offset


    if down_sample_factor >= 2:
        indices = np.random.choice(len(t_orig), len(t_orig)/down_sample_factor, replace=False, p=None)
        indices = np.sort(indices)
    
        t = t_orig[indices]

    return t


f = 0.0492934969222

if dataset != None:
    dat = np.loadtxt(dataset, usecols=(0,1), skiprows=1)
    t = dat[:,0]
    y = dat[:,1]
    duration = max(t) - min(t)
    n = len(t)
else:

    duration = 0
    while duration < 30:
        t = select_dataset()
        duration = max(t) - min(t)
    
    mean_t = np.mean(t)
    t -= mean_t
        
    n = len(t)
    
    var = 1.0
    sig_var = 0.314303915092#np.random.uniform(0.2, 0.8)
    noise_var = np.ones(n) * (var - sig_var)
    trend_var = 0.0177057202816#np.random.uniform(0.0, 1.0) * var / duration
    mean = 0.5
    
    mean = 0.5
    
    k = calc_cov(t, f, sig_var, trend_var, 0.0) + np.diag(noise_var)
    l = la.cholesky(k)
    s = np.random.normal(0, 1, n)
    
    y = np.repeat(mean, n) + np.dot(l, s)
    y += mean
    
    t += mean_t


dat = np.column_stack((t, y))
np.savetxt("counter_example.dat", dat, fmt='%f')


freq_start = 0.0001
freq_end = 0.5
freq_count = 1000

noise_var_prop = mw_utils.get_seasonal_noise_var(t, y)
w = np.ones(n) / noise_var_prop

start = time.time()

slope, intercept, r_value, p_value, std_err = stats.linregress(t, y)
print "slope, intercept", slope, intercept
bglst = BGLST.BGLST(t, y, w, 
                    w_A = 2.0/np.var(y), A_hat = 0.0,
                    w_B = 2.0/np.var(y), B_hat = 0.0,
                    w_alpha = duration**2 / np.var(y), alpha_hat = slope, 
                    w_beta = 1.0 / (np.var(y) + intercept**2), beta_hat = intercept)
(freqs, probs) = bglst.calc_all(freq_start, freq_end, freq_count)
end = time.time()
print(end - start)

#probs = np.zeros(freq_count)
#bglst = BGLST.BGLST(t, y, w)
#start = time.time()
#i = 0
#for f in freqs:
#    probs[i] = bglst.calc(f)
#    i += 1
#    #probs.append(calc_BGLS(t, y, w, f))
#end = time.time()
#print(end - start)


#print probs - probs1

bglst_local_maxima_inds = mw_utils.find_local_maxima(probs)
f_opt_bglst_ind = np.argmax(probs[bglst_local_maxima_inds])
best_freq = freqs[bglst_local_maxima_inds][f_opt_bglst_ind]

best_freq_ind = np.argmax(probs)
best_freq = freqs[best_freq_ind]

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=False)
fig.set_size_inches(6, 7)

ax1.text(0.95, 0.9,'(a)', horizontalalignment='center', transform=ax1.transAxes)
ax2.text(0.95, 0.9,'(b)', horizontalalignment='center', transform=ax2.transAxes)


ax1.scatter(t, y, marker='+', color ='k')
tau, (A, B, alpha, beta), _, y_model_1, loglik = bglst.model(best_freq)
bic = 2 * loglik - np.log(n) * 5
print "A, B, alpha, beta, tau", A, B, alpha, beta, tau
t_model = np.linspace(min(t), max(t), 1000)
y_model = np.cos(t_model * 2.0 * np.pi * best_freq - tau) * A  + np.sin(t_model * 2.0 * np.pi * best_freq - tau) * B + t_model * alpha + beta
ax1.plot(t_model, y_model, 'r-')
#ax1.plot(t_model, np.sin(t_model * 2.0 * np.pi * best_freq - tau) * B, 'r--')
#ax1.plot(t_model, np.cos(t_model * 2.0 * np.pi * best_freq - tau) * A, 'r-.')
#ax1.plot(t_model, t_model * alpha + beta, 'r:')


max_prob = max(probs)
min_prob = min(probs)
norm_probs = (probs - min_prob) / (max_prob - min_prob)
ax2.plot(freqs, norm_probs, 'r-')
ax2.plot([best_freq, best_freq], [0, norm_probs[bglst_local_maxima_inds][f_opt_bglst_ind]], 'r--')

print "BGLST: ", f, best_freq, max_prob


_, _, _, loglik_null = bayes_lin_reg.bayes_lin_reg(t, y, w)
bic_null = 2 * loglik_null - np.log(n) * 2

print bic - bic_null

###############################################################################
# LS

ls = LombScargle(t, y, np.sqrt(noise_var_prop))
power = ls.power(freqs, normalization='psd')#/np.var(y)

max_power_ind = np.argmax(power)
max_power = power[max_power_ind]
best_freq = freqs[max_power_ind]
y_model = ls.model(t_model, best_freq)
#ax1.plot(t_model, y_model, 'g-')
print "LS: ", f, best_freq, max_power

min_power = min(power)
norm_powers = (power - min_power) / (max_power - min_power)

#ax2.plot(freqs, norm_powers, 'g-')
#ax2.plot([best_freq, best_freq], [0, norm_powers[max_power_ind]], 'g--')

###############################################################################
# LS detrended

y_fit = t * slope + intercept
y -= y_fit
ls = LombScargle(t, y, np.sqrt(noise_var_prop))
power = ls.power(freqs, normalization='psd')#/np.var(y)

max_power_ind = np.argmax(power)
max_power = power[max_power_ind]
best_freq = freqs[max_power_ind]

print "LS detrended: ", f, best_freq, max_power

y_model = ls.model(t_model, best_freq)
ax1.plot(t_model, y_model+t_model * slope + intercept, 'b-')
ax1.plot(t_model, t_model * slope + intercept, 'b--')

min_power = min(power)
norm_powers = (power - min_power) / (max_power - min_power)

ax2.plot(freqs, norm_powers, 'b-')
ax2.plot([best_freq, best_freq], [0, norm_powers[max_power_ind]], 'b--')

ax1.set_xlabel(r'Time')#,fontsize=20)
ax1.set_ylabel(r'Signal')#,fontsize=20)
ax1.set_xlim([min(t), max(t)])

ax2.set_xlabel(r'Frequency')#,fontsize=20)
ax2.set_ylabel(r'Normalized log probability/power')#,fontsize=20)
ax2.set_xlim([0.001, 0.5])

fig.savefig("counter_example.eps")

