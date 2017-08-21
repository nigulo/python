import numpy as np
import matplotlib.pyplot as plt
import time
import BGLST
import bayes_lin_reg
from scipy import stats
from astropy.stats import LombScargle
from scipy.stats import norm

def calc_BGLS(t, y, w, freq):
    tau = 0.5 * np.arctan(sum(w * np.sin(4 * np.pi * t * freq))/sum(w * np.cos(4 * np.pi * t * freq)))
    c = sum(w * np.cos(2.0 * np.pi * t * freq - tau))
    s = sum(w * np.sin(2.0 * np.pi * t * freq - tau))
    cc = sum(w * np.cos(2.0 * np.pi * t * freq - tau)**2)
    ss = sum(w * np.sin(2.0 * np.pi * t * freq - tau)**2)
    yc = sum(w * y * np.cos(2.0 * np.pi * t * freq - tau))
    ys = sum(w * y * np.sin(2.0 * np.pi * t * freq - tau))
    Y = sum(w * y)
    W = sum(w)

    assert(cc > 0)
    assert(ss > 0)
    
    K = (c**2/cc + s**2/ss - W)/2.0
    L = Y - c*yc/cc - s*ys/ss
    M = (yc**2/cc + ys**2/ss)/2.0
    log_prob = np.log(1.0 / np.sqrt(abs(K) * cc * ss)) + (M - L**2/4.0/K)
    return log_prob

time_range = 200.0
n = 500
t = np.random.uniform(0.0, time_range, n)
#t = np.sort(t)
#t1 = t[0:int(0.1*n)]
#t2 = t[int(0.2*n):int(0.3*n)]
#t3 = t[int(0.5*n):int(0.7*n)]
#t4 = t[int(0.8*n):int(0.9*n)]
#t = np.concatenate([t1, t2, t3, t4])
#print t
n = len(t)
duration = max(t) - min(t)
#t = np.concatenate([t[:n/4], t[n/2:n*3/4]])
#n = len(t)
#t = np.random.randint(time_range, size=n)+np.random.rand(n)
freq = 1.0/278.12345678
sigma = 0.05
epsilon = np.random.normal(0, sigma, n)
#y = np.ones(len(t))
#y = 0.01*np.cos(2 * np.pi * freq * t) + sigma + t/20
#y = sigma - t/20
y = np.cos(2 * np.pi * freq * t) + epsilon



#y = sigma
w = np.ones(n)/sigma**2


freq_start = 0.001
freq_end = 0.1
freq_count = 1000

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

max_prob = max(probs)
max_prob_index = np.argmax(probs)
best_freq = freqs[max_prob_index]

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=False)
fig.set_size_inches(6, 7)

ax1.text(0.95, 0.9,'(a)', horizontalalignment='center', transform=ax1.transAxes)
ax2.text(0.95, 0.9,'(b)', horizontalalignment='center', transform=ax2.transAxes)


ax1.scatter(t, y, marker='+', color ='k')
tau, (A, B, alpha, beta), _, y_model_1, loglik = bglst.model(best_freq)
bic = 2 * loglik - np.log(n) * 5
print A, B, alpha, beta
t_model = np.linspace(min(t), max(t), 1000)
y_model = np.cos(t_model * 2.0 * np.pi * best_freq - tau) * A  + np.sin(t_model * 2.0 * np.pi * best_freq - tau) * B + t_model * alpha + beta
ax1.plot(t_model, y_model, 'r-')

min_prob = min(probs)
norm_probs = (probs - min_prob) / (max_prob - min_prob)
ax2.plot(freqs, norm_probs, 'r-')
ax2.plot([best_freq, best_freq], [0, norm_probs[max_prob_index]], 'r--')

bglst_m = BGLST.BGLST(t, y_model_1, np.ones(n)/np.var(y),
                    w_A = 2.0/np.var(y), A_hat = 0.0,
                    w_B = 2.0/np.var(y), B_hat = 0.0,
                    w_alpha = duration**2 / np.var(y), alpha_hat = slope, 
                    w_beta = 1.0 / (np.var(y) + intercept**2), beta_hat = intercept)
(freqs, probs_m) = bglst_m.calc_all(freq_start, freq_end, freq_count)
max_prob_m = max(probs_m)
max_prob_index_m = np.argmax(probs_m)
best_freq_m = freqs[max_prob_index_m]
min_prob_m = min(probs_m)
norm_probs_m = (probs_m- min_prob_m) / (max_prob_m - min_prob_m)
#ax2.plot(freqs, norm_probs_m, 'g-')
max_prob_m = max(probs_m)
max_prob_index_m = np.argmax(probs_m)
best_freq_m = freqs[max_prob_index_m]
min_prob_m = min(probs_m)
norm_probs_m = (probs_m- min_prob_m) / (max_prob_m - min_prob_m)


print "BGLST: ", freq, best_freq, max_prob

_, _, _, loglik_null = bayes_lin_reg.bayes_lin_reg(t, y, w)
bic_null = 2 * loglik_null - np.log(n) * 2

print bic - bic_null


###############################################################################
# LS

ls = LombScargle(t, y, sigma)
power = ls.power(freqs, normalization='psd')#/np.var(y)

max_power_ind = np.argmax(power)
max_power = power[max_power_ind]
best_freq = freqs[max_power_ind]
y_model = ls.model(t_model, best_freq)
ax1.plot(t_model, y_model, 'g--')
print "LS: ", freq, best_freq, max_power

min_power = min(power)
norm_powers = (power - min_power) / (max_power - min_power)

ax2.plot(freqs, norm_powers, 'g-')
ax2.plot([best_freq, best_freq], [0, norm_powers[max_power_ind]], 'g--')

###############################################################################
# LS detrended

y_fit = t * slope + intercept
y -= y_fit
ls = LombScargle(t, y, sigma)
power = ls.power(freqs, normalization='psd')#/np.var(y)

max_power_ind = np.argmax(power)
max_power = power[max_power_ind]
best_freq = freqs[max_power_ind]
y_model = ls.model(t_model, best_freq)
ax1.plot(t_model, y_model+t_model * slope + intercept, 'b-')
ax1.plot(t_model, t_model * slope + intercept, 'b-.')
print "LS detrended: ", freq, best_freq, max_power

min_power = min(power)
norm_powers = (power - min_power) / (max_power - min_power)

ax2.plot(freqs, norm_powers, 'b-')
ax2.plot([best_freq, best_freq], [0, norm_powers[max_power_ind]], 'b--')

ax1.set_xlabel(r'Time')#,fontsize=20)
ax1.set_ylabel(r'Signal')#,fontsize=20)
ax1.set_xlim([0, 180])

ax2.set_xlabel(r'Frequency')#,fontsize=20)
ax2.set_ylabel(r'Normalized log probability/power')#,fontsize=20)
ax2.set_xlim([0.001, 0.04])

fig.savefig("testBGLST_long.eps")
