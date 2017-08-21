# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:10:51 2017

@author: nigul
"""

from astropy.stats import LombScargle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import os
import os.path

n = 5000
t_max = 500.0
season_period = 10.0
num_seasons = np.floor(t_max / season_period)
season_size = 10.0
nyquist = float(n) / t_max / 2
t = np.linspace(0, t_max, n)
#t = np.random.randint(t_max, size=n)+np.random.rand(n)
#season_starts = np.floor(t/season_period) * season_period
#t = t[np.where(np.logical_and(t >= season_starts, t < season_starts + season_size))]
print(len(t))
mu = 0
sigma = 1
freqs = np.linspace(0, nyquist/100, 100)
#y = np.sin(0.5*t) + np.random.normal(mu, sigma, len(t))

#y = np.random.normal(mu, sigma, num_seasons)
#y = np.repeat(y, n / num_seasons)
#t = t[:len(y)]

y = np.zeros(len(t))
season_start = 0
season_mean_var = 1
season_var = 1
season_mean = np.random.normal(0, season_mean_var)
for i in np.arange(0, len(t)):
    if (t[i] >= season_start + season_period):
        season_start += season_period
        season_mean = np.random.normal(0, season_mean_var)
    y[i] = np.random.normal(season_mean, season_var)
    
mean = np.mean(y)
std = np.std(y)
y -= mean
y /= std

power = LombScargle(t, y, nterms=1).power(freqs, normalization='psd')#/np.var(y)
plt.plot(t, y, 'b+')
plt.savefig("mixed_whitenoise_dat.png")
plt.close()
plt.plot(freqs, power, "b-")
plt.savefig("mixed_whitenoise_spec.png")
plt.close()