# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import numpy.linalg as la
import sys
import random
from filelock import FileLock

import os
import os.path

def test_validity():
    for n in np.random.randint(10000, size=10000):
        time_range = 100
        #x = np.arange(0, time_range, time_range/float(n))
        t = np.random.randint(time_range, size=n)+np.random.rand(n)
        #y = np.random.rand(n)#np.cos(x * w*np.random.rand())#np.random.rand(n)
        w = np.random.normal(10, 5, n)
        w = abs(w)
        freq = np.random.rand() * time_range
        while freq < 1.0 / time_range:
            freq = np.random.rand() * time_range
            
        
        c = sum(w * np.cos(2 * np.pi * t * freq))
        s = sum(w * np.sin(2 * np.pi * t * freq))
        cc = sum(w * np.cos(2 * np.pi * t * freq)**2)
        ss = sum(w * np.sin(2 * np.pi * t * freq)**2)
        ct = sum(w * np.cos(2 * np.pi * t * freq) * t)
        st = sum(w * np.sin(2 * np.pi * t * freq) * t)
        tt = sum(w * t * t)
        T = sum(w * t)
        W = sum(w)
        
        K1 = c**2/cc + s**2/ss - W
        K = ct**2/cc + st**2/ss - tt
        N = (ct * c * ss + st * s * cc - T * cc * ss) / cc / ss
        Q = c**2/2/cc + s**2/2/ss - W - N**2/4/K
        if K1 >= 0:
            print "K1", K1, n, freq
        if K >= 0:
            print "K", K, n, freq
        if Q >= 0:
            print "Q", Q, n, freq


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

def calc_BGLST(t, y, w, freq):
    tau = 0.5 * np.arctan(sum(w * np.sin(4 * np.pi * t * freq))/sum(w * np.cos(4 * np.pi * t * freq)))
    c = sum(w * np.cos(2.0 * np.pi * t * freq - tau))
    s = sum(w * np.sin(2.0 * np.pi * t * freq - tau))
    cc = sum(w * np.cos(2.0 * np.pi * t * freq - tau)**2)
    ss = sum(w * np.sin(2.0 * np.pi * t * freq - tau)**2)
    ct = sum(w * np.cos(2.0 * np.pi * t * freq - tau) * t)
    st = sum(w * np.sin(2.0 * np.pi * t * freq - tau) * t)
    tt = sum(w * t * t)
    T = sum(w * t)
    yt = sum(w * y * t)
    yc = sum(w * y * np.cos(2.0 * np.pi * t * freq - tau))
    ys = sum(w * y * np.sin(2.0 * np.pi * t * freq - tau))
    yy = sum(w * y * y)
    Y = sum(w * y)
    W = sum(w)
    assert(cc > 0)
    assert(ss > 0)
    
    K = (ct**2/cc + st**2/ss - tt)/2.0
    assert(K < 0)
    M = (yt * cc * ss - yc * ct * ss - ys * st * cc) / cc / ss
    N = (ct * c * ss + st * s * cc - T * cc * ss) / cc / ss
    Q = c**2/2.0/cc + s**2/2.0/ss - W/2.0 - N**2/4.0/K
    assert(Q < 0)
    P = -yc*c/cc - ys*s/ss + Y - M*N/2.0/K
    log_prob = np.log(2.0 * np.pi**2 / np.sqrt(cc * ss * K * Q)) + (yc**2/2.0/cc + ys**2/2.0/ss - M**2/4.0/K - P**2/4.0/Q - yy/2.0)
    return log_prob


time_range = 100.0
n = 1000
t = np.random.uniform(0.0, time_range, n)
#t = np.random.randint(time_range, size=n)+np.random.rand(n)
t = np.sort(t)
freq = 1/12.345678
sigma = np.random.normal(0, 1, n)
assert(np.all(sigma != 0))
y = np.cos(2 * np.pi * freq * t) + sigma + t/20
w = np.ones(len(sigma))/sigma**2

plt.scatter(t, y)
plt.show()

freqs = np.arange(0.0001, 0.2, 0.0001)
probs = []
for f in freqs:
    probs.append(calc_BGLST(t, y, w, f))

max_prob = max(probs)
max_prob_index = np.argmax(probs)
max_freq = freqs[max_prob_index]
min_prob = min(probs)
norm_probs = (probs - min_prob) / (max_prob - min_prob)
plt.plot(freqs, norm_probs)
plt.stem([max_freq], [norm_probs[max_prob_index]])
plt.show()
print freq, max_freq
