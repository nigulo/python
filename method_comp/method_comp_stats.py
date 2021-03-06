# -*- coding: utf-8 -*-
"""
Test of equivalence of general D^2 statistic in original and vector formulation
@author: nigul
"""

import sys
sys.path.append('../kalman/')
import numpy as np
from scipy import stats
from collections import OrderedDict as od
import numpy.linalg as la
import GPR_QP
import kalman_utils as ku
import method_comp_utils as mcu
from filelock import FileLock
import os

print np.version.version
#cov_type = "periodic"
cov_type = "quasiperiodic"

n_fixed = 50
sig_var_fixed = 0.1

if len(sys.argv) > 1:
    sig_var_fixed = float(sys.argv[1])

uniform_sampling = True

if len(sys.argv) > 2:
    uniform_sampling = int(sys.argv[2]) == 1

num_experiments = 100
if len(sys.argv) > 3:
    num_experiments = int(sys.argv[3])


print num_experiments, n_fixed, sig_var_fixed, uniform_sampling

for _ in np.arange(0, num_experiments):

    with FileLock("GPRLock"):
        if not os.path.isfile("exp_no.txt"):
            with open("exp_no.txt", "w") as fp:
               experiment_index = 0
               fp.write("%s\n" % 1)
        else:
            with open("exp_no.txt", "r") as fp:
               experiment_index = int(fp.readline())
            file = open("exp_no.txt", "w") 
            file.write("%s\n" % (experiment_index+1))
            file.close()
   
    print "experiment_index", experiment_index
    sys.stdout.flush()
    if n_fixed is not None:
        n = n_fixed
    else:
        n = np.random.randint(5, 50)
    time_range = 200
    if uniform_sampling:
        t = np.random.uniform(0.0, time_range, n)
    else:
        num_seasons = 5
        season_length = float(time_range)/num_seasons
        t = season_length*np.random.randint(num_seasons, size = n) + np.random.uniform(0.0, season_length/2, n)
    
    var = 1.0
    if sig_var_fixed is not None:
        sig_var = sig_var_fixed
    else:
        sig_var = np.random.uniform(0.01, 0.99)
    noise_var = var - sig_var
    t = np.sort(t)
    t -= np.mean(t)
    
    #p = time_range/12.54321#
    p = np.random.uniform(time_range/20, time_range/5)
    freq = 1.0/p
    mean = 0.0
    
    if cov_type == "periodic":
        length_scale = 1e10*p
        k = mcu.calc_cov_p(t, freq, sig_var) + np.diag(np.ones(n) * noise_var)
    else:
        length_scale = np.random.uniform(p/2, 10.0*p)
        k = mcu.calc_cov_qp(t, freq, length_scale, sig_var) + np.diag(np.ones(n) * noise_var)
        
    l = la.cholesky(k)
    s = np.random.normal(0.0, 1.0, n)
    
    y = np.repeat(mean, n) + np.dot(l, s)
    y += mean
    kalman_utils = ku.kalman_utils(t, y, num_iterations=3)
    #kalman_utils.add_component("periodic", [np.zeros(1), np.zeros(1), np.zeros(1)], {"j_max":2})
    kalman_utils.add_component("quasiperiodic", [np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)], {"j_max":1})
    kalman_utils.add_component("white_noise", [np.zeros(1)])
    
    num_freqs = 100
    num_cohs = 20
    
    if cov_type == "periodic":
        num_cohs = 1
        
    coh_ind = 0
    
    t_cohs = np.linspace(0.1, length_scale*2, num_cohs)
    
    max_loglik_full_gp = None
    max_coh_full_gp = 0
    max_freq_full_gp = 0
    
    max_fg = None
    max_coh_fg = 0
    max_freq_fg = 0
    
    max_d2 = None
    max_coh_d2 = 0
    max_freq_d2 = 0
    
    max_loglik_kalman = None
    max_coh_kalman = 0
    max_freq_kalman = 0
    
    o = np.ones(len(t))
    y2 = y * y
    
    for t_coh in t_cohs:
        print coh_ind
        sys.stdout.flush()
        f_ind = 0
        fs = np.linspace(0.01, 2.0*freq, num_freqs)
        for f in fs:
            if cov_type == "periodic":
                k = mcu.calc_cov_p(t, f, sig_var)
                sel_fn = mcu.calc_sel_fn_p(t, f, sig_var)
            else:
                k = mcu.calc_cov_qp(t, f, t_coh, sig_var)
                sel_fn = mcu.calc_sel_fn_qp(t, f, t_coh, sig_var)
                
            k += np.diag(np.ones(n) * noise_var)
            g_1, g_2, g_log = mcu.calc_g(t, sig_var, noise_var, k)
            fg = np.dot(o, np.dot(g_1, y2)) - np.dot(y, np.dot(g_2, y)) +0.5*g_log
            d2 = mcu.calc_d2(t, y, sel_fn, normalize=True)
            gpr_gp = GPR_QP.GPR_QP(sig_var=sig_var, length_scale=t_coh, freq=f, noise_var=noise_var, rot_freq=0.0, rot_amplitude=0.0, trend_var=0.0, c=0.0)
            t_test = np.linspace(min(t), max(t), 2)
            gpr_gp.init(t, y)
            (_, _, loglik) = gpr_gp.fit(t_test)
            
            loglik_kalman = mcu.calc_kalman(kalman_utils, t, y, sig_var, noise_var, t_coh, f, plot=False, coh_ind=coh_ind, f_ind=f_ind)
            
            if max_loglik_full_gp is None or loglik > max_loglik_full_gp:
                max_loglik_full_gp = loglik
                max_coh_full_gp = t_coh
                max_freq_full_gp = f
            if max_fg is None or -fg > max_fg:
                max_fg = -fg
                max_coh_fg = t_coh
                max_freq_fg = f
            if max_d2 is None or -d2 > max_d2:
                max_d2 = -d2
                max_coh_d2 = t_coh
                max_freq_d2 = f
            if max_loglik_kalman is None or loglik_kalman > max_loglik_kalman:
                max_loglik_kalman = loglik_kalman
                max_coh_kalman = t_coh
                max_freq_kalman = f
            
            f_ind += 1
    
        coh_ind += 1
    
    index = experiment_index
    with FileLock("GPRLock"):
        output = open("results.txt", "a") 
        output.write("%s %s %s %s %s %s %s %s %s %s %s %s %s\n" % (index, n, sig_var, freq, max_freq_full_gp, max_freq_fg, max_freq_d2, max_freq_kalman, length_scale, max_coh_full_gp, max_coh_fg, max_coh_d2, max_coh_kalman))
        output.close()
