# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 17:05:33 2017

@author: nigul
"""

import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import scipy.stats
import os
import os.path
import mw_utils
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import GPR_QP
from astropy.stats import LombScargle

num_iters = 300
n_eff_col = 9

do_fits = True
calc_residues = False
save_results = True
calc_spread = True

prefix = ""

data_dir = "../GP_input/"

try:
    from itertools import izip_longest  # added in Py 2.6
except ImportError:
    from itertools import zip_longest as izip_longest  # name change in Py 3.x

try:
    from itertools import accumulate  # added in Py 3.2
except ImportError:
    def accumulate(iterable):
        'Return running totals (simplified version).'
        total = next(iterable)
        yield total
        for value in iterable:
            total += value
            yield total

def make_parser(fieldwidths):
    cuts = tuple(cut for cut in accumulate(abs(fw) for fw in fieldwidths))
    pads = tuple(fw < 0 for fw in fieldwidths) # bool values for padding fields
    flds = tuple(izip_longest(pads, (0,)+cuts, cuts))[:-1]  # ignore final one
    parse = lambda line: tuple(line[i:j] for pad, i, j in flds if not pad)
    # optional informational function attributes
    parse.size = sum(abs(fw) for fw in fieldwidths)
    parse.fmtstring = ' '.join('{}{}'.format(abs(fw), 'x' if fw < 0 else 's')
                                                for fw in fieldwidths)
    return parse


#fieldwidths_rot = (13, 7, 8, 7, 7, 7, 7, 7, 7, 7, 7)  # negative widths represent ignored padding fields
#parse_rot = make_parser(fieldwidths_rot)

fieldwidths = (16, 7, 8, 7, 7, 7, 7, 7, 7, 7, 7)  # negative widths represent ignored padding fields
parse = make_parser(fieldwidths)

rot_periods = mw_utils.load_rot_periods("../")

if save_results:
    output_cycles = open(prefix+"processed_with_cycles.txt", "w")

#data = pd.read_csv(prefix+"results/"+"results.txt", names=['star', 'index', 'validity', 'cyc', 'cyc_se', 'cyc_std', 'length_scale', 'length_scale_se', 'length_scale_std', 'trend_var', 'trend_var_se', 'trend_var_std', 'm', 'sig_var', 'fvu', 'delta_bic', 'length_scale2', 'length_scale2_se', 'length_scale2_std'], dtype=None, sep='\s+', engine='python').as_matrix()
data = pd.read_csv(prefix+"results/"+"results.txt", names=['star', 'index', 'validity', 'cyc', 'cyc_se', 'cyc_std', 'length_scale', 'length_scale_se', 'length_scale_std', 'trend_var', 'trend_var_se', 'trend_var_std', 'm', 'sig_var', 'fvu', 'delta_bic'], dtype=None, sep='\s+', engine='python').as_matrix()

#data = np.genfromtxt(file, dtype=None, skip_header=1)
#for [star, index, validity, cyc, cyc_se, cyc_std, length_scale, length_scale_se, length_scale_std, trend_var, trend_var_se, trend_var_std, m, sig_var, fvu, delta_bic, length_scale2, length_scale2_se, length_scale2_std] in data:
for [star, index, validity, cyc, cyc_se, cyc_std, length_scale, length_scale_se, length_scale_std, trend_var, trend_var_se, trend_var_std, m, sig_var, fvu, delta_bic] in data:

    #if star != "201091":
    #    continue
    #if delta_bic < 6:
    #    continue
    star = str(star)
    file_name = prefix + "results/"+star + "_" + str(index) + "_results.txt"
    if not os.path.isfile(file_name) and index == 0:
        file_name = prefix + "results/"+ star + "_results.txt"
    print "Loading " + star + " " + str(index)
    n_eff = -1
    sig_var = -1
    m = -1
    with open(file_name, "r") as ins:
        i = 0
        for line in ins:
            if i < 5:
                i += 1
                continue
            #if rot_periods.has_key(star):
            #    fields = parse_rot(line)
            #else:
            fields = parse(line)
            var = fields[0].strip()
            var = var.replace(' ', '')
            if var == "freq":
                n_eff = float(fields[n_eff_col].strip())
            if var == "sig_var":
                sig_var = float(fields[1].strip())
            if var == "m":
                m = float(fields[1].strip())
            
    assert(n_eff >= 0)
    assert(sig_var >= 0)
    #assert(m >= 0)
    print n_eff
    if n_eff >= 5:#float(num_iters) / 10:
        dat = np.loadtxt(data_dir+star + ".dat", usecols=(0,1), skiprows=0)
        
        offset = 1979.3452
        
        t_orig = dat[:,0]
        y = dat[:,1]
        
        t = t_orig/365.25
        t += offset
        t_mean = np.mean(t)
        min_t = min(t)
        max_t = max(t)
        t -= t_mean
        if do_fits and not os.path.isfile("fits/" + prefix + star + '.pdf'):
            print cyc, length_scale, sig_var, trend_var, m
    
            ###################################################################
    
    
            noise_var = mw_utils.get_seasonal_noise_var(t, y)
            t_ds, y_ds, noise_var_ds = mw_utils.downsample(t, y, noise_var, 10.0/365.25)
    
            # Full fit
            #gpr_gp = GPR_QP.GPR_QP(sig_var=sig_var, length_scale=length_scale, freq=1.0/cyc, noise_var=noise_var, rot_freq=0, rot_amplitude=0, trend_var=trend_var, c=0.0, length_scale2=length_scale2)
            gpr_gp = GPR_QP.GPR_QP(sig_var=sig_var, length_scale=length_scale, freq=1.0/cyc, noise_var=noise_var_ds, rot_freq=0, rot_amplitude=0, trend_var=trend_var, c=0.0, length_scale2=0.0)
            t_test = np.linspace(min(t), max(t), 200)
            gpr_gp.init(t_ds, y_ds-m)
            (f_mean, pred_var, loglik) = gpr_gp.fit(t_test)
            pred_var = pred_var# + mw_utils.get_test_point_noise_var(t, y, t_test, sliding=True)
            f_mean += m
            
            t_test += t_mean
            
            fig, ax1 = plt.subplots(nrows=1, ncols=1)
            #fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
            fig.set_size_inches(9, 5)
            ax1.text(0.95, 0.9,'(c)', horizontalalignment='center', transform=ax1.transAxes, fontsize=15)
            ax1.set_xlim([min_t, max_t])
        
            ax1.plot(t + t_mean, y, 'k+', lw=0.5)
            #ax2.plot(t, y_wo_rot, 'r+')
            ax1.plot(t_test, f_mean, 'r-', lw=2)
            ax1.fill_between(t_test, f_mean + 3.0 * np.sqrt(pred_var), f_mean - 3.0 * np.sqrt(pred_var), alpha=0.8, facecolor='lightsalmon', interpolate=True)
        
            ax1.set_ylabel(r'S-index', fontsize=15)
            ax1.set_xlabel(r'$t$ [yr]', fontsize=15)
      
            fig.savefig("fits/" + prefix + star + '.pdf')
            plt.close()
            
            if calc_residues:
                (f_t, _, _) = gpr_gp.fit(t)
                np.savetxt("residues/" + star + ".dat", np.column_stack((t + t_mean, y - f_t - m)), fmt='%f')
            spread = 0
            if calc_spread:
                min_freq = 0.001
                max_freq = 0.5
                n_out = 1000
                freqs = np.linspace(min_freq, max_freq, n_out)
                power = LombScargle(t_test, f_mean, nterms=1).power(freqs, normalization='psd')#/np.var(y)
                
                normalized_power = power - min(power)
                normalized_power /= sum(normalized_power)
                spread = np.sqrt(sum((freqs-1.0/cyc)**2 * normalized_power))
                
                spread -= 1.0/(max(t) - min(t))
                
        if save_results and delta_bic >= 6.0 and cyc < (max_t - min_t) / 1.5 and cyc > 2.0 and cyc_std < cyc/4:
            output_cycles.write(star + " " + str(validity) + " " + str(cyc) + " " + str(cyc_std) + " " + str(delta_bic) + " " + str(spread*cyc**2) +  "\n")
            output_cycles.flush()
                
            
    else:
        print "Omitting " + star + " " + str(index) + " due to too low n_eff"

if save_results:
    output_cycles.close()

