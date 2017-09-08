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
import mw_utils
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import GPR_QP

num_iters = 300
freq_row = 2
n_eff_col = 9

data_dir = "../GP_input"
if data_dir == "../cleaned":
    skiprows = 1
else:
    skiprows = 0


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

output_cycles = open("processed_with_cycles.txt", "w")

data = pd.read_csv("results/results.txt", names=['star', 'index', 'validity', 'cyc', 'cyc_se', 'cyc_std', 'length_scale', 'length_scale_se', 'length_scale_std', 'trend_var', 'trend_var_se', 'trend_var_std', 'rot_amplitude', 'fvu', 'delta_bic'], dtype=None, sep='\s+', engine='python').as_matrix()

#data = np.genfromtxt(file, dtype=None, skip_header=1)
for [star, index, validity, cyc, cyc_se, cyc_std, length_scale, length_scale_se, length_scale_std, trend_var, trend_var_se, trend_var_std, rot_amplitude, fvu, delta_bic] in data:

    if delta_bic < 6:
        continue
    file_name = "results/" + star + "_" + str(index) + "_results.txt"
    if not os.path.isfile(file_name) and index == 0:
        file_name = "results/" + star + "_results.txt"
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
    assert(m >= 0)
    print n_eff
    if n_eff >= 5:#float(num_iters) / 10:
    #details = np.genfromtxt(file_name, usecols=(n_eff_col), skip_header=5, skip_footer=5)
    #freq_row = -1
    #for row in np.arange(0, np.shape(details)[0]):
    #    if details[row, 0] == "freq":
    #        freq_row = row
    #        break
    #assert(freq_row >= 0)
    #if (details[freq_row]) >= float(num_iters) / 10:
        #cycles.append((cyc*365.25, std_2/2*365.25)) # one sigma
        output_cycles.write(star + " " + str(validity) + " " + str(cyc) + " " + str(cyc_std) + " " + str(delta_bic) + "\n")    
    
        dat = np.loadtxt(data_dir+"/"+star + ".dat", usecols=(0,1), skiprows=skiprows)
        
        offset = 1979.3452
        
        t_orig = dat[:,0]
        y = dat[:,1]
        
        t = t_orig/365.25
        t += offset
        noise_var = mw_utils.get_seasonal_noise_var(t, y)
        n = len(t)
        
        gpr_gp = GPR_QP.GPR_QP(sig_var=sig_var, length_scale=length_scale, freq=1.0/cyc, noise_var=noise_var, rot_freq=0, rot_amplitude=0, trend_var=trend_var, c=0.0)
        t_test = np.linspace(min(t), max(t), 500)
        (f_mean, pred_var, loglik) = gpr_gp.fit(t, y-m, t_test)
        (f_t, _, _) = gpr_gp.fit(t, y-m, t)
        f_mean += m
        residue = y - (f_t + m)
        
        dat = np.column_stack((t_orig, residue))
        np.savetxt("residues/" + star + ".dat", dat, fmt='%f')
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        fig.set_size_inches(18, 12)
    
        ax1.plot(t, y, 'b+')
        #ax2.plot(t, y_wo_rot, 'r+')
        ax1.plot(t_test, f_mean, 'k-')
        ax1.fill_between(t_test, f_mean + 2.0 * np.sqrt(pred_var), f_mean - 2.0 * np.sqrt(pred_var), alpha=0.1, facecolor='lightgray', interpolate=True)
    
        ax2.plot(t, residue, 'b+')
  
        fig.savefig("fits/"+star + '.png')
        plt.close()
    
    else:
        print "Omitting " + star + " " + str(index) + " due to too low n_eff"

output_cycles.close()
