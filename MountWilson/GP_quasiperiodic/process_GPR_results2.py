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
import numpy.linalg as la

num_iters = 300
freq_row = 2
n_eff_col = 9

peak_no = 1
use_residue_as_data = True
use_residue_from = 0

peak_no_str = ""
prefix = ""
if peak_no > 0:
    peak_no_str = str(peak_no) + "/"

data_dir = "../GP_input"
use_residue_from_str = ""
if use_residue_as_data:
    if use_residue_from > 0:
        use_residue_from_str = str(use_residue_from) + "/"
    data_dir = "residues/" + use_residue_from_str
    prefix = str(use_residue_from) + "_" + str(peak_no) + "/"


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

output_cycles = open("processed/"+prefix+peak_no_str+"processed_with_cycles.txt", "w")

data = pd.read_csv(prefix+"results/"+peak_no_str+"results.txt", names=['star', 'index', 'validity', 'cyc', 'cyc_se', 'cyc_std', 'length_scale', 'length_scale_se', 'length_scale_std', 'trend_var', 'trend_var_se', 'trend_var_std', 'rot_amplitude', 'fvu', 'delta_bic'], dtype=None, sep='\s+', engine='python').as_matrix()

#data = np.genfromtxt(file, dtype=None, skip_header=1)
for [star, index, validity, cyc, cyc_se, cyc_std, length_scale, length_scale_se, length_scale_std, trend_var, trend_var_se, trend_var_std, rot_amplitude, fvu, delta_bic] in data:

    #if delta_bic < 6:# and star != "101501":
    #    continue
    star = str(star)
    file_name = "results/"+peak_no_str + star + "_" + str(index) + "_results.txt"
    if not os.path.isfile(file_name) and index == 0:
        file_name = "results/"+ peak_no_str + star + "_results.txt"
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
        print cyc, length_scale, sig_var, trend_var, m
    #details = np.genfromtxt(file_name, usecols=(n_eff_col), skip_header=5, skip_footer=5)
    #freq_row = -1
    #for row in np.arange(0, np.shape(details)[0]):
    #    if details[row, 0] == "freq":
    #        freq_row = row
    #        break
    #assert(freq_row >= 0)
    #if (details[freq_row]) >= float(num_iters) / 10:
        #cycles.append((cyc*365.25, std_2/2*365.25)) # one sigma
    
        if not os.path.isfile("residues/" + prefix+ peak_no_str + star + ".dat"):
            dat = np.loadtxt(data_dir+"/"+star + ".dat", usecols=(0,1), skiprows=0)
            
            offset = 1979.3452
            
            t_orig = dat[:,0]
            y = dat[:,1]
            
            t = t_orig/365.25
            t += offset
            t -= np.mean(t)

            noise_var = mw_utils.get_seasonal_noise_var(t, y)

            ###################################################################
            # Cross-validation


            #gpr_gp = GPR_QP.GPR_QP(sig_var=sig_var, length_scale=length_scale, freq=1.0/cyc, noise_var=noise_var, rot_freq=0, rot_amplitude=0, trend_var=trend_var, c=0.0)
            #gpr_gp_null = GPR_QP.GPR_QP(sig_var=0.0, length_scale=length_scale, freq=0.0, noise_var=noise_var, rot_freq=0.0, rot_amplitude=0.0, trend_var=trend_var, c=0.0)
            #gpr_gp.init(t, y-m)
            #gpr_gp_null.init(t, y-m)
            #seasonal_means = mw_utils.get_seasonal_means(t, y-m)
            #seasonal_noise = mw_utils.get_seasonal_noise_var(t, y, per_point=False)
            #l_loo = 0.0
            #l_loo_null = 0.0
            #season_index = 0
            #print seasonal_means[:,0]
            #print seasonal_means[:,1]
            #(_, _, loglik_test) = gpr_gp.cv(seasonal_means[:,0], seasonal_means[:,1], seasonal_noise)
            #(_, _, loglik_test_null) = gpr_gp_null.cv(seasonal_means[:,0], seasonal_means[:,1], seasonal_noise)
            
            
            seasonal_means = mw_utils.get_seasonal_means(t, y-m)
            seasons = mw_utils.get_seasons(zip(t, y), 1.0, True)
            seasonal_noise = mw_utils.get_seasonal_noise_var(t, y, per_point=False)
            l_loo = 0.0
            l_loo_null = 0.0
            dat = np.column_stack((t, y))
            season_index = 0
            for season in seasons:
                season_start = min(season[:,0])
                season_end = max(season[:,0])
                print "cv for season: ", season_index, season_start, season_end
                dat_test = seasonal_means[season_index]
                if season_index == len(seasons) - 1:
                    indices = np.where(dat[:,0] < season_start)[0]
                    dat_train = dat[indices,:]
                    noise_train = noise_var[indices]
                    #dat_test = dat[np.where(dat[:,0] >= season_start)[0],:]
                else:
                    dat_season = dat[np.where(dat[:,0] < season_end)[0],:]
                    indices_after = np.where(dat[:,0] >= season_end)[0]
                    dat_after = dat[indices_after,:]
                    indices_before = np.where(dat_season[:,0] < season_start)[0]
                    dat_before = dat_season[indices_before,:]
                    #dat_test = seasonal_means[season_index]# dat_season[np.where(dat_season[:,0] >= season_start)[0],:]
                    dat_train = np.concatenate((dat_before, dat_after), axis=0)
                    noise_before = noise_var[indices_before]
                    noise_after = noise_var[indices_after]
                    noise_train = np.concatenate((noise_before, noise_after), axis=0)
                #test_mat = np.array([[1.16490151e-08, 1.16493677e-08], [1.16493677e-08, 1.16497061e-08]])
                #test_mat = np.array([[1.16490151e-08, 1.16e-08], [1.16e-08, 1.16497061e-08]])
                #test_mat *= 1e8
                #print test_mat
                #L_test_covar = la.cholesky(test_mat)
                
                #print indices_before, indices_after, noise_train
                gpr_gp = GPR_QP.GPR_QP(sig_var=sig_var, length_scale=length_scale, freq=1.0/cyc, noise_var=noise_train, rot_freq=0, rot_amplitude=0, trend_var=trend_var, c=0.0)
                gpr_gp_null = GPR_QP.GPR_QP(sig_var=0.0, length_scale=length_scale, freq=0.0, noise_var=noise_train, rot_freq=0.0, rot_amplitude=0.0, trend_var=trend_var, c=0.0)
                gpr_gp.init(dat_train[:,0], dat_train[:,1]-m)
                (_, _, loglik_test) = gpr_gp.cv(dat_test[0], dat_test[1], seasonal_noise[season_index])
                l_loo += loglik_test
                gpr_gp_null.init(dat_train[:,0], dat_train[:,1]-m)
                (_, _, loglik_test_null) = gpr_gp_null.cv(dat_test[0], dat_test[1], seasonal_noise[season_index])
                l_loo_null += loglik_test_null
                season_index += 1

            print "l_loo, l_loo_null", l_loo, l_loo_null
            ###################################################################

            # Full fit
            gpr_gp = GPR_QP.GPR_QP(sig_var=sig_var, length_scale=length_scale, freq=1.0/cyc, noise_var=noise_var, rot_freq=0, rot_amplitude=0, trend_var=trend_var, c=0.0)
            t_test = np.linspace(min(t), max(t), 200)
            gpr_gp.init(t, y-m)
            (f_mean, pred_var, loglik) = gpr_gp.fit(t_test)
            (f_t, _, _) = gpr_gp.fit(t)
            f_mean += m
            residue = y - (f_t + m)
            
            dat = np.column_stack((t_orig, residue))
            np.savetxt("residues/" + prefix + peak_no_str + star + ".dat", dat, fmt='%f')
            fig, ax1 = plt.subplots(nrows=1, ncols=1)
            #fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
            fig.set_size_inches(9, 5)
            ax1.text(0.95, 0.9,'(b)', horizontalalignment='center', transform=ax1.transAxes, fontsize=15)
            ax1.set_xlim([min(t), max(t)])
        
            ax1.plot(t, y, 'k+', lw=0.5)
            #ax2.plot(t, y_wo_rot, 'r+')
            ax1.plot(t_test, f_mean, 'r-', lw=2)
            ax1.fill_between(t_test, f_mean + 3.0 * np.sqrt(pred_var), f_mean - 3.0 * np.sqrt(pred_var), alpha=0.1, facecolor='lightsalmon', interpolate=True)
        
            ax1.set_ylabel(r'S-index', fontsize=15)
            ax1.set_xlabel(r'$t$ [yr]', fontsize=15)
            #ax2.plot(t, residue, 'b+')
      
            fig.savefig("fits/" + prefix + peak_no_str+star + '.pdf')
            plt.close()
    
            output_cycles.write(star + " " + str(validity) + " " + str(cyc) + " " + str(cyc_std) + " " + str(l_loo - l_loo_null) + "\n")    

    else:
        print "Omitting " + star + " " + str(index) + " due to too low n_eff"

output_cycles.close()
