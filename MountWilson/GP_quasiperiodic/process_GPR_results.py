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

num_iters = 300
freq_row = 2
n_eff_col = 9


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

def read_cycles():
    all_cycles = dict()
    counts = dict()
    data = pd.read_csv("results/results.txt", names=['star', 'index', 'validity', 'cyc', 'cyc_se', 'cyc_std', 'length_scale', 'length_scale_se', 'length_scale_std', 'trend_var', 'trend_var_se', 'trend_var_std', 'rot_amplitude', 'fvu', 'delta_bic'], dtype=None, sep='\s+', engine='python').as_matrix()
    
    #data = np.genfromtxt(file, dtype=None, skip_header=1)
    for [star, index, validity, cyc, cyc_se, cyc_std, length_scale, length_scale_se, length_scale_std, trend_var, trend_var_se, trend_var_std, rot_amplitude, fvu, delta_bic] in data:

        if delta_bic < 6:
            continue
        print type(index)
        file_name = "results/" + star + "_" + str(index) + "_results.txt"
        if not os.path.isfile(file_name) and index == 0:
            file_name = "results/" + star + "_results.txt"
        print "Loading " + star + " " + str(index)
        n_eff = -1
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
                    break
        assert(n_eff >= 0)
        if not counts.has_key(star):
            counts[star] = 0
        counts[star] = counts[star] + 1
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
            if not all_cycles.has_key(star):
                all_cycles[star] = []
            cycles = all_cycles[star]
            #cycles.append((cyc*365.25, std_2/2*365.25)) # one sigma
            cycles.append((cyc, cyc_std, validity, delta_bic))
            all_cycles[star] = cycles
        else:
            print "Omitting " + star + " " + str(index) + " due to too low n_eff"
    return all_cycles, counts


cycles, counts = read_cycles()

keys = np.asarray(cycles.keys())
keys = np.sort(keys)

output = open("processed_results.txt", "w")
output_cycles = open("processed_with_cycles.txt", "w")

for star in counts.keys():
    if not cycles.has_key(star):
        print "Fully omitting " + star

for star in keys:
    star_results = cycles[star]
    count = np.shape(star_results)[0]
    star_cycle_samples = np.array([])
    star_bic_samples = []
    (_, _, validity0, delta_bic) = star_results[0]
    for bs_ind in np.arange(0, 100):
        (cyc, cyc_std, validity, delta_bic) = star_results[np.random.choice(count, size=None, replace=True)]
        if validity == validity0:
            star_cycle_samples = np.concatenate([star_cycle_samples, np.random.normal(cyc, cyc_std, 100)])
            star_bic_samples.append(delta_bic)
    (skewKurt, normality) = scipy.stats.normaltest(star_cycle_samples)
    output.write(star + " " + str(counts[star]) + " " + str(count) + " " + str(validity) + " " + str(np.mean(star_cycle_samples)) + " " + str(np.std(star_cycle_samples)) + " " + str(normality) + " " + str(np.mean(star_bic_samples)) + " " + str(max(star_bic_samples) - min(star_bic_samples)) + "\n")    
    if validity and np.mean(star_bic_samples) >= 2:
        output_cycles.write(star + " " + str(counts[star]) + " " + str(count) + " " + str(validity) + " " + str(np.mean(star_cycle_samples)) + " " + str(np.std(star_cycle_samples)) + " " + str(normality) + " " + str(np.mean(star_bic_samples)) + " " + str(max(star_bic_samples) - min(star_bic_samples)) + "\n")    
        n, bins, patches = plt.hist(star_cycle_samples, 50, normed=1, facecolor='green', alpha=0.75)
        y = mlab.normpdf(bins, np.mean(star_cycle_samples), np.std(star_cycle_samples))
        l = plt.plot(bins, y, 'r--', linewidth=1)
        
        plt.savefig("hists/" + star + "_hist.png")
        plt.close()
        
output.close()
output_cycles.close()
