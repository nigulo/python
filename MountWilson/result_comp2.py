# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:13:12 2017

@author: olspern1
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mw_utils

data = np.genfromtxt("Goettingen.csv", dtype=None, delimiter=';')
goet_cycles = dict()
for [star, per, cyc1, cyc2, cyc3] in data:
    cycles = list()
    if not np.isnan(cyc1):
        cycles.append(cyc1)
    if not np.isnan(cyc2):
        cycles.append(cyc2)
    if not np.isnan(cyc3):
        cycles.append(cyc3)
    goet_cycles[star] = np.asarray(cycles)

data = np.genfromtxt("Olah.csv", dtype=None, delimiter=';')
olah_cycles = dict()
for [star, cyc1, cyc2, cyc3, cyc4, cyc5] in data:
    cycles = list()
    if not np.isnan(cyc1):
        cycles.append(cyc1)
    if not np.isnan(cyc2):
        cycles.append(cyc2)
    if not np.isnan(cyc3):
        cycles.append(cyc3)
    if not np.isnan(cyc4):
        cycles.append(cyc4)
    if not np.isnan(cyc5):
        cycles.append(cyc5)
    olah_cycles[star] = np.asarray(cycles)

data = np.genfromtxt("Baliunas.csv", dtype='str', delimiter=';')
baliunas_cycles = dict()
for [star, cyc1, grade1, cyc2, grade2] in data:
    cycles = list()
    if len(cyc1) > 0:
        cycles.append((cyc1, grade1))
    if len(cyc2) > 0:
        cycles.append((cyc2, grade2))
    baliunas_cycles[star] = cycles

baliunas_stars = np.genfromtxt("baliunas_stars.txt", dtype=None)

time_ranges_data = np.genfromtxt("time_ranges.dat", usecols=(0,1,2), dtype=None)
time_ranges = dict()
end_times = dict()
for [star, time_range, end_time] in time_ranges_data:
    if star == 'SUN':
        star = 'Sun'
    time_ranges[star] = time_range
    end_times[star] = end_time
    found = False
    for baliunas_star in baliunas_stars:
        if baliunas_star == star:
            found = True
            break
    if found:
        if not baliunas_cycles.has_key(star):
            print star
        assert baliunas_cycles.has_key(star)

ms_stars = np.genfromtxt("MS.dat", usecols=(0), dtype=None)

data = pd.read_csv("BGLST_BIC_6/results.txt", names=['star', 'f', 'sigma', 'normality', 'bic'], header=None, dtype=None, sep='\s+', engine='python').as_matrix()
bglst_cycles = dict()
for [star, f, std, normality, bic] in data:
    if star == 'SUN':
        star = 'Sun'
    if not bglst_cycles.has_key(star):
        bglst_cycles[star] = list()
    all_cycles = bglst_cycles[star]
    cycles = list()
    cyc = 1.0/f
    f_samples = np.random.normal(loc=f, scale=std, size=1000)
    cyc_std = np.std(np.ones(len(f_samples))/f_samples)
    if not np.isnan(cyc) and cyc < time_ranges[star] / 1.5:
        cycles.append(cyc)
        cycles.append(cyc_std)
        cycles.append(bic)
    all_cycles.append(np.asarray(cycles))


gp_p_cycles = mw_utils.read_gp_cycles("GP_periodic/results_combined.txt")
gp_qp_cycles = mw_utils.read_gp_cycles("GP_quasiperiodic/results_combined.txt")

keys = np.asarray(time_ranges.keys())
keys = np.sort(keys)

spec_types = mw_utils.load_spec_types()
rot_periods = mw_utils.load_rot_periods()
r_hks = mw_utils.load_r_hk()
rot_periods["Sun"] = (26.09, 0)
r_hks["Sun"] = -4.911

for star in keys:
    
    if not bglst_cycles.has_key(star) and not gp_p_cycles.has_key(star) and not gp_qp_cycles.has_key(star) and (not baliunas_cycles.has_key(star) or len(baliunas_cycles[star]) == 0):
        continue
    
    is_ms = "\\xmark"
    if len(np.where(ms_stars == star.upper())[0] > 0):
        is_ms = "\\cmark"
    if end_times[star] >= 2000:
        hd_str = "{\\bf HD" + star +"}"
    else:
        hd_str = "HD" + star

    rot_period = r"$\dots$"
    rot_period_err = ""
    if rot_periods.has_key(star):
        (rot_period, rot_period_err) = rot_periods[star]
        if rot_period_err > 0:
            rot_period_err = " $\pm$ " + str(rot_period_err)
        else:
            rot_period_err = ""
    r_hk = r"$\dots$"
    if r_hks.has_key(star):
        r_hk = r_hks[star]
        
    output = hd_str + " & " + str(round(time_ranges[star],1)) + " & " + spec_types[star.upper()] + " & " + is_ms +  " & " + str(rot_period) + rot_period_err + " & " + str(r_hk) + " & "
    if bglst_cycles.has_key(star):
        i = 0
        cycle_output = ""
        for cycles in bglst_cycles[star]:
            if len(cycles) > 0:
                cycle_output += " " + str(round(cycles[0],2)) + " $\pm$ " + str(round(cycles[1],2)) + " (" + str(round(cycles[2],1)) + ")"
                if i < np.shape(bglst_cycles[star])[0] - 1:        
                    cycle_output += r"\\ "
            i += 1
        if i > 1:
            cycle_output = r"\begin{tabular}[t]{@{}l@{}}" + cycle_output + "\end{tabular}"
        output += cycle_output
    else:
        output += "--"
    output += " & "

    if gp_p_cycles.has_key(star):
        i = 0
        cycle_output = ""
        for cycles in gp_p_cycles[star]:
            if len(cycles) > 0:
                cycle_output += " " + str(round(cycles[0],2)) + " $\pm$ " + str(round(cycles[1],2)) + " (" + str(round(cycles[2],1)) + ")"
                if i < np.shape(gp_p_cycles[star])[0] - 1:        
                    cycle_output += r"\\ "
            i += 1
        if i > 1:
            cycle_output = r"\begin{tabular}[t]{@{}l@{}}" + cycle_output + "\end{tabular}"
        output += cycle_output
    else:
        output += "--"
    output += " & "

    if gp_qp_cycles.has_key(star):
        i = 0
        cycle_output = ""
        for cycles in gp_qp_cycles[star]:
            if len(cycles) > 0:
                cycle_output += " " + str(round(cycles[0],2)) + " $\pm$ " + str(round(cycles[1],2)) + " (" + str(round(cycles[2],1)) + ")"
                if i < np.shape(gp_qp_cycles[star])[0] - 1:        
                    cycle_output += r"\\ "
            i += 1        
        if i > 1:
            cycle_output = r"\begin{tabular}[t]{@{}l@{}}" + cycle_output + "\end{tabular}"
        output += cycle_output
    else:
        output += "--"
    output += " & "

    ##### Commenting Goettingen out for now
    #if goet_cycles.has_key(star):
    #    for cycle in goet_cycles[star]:
    #        output += " " + str(round(cycle,2))
    #else:
    #    output += "NA"
    #output += " & "

    #if olah_cycles.has_key(star):
    #    if len(olah_cycles[star]) == 0:
    #        output += "--"
    #    else:
    #        for cycle in olah_cycles[star]:
    #            output += " " + str(round(cycle,2))
    #else:
    #    output += "$\dots$"
    #output += " & "

    if baliunas_cycles.has_key(star):
        if len(baliunas_cycles[star]) == 0:
            output += "--"
        else:
            for (cycle, grade) in baliunas_cycles[star]:
                if len(grade) > 0:
                    grade = ' (' + grade + ')'
                output += " " + cycle + grade#str(round(cycle,2))
    else:
        output += r"$\dots$"
    output += " \\\\ "

    print output    
