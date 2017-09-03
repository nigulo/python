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
for [star, cyc1, cyc2] in data:
    cycles = list()
    if len(cyc1) > 0:
    #if not np.isnan(cyc1):
        cycles.append(cyc1)
    if len(cyc2) > 0:
    #if not np.isnan(cyc2):
        cycles.append(cyc2)
    baliunas_cycles[star] = np.asarray(cycles)

baliunas_stars = np.genfromtxt("baliunas_stars.txt", dtype=None)

time_ranges_data = np.genfromtxt("time_ranges.dat", usecols=(0,1), dtype=None)
time_ranges = dict()
for [star, time_range] in time_ranges_data:
    if star == 'SUN':
        star = 'Sun'
    time_ranges[star] = time_range
    found = False
    for baliunas_star in baliunas_stars:
        if baliunas_star == star:
            found = True
            break
    if found:
        if not baliunas_cycles.has_key(star):
            print star
        assert baliunas_cycles.has_key(star)

data = pd.read_csv("BGLST_BIC_6/results.txt", names=['star', 'f', 'sigma', 'normality', 'bic'], header=0, dtype=None, sep='\s+', engine='python').as_matrix()
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

data = []#pd.read_csv("GP_periodic/processed_with_cycles.txt", names=['star', 'validity', 'cyc', 'sigma', 'bic'], header=0, dtype=None, sep='\s+', engine='python').as_matrix()
gp_p_cycles = dict()
#for [star, count, count_used, validity, cyc, std, normality, bic, bic_diff] in data:
for [star, validity, cyc, std, bic] in data:
    if star == 'SUN':
        star = 'Sun'
    if not gp_p_cycles.has_key(star):
        gp_p_cycles[star] = list()
    all_cycles = gp_p_cycles[star]
    cycles = list()
    if not np.isnan(cyc) and cyc < time_ranges[star] / 1.5:
        cycles.append(cyc)
        cycles.append(std)
        cycles.append(bic)
    all_cycles.append(np.asarray(cycles))

data = pd.read_csv("GP_quasiperiodic/processed_with_cycles.txt", names=['star', 'validity', 'cyc', 'sigma', 'bic'], header=0, dtype=None, sep='\s+', engine='python').as_matrix()
gp_qp_cycles = dict()
#for [star, count, count_used, validity, cyc, std, normality, bic, bic_diff] in data:
for [star, validity, cyc, std, bic] in data:
    if star == 'SUN':
        star = 'Sun'
    if not gp_qp_cycles.has_key(star):
        gp_qp_cycles[star] = list()
    all_cycles = gp_qp_cycles[star]
    cycles = list()
    if not np.isnan(cyc) and cyc < time_ranges[star] / 1.5:
        cycles.append(cyc)
        cycles.append(std)
        cycles.append(bic)
    all_cycles.append(np.asarray(cycles))


keys = np.asarray(time_ranges.keys())
keys = np.sort(keys)

spec_types = mw_utils.load_spec_types()

for star in keys:
    output = "HD" + star + " & " + str(round(time_ranges[star],1)) + " & " + spec_types[star] + " & "
    if bglst_cycles.has_key(star):
        i = 0
        for cycles in bglst_cycles[star]:
            if len(cycles) > 0:
                output += " " + str(round(cycles[0],2)) + " $\pm$ " + str(round(cycles[1],2)) + " (" + str(round(cycles[2],1)) + ")"
                if i < np.shape(bglst_cycles[star])[0] - 1:        
                    output += ", "
            i += 1        
    output += " & "

    if gp_p_cycles.has_key(star):
        i = 0
        for cycles in gp_p_cycles[star]:
            if len(cycles) > 0:
                output += " " + str(round(cycles[0],2)) + " $\pm$ " + str(round(cycles[1],2)) + " (" + str(round(cycles[2],1)) + ")"
                if i < np.shape(gp_p_cycles[star])[0] - 1:        
                    output += ", "
            i += 1        
    output += " & "

    if gp_qp_cycles.has_key(star):
        i = 0
        for cycles in gp_qp_cycles[star]:
            if len(cycles) > 0:
                output += " " + str(round(cycles[0],2)) + " $\pm$ " + str(round(cycles[1],2)) + " (" + str(round(cycles[2],1)) + ")"
                if i < np.shape(gp_qp_cycles[star])[0] - 1:        
                    output += ", "
            i += 1        
    output += " & "

    ##### Commenting Goettingen out for now
    #if goet_cycles.has_key(star):
    #    for cycle in goet_cycles[star]:
    #        output += " " + str(round(cycle,2))
    #else:
    #    output += "NA"
    #output += " & "

    if olah_cycles.has_key(star):
        for cycle in olah_cycles[star]:
            output += " " + str(round(cycle,2))
    else:
        output += "NA"
    output += " & "

    if baliunas_cycles.has_key(star):
        for cycle in baliunas_cycles[star]:
            output += " " + cycle#str(round(cycle,2))
    else:
        output += "NA"
    output += " \\\\ "

    print output    
