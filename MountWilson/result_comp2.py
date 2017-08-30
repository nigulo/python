# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:13:12 2017

@author: olspern1
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

data = np.genfromtxt("Baliunas.csv", dtype=None, delimiter=';')
baliunas_cycles = dict()
for [star, cyc1, cyc2] in data:
    cycles = list()
    if not np.isnan(cyc1):
        cycles.append(cyc1)
    if not np.isnan(cyc2):
        cycles.append(cyc2)
    baliunas_cycles[star] = np.asarray(cycles)

time_ranges_data = np.genfromtxt("time_ranges.dat", usecols=(0,1), dtype=None)
time_ranges = dict()
for [star, time_range] in time_ranges_data:
    if star == 'SUN':
        star = 'Sun'
    time_ranges[star] = time_range

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
    if not np.isnan(cyc) and cyc < time_ranges[star] / 1.5:
        cycles.append(cyc)
        cycles.append(std/f/f)
        cycles.append(bic)
    all_cycles.append(np.asarray(cycles))

data = pd.read_csv("GP_quasiperiodic/processed_with_cycles.txt", names=['star', 'validity', 'cyc', 'sigma', 'bic'], header=0, dtype=None, sep='\s+', engine='python').as_matrix()
gp_cycles = dict()
#for [star, count, count_used, validity, cyc, std, normality, bic, bic_diff] in data:
for [star, validity, cyc, std, bic] in data:
    if star == 'SUN':
        star = 'Sun'
    if not gp_cycles.has_key(star):
        gp_cycles[star] = list()
    all_cycles = gp_cycles[star]
    cycles = list()
    if not np.isnan(cyc) and cyc < time_ranges[star] / 1.5:
        cycles.append(cyc)
        cycles.append(std)
        cycles.append(bic)
    all_cycles.append(np.asarray(cycles))


keys = np.asarray(time_ranges.keys())
keys = np.sort(keys)

for star in keys:
    output = "HD" + star + " & "
    if bglst_cycles.has_key(star):
        i = 0
        for cycles in bglst_cycles[star]:
            if len(cycles) > 0:
                output += " " + str(round(cycles[0],2)) + " $\pm$ " + str(round(cycles[1],2)) + " (" + str(round(cycles[2],1)) + ")"
                if i < np.shape(bglst_cycles[star])[0] - 1:        
                    output += ", "
            i += 1        
    output += " & "

    if gp_cycles.has_key(star):
        i = 0
        for cycles in gp_cycles[star]:
            if len(cycles) > 0:
                output += " " + str(round(cycles[0],2)) + " $\pm$ " + str(round(cycles[1],2)) + " (" + str(round(cycles[2],1)) + ")"
                if i < np.shape(gp_cycles[star])[0] - 1:        
                    output += ", "
            i += 1        
    output += " & "

    if goet_cycles.has_key(star):
        for cycle in goet_cycles[star]:
            output += " " + str(round(cycle,2))
    else:
        output += "NA"
    output += " & "

    if olah_cycles.has_key(star):
        for cycle in olah_cycles[star]:
            output += " " + str(round(cycle,2))
    else:
        output += "NA"
    output += " & "

    if baliunas_cycles.has_key(star):
        for cycle in baliunas_cycles[star]:
            output += " " + str(round(cycle,2))
    else:
        output += "NA"
    output += " \\\\ "

    print output    
