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
    time_ranges[star] = time_range

data = pd.read_csv("ls_results0.01/results.txt", names=['star', 'cyc1', 'err1', 'n1', 'cyc2', 'err2', 'n2'], header=None, dtype=None, sep=' ').as_matrix()
ls_cycles = dict()
for [star, cyc1, err1, n1, cyc2, err2, n2] in data:
    if not ls_cycles.has_key(star):
        ls_cycles[star] = list()
    all_cycles = ls_cycles[star]
    cycles = list()
    if not np.isnan(cyc1) and cyc1 < time_ranges[star] / 1.5:
        cycles.append(cyc1)
        cycles.append(err1)
    if not np.isnan(cyc2) and cyc1 < time_ranges[star] / 1.5:
        cycles.append(cyc2)
        cycles.append(err2)
    all_cycles.append(np.asarray(cycles))

data = pd.read_csv("ls_results0.05/results.txt", names=['star', 'cyc1', 'err1', 'n1', 'cyc2', 'err2', 'n2'], header=None, dtype=None, sep=' ').as_matrix()
for [star, cyc1, err1, n1, cyc2, err2, n2] in data:
    if not ls_cycles.has_key(star):
        ls_cycles[star] = list()
    all_cycles = ls_cycles[star]
    cycles = list()
    if not np.isnan(cyc1) and cyc1 < time_ranges[star] / 1.5:
        cycles.append(cyc1)
        cycles.append(err1)
    if not np.isnan(cyc2) and cyc1 < time_ranges[star] / 1.5:
        cycles.append(cyc2)
        cycles.append(err2)
    all_cycles.append(np.asarray(cycles))


data = pd.read_csv("d2_results/results.txt", names=np.core.defchararray.add(np.repeat("col", 4 * 20 + 1), + np.arange(0, 4 * 20 + 1).astype(str)), header=None, dtype=None, sep=' ').as_matrix()
for columns in data:
    star = columns[0]
    if not ls_cycles.has_key(star):
        ls_cycles[star] = list()
    other_columns = np.reshape(columns[1:], (len(columns) / 4, 4))
    cycles = dict()
    for [cyc, err, p_value, n] in other_columns:
        if not np.isnan(cyc) and cyc < time_ranges[star] / 1.5:
            if not cycles.has_key(p_value):
                cycles[p_value] = list()
            cycles_with_p_value = cycles[p_value]
            cycles_with_p_value.append(cyc)
            cycles_with_p_value.append(err)
    all_cycles = ls_cycles[star]
    found = False
    for p_value in cycles.keys():
        if p_value == 0.01:
            found = True
            all_cycles.append(cycles[p_value])
    if not found:
        all_cycles.append(list())
        
    found = False
    for p_value in cycles.keys():
        if p_value == 0.05:
            found = True
            all_cycles.append(cycles[p_value])
    if not found:
        all_cycles.append(list())

for star in ls_cycles.keys():
    output = star + ";"
    for cycles in ls_cycles[star]:
        for cycle in cycles:
            output += " " + str(round(cycle,2))
        output += ";"
    
    if goet_cycles.has_key(star):
        for cycle in goet_cycles[star]:
            output += " " + str(round(cycle,2))
    else:
        output += "NA"
    output += ";"

    if olah_cycles.has_key(star):
        for cycle in olah_cycles[star]:
            output += " " + str(round(cycle,2))
    else:
        output += "NA"
    output += ";"

    if baliunas_cycles.has_key(star):
        for cycle in baliunas_cycles[star]:
            output += " " + str(round(cycle,2))
    else:
        output += "NA"

    print output    
