# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 09:35:24 2017

@author: olspern1
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors as mcolors
import matplotlib.markers as markers
import os
import os.path
from numpy import linalg as LA
from matplotlib.patches import Ellipse
import mw_utils
from bayes_lin_reg import bayes_lin_reg
from GPR_QP import GPR_QP

include_non_ms = False#True
fit_with_baliunas = True

use_secondary_clusters = False
plot_ro = False

axis_label_fs = 15
panel_label_fs = 15
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

fieldwidths = (10, 8, 8, 6, 8, 9, 11, 9, 4)  # negative widths represent ignored padding fields
parse = make_parser(fieldwidths)

ms_stars = np.genfromtxt("MS.dat", usecols=(0), dtype=None)

def star_is_ms(star):
    return len(np.where(ms_stars == star.upper())[0] > 0)

ext_data_for_plot = dict()
data_for_fit = dict()
#run, Omega, P_cyc, E_mag/E_kin
sim_data_joern = [
    ["M0.5", 0.5, 3.1, 0.06],
    ["M1", 1.0, 22.5, 0.16],
    ["M1.5", 1.5, 29.2, 0.17],
    ["M2", 2.0, 29.2, 0.09],
    ["M2.5", 2.5, 7.7, 0.10],
    ["M3", 3.0, 4.6, 0.13],
    ["M4", 4.0, 2.7, 0.21],
    ["M5", 5.0, 2.3, 0.29],
    ["M7", 7.0, 2.7, 0.39],
    ["M10", 10.0, 2.5, 0.52],
    ["M15", 15.0, 3.8, 0.84]
    ]

#run, Omega, E_kin, E_mag P_cyc
sim_data_mariangela = [
    [r'A1', 1, 4.428, 0.876, 3.72],
    [r'A2', 1, 5.055, 0.995, 4.13],
    [r'B', 1.5, 3.263, 0.715, 2.45],
    [r'C1', 1.8, 3.153, 0.504, 3.53],
    [r'C2', 1.8, 3.631, 0.488, 4.37],
    [r'C3', 1.8, 6.572, 0.891, 3.13],
    [r'D', 2.1, 3.181, 0.671, 18.25],
    [r'E', 2.9, 4.189, 0.579, 10.31],
    [r'F1', 4.3, 2.485, 1.363, 6.68],
    [r'F2', 4.3, 2.898, 1.082, 8.05],
    [r'F3', 4.3, 2.7, 0.767, 5.74],
    [r'G$^{a}$', 4.9, 2.748, 0.754, 7.43],
    #[r'G$^{W}$', 4.8, 3.522, 0.975, 2.37],
    [r'H', 7.1, 2.153, 1.049, 27.34],
    [r'H$^{a}$', 7.8, 1.704, 1.449, 7.17],
    [r'I', 9.6, 1.706, 1.361, 7.75],
    #[r'I$^{W}$', 9.6, 1.625, 1.197, 4.44],
    [r'J', 14.5, 0.58, 0.113, 8.25],
    #[r'J$^{W}$', 15.5, 0.786, 0.9, 4.05],
    [r'K1', 21.4, 2.325, 0.426, 1.24],
    [r'K2', 21.4, 1.549, 1.029, 5.1],
    #[r'L$^{a}$', 23.3, 0.708, 1.928, 3.13],
    #[r'L$^{W}$ ', 23.3, 0.415, 1.102, 5.68],
    [r'M', 28.5, 2.053, 0.967, 6.64],
    #[r'M$^{W}$', 31, 0.328, 1.024, 4.1]
    ]

min_jyris_grade = 0.0
max_jyris_grade = 3.0
def get_jyris_grade(grade_string):
    if grade_string == 'excellent':
        return 3.0
    elif grade_string == 'good':
        return 2.0
    elif grade_string == 'fair':
        return 1.0
    else:
        return 0.0

#star, r_hk, p_rot, delta_p_rot, p_cyc_1, grade_cyc_1, p_cyc_2, grade_cyc_2
data_jyri = [
    ['HD1405', -4.217, 1.75622, 0.00062, 8, 'good', 0, ''], 
    ['HD10008', -4.480, 6.78, 0.13, 10.9, 'fair', 0, ''], 
    ['HD26923', -4.618, 11.08, 0.2, 7, 'poor', 0, ''], 
    ['HD29697', -4.036, 3.9651, 0.0059, 7.3, 'poor', 0, ''], 
    ['HD41593', -4.427, 8.135, 0.03, 3.3, 'fair', 0, ''], 
    ['HD43162', -4.425, 7.168, 0.038, 8.1, 'poor', 0, ''], 
    ['HD63433', -4.452, 6.462, 0.04, 2.7, 'fair', 8, 'poor'], 
    ['HD70573', -4.488, 3.3143, 0.0034, 6.9, 'good', 0, ''], 
    ['HD72760', -4.609, 9.57, 0.11, 0, '', 0, ''], 
    ['HD73350', -4.700, 12.14, 0.13, 3.5, 'fair', 0, ''], 
    ['HD82443', -4.286, 5.4244, 0.0043, 4.1, 'fair', 20, 'good'], 
    ['HD82558', -4.079, 1.60435, 0.00042, 17.4, 'excellent', 0, ''], 
    ['HD116956', -4.366, 7.86, 0.016, 2.9, 'fair', 14.7, 'good'], 
    ['HD128987', -4.505, 9.8, 0.12, 5.4, 'fair', 0, ''], 
    ['HD130948', -4.533, 7.849, 0.026, 3.9, 'poor', 0, ''], 
    ['HD135599', -4.462, 5.529, 0.068, 14.6, 'good,', 0, ''], 
    ['HD141272', -4.566, 13.843, 0.084, 6.4, 'poor', 0, ''], 
    ['HD171488', -4.175, 1.3454, 0.0013, 9.5, 'good', 0, ''], 
    ['HD180161', -4.541, 9.91, 0.11, 0, '', 0, ''], 
    ['HD220182', -4.388, 7.678, 0.023, 13.7, 'fair', 0, ''], 
    ['SAO51891', -4.327, 2.4179, 0.0041, 0, '', 0, '']
    ]

for sim, omega, p_cyc, e_mag_div_e_kin in sim_data_joern:
    omega *= 365.25/26.09
    r_hk = e_mag_div_e_kin-4.911-0.16
    r = 0.5
    g = 0.5
    b = 0.5
    #print r_hk, -np.log10(p_cyc * omega), p_cyc, 1.0/omega
    ext_data_for_plot["Joern_" + sim] = [[r_hk, -np.log10(p_cyc * omega), 0.0, 0.0, r, g, b, 1.0, 0, 's', 1/omega, p_cyc, 0, 0, 100, 0, "Warnecke 2018"]]

for sim, omega, e_kin, e_mag, p_cyc in sim_data_mariangela:
    size = 100
    if sim[-4:] == r"{a}$":
        # Make points of high resulution runs bigger
        size = 200
    omega *= 365.25/26.09
    r_hk = e_mag/e_kin-4.911-0.197
    r = 0.5
    g = 0.5
    b = 0.5
    #print sim, r_hk, -np.log10(p_cyc * omega), p_cyc, 1.0/omega
    ext_data_for_plot["Mariangela_" + sim] = [[r_hk, -np.log10(p_cyc * omega), 0.0, 0.0, r, g, b, 1.0, 0, '^', 1/omega, p_cyc, 0, 0, size, 0, "Viviani et al. 2017"]]

for star, r_hk, p_rot, d_p_rot, p_cyc_1, grade1, p_cyc_2, grade2 in data_jyri:
    p_rot /= 365.25
    err = d_p_rot/p_rot/np.log(10)
    cycles = []
    if p_cyc_1 > 0:
        grade1 = get_jyris_grade(grade1)
        if grade1 >= min_jyris_grade:
            c = 0.5*(1.0 - (grade1 - min_jyris_grade)/(max_jyris_grade - min_jyris_grade))
            r = c
            g = 0.75
            b = c  
            cycles.append([r_hk, np.log10(p_rot/p_cyc_1), 0.0, 0.0, r, g, b, 1.0, 0, '*', 1/omega, p_cyc_1, 0, 0, 150, d_p_rot, "Lehtinen et al. 2016"])
    if p_cyc_2 > 0:
        grade2 = get_jyris_grade(grade2)
        if grade2 >= min_jyris_grade:
            c = 0.5*(1.0 - (grade2 - min_jyris_grade)/(max_jyris_grade - min_jyris_grade))
            r = c
            g = 0.75
            b = c        
            cycles.append([r_hk, np.log10(p_rot/p_cyc_2), 0.0, 0.0, r, g, b, 1.0, 0, '*', 1/omega, p_cyc_2, 0, 0, 150, d_p_rot, "Lehtinen et al. 2016"])
    if len(cycles) > 0:
        ext_data_for_plot["Jyri_" + star] = cycles
        data_for_fit[star[2:]] = cycles

def read_gp_cycles(file):
    max_bic = None
    min_bic = None
    all_cycles = dict()
    data = pd.read_csv(file, names=['star', 'validity', 'cyc', 'sigma', 'bic'], header=None, dtype=None, sep='\s+', engine='python').as_matrix()
    
    #data = np.genfromtxt(file, dtype=None, skip_header=1)
    for [star, validity, cyc, std, bic] in data:
        #if star == 'SUNALL':
        #    star = 'SUN'
        #print star, cyc, std_2
        if not np.isnan(cyc):
            if not all_cycles.has_key(star):
                all_cycles[star] = []
            cycles = all_cycles[star]
            log_bic = np.log(bic)
            if max_bic is None or log_bic > max_bic:
                max_bic = log_bic
            if min_bic is None or log_bic < min_bic:
                min_bic = log_bic
                
            if std < cyc:
                cycles.append((cyc*365.25, std*365.25, log_bic)) # three sigma
                all_cycles[star] = cycles
    return min_bic, max_bic, all_cycles

min_baliunas_grade = 1.0
max_baliunas_grade = 3.0
def get_baliunas_grade(grade_string):
    if grade_string == 'E':
        return 3.0
    elif grade_string == 'G':
        return 2.0
    elif grade_string == 'F':
        return 1.0
    else:
        return 0.0
        
data = np.genfromtxt("Baliunas.csv", dtype='str', delimiter=';')
baliunas_cycles = dict()
for [star, cyc1, grade1, cyc2, grade2] in data:
    cycles = list()
    if len(cyc1) > 0:
        try:
            grade = 0
            cyc1 = float(cyc1)
            grade = get_baliunas_grade(grade1)
            if grade >= min_baliunas_grade:
                cycles.append((cyc1*365.25, 0, grade))
        except ValueError:
            print "Omitting cycle"
    if len(cyc2) > 0:
        try:
            cyc2 = float(cyc2)
            grade = get_baliunas_grade(grade2)
            if grade >= min_baliunas_grade:
                cycles.append((cyc2*365.25, 0, grade))
        except ValueError:
            print "Omitting cycle"
    if len(cycles) > 0:
        baliunas_cycles[star] = cycles

def read_FeH_dR(file):
    star_FeH_dR = dict()
    data = pd.read_csv(file, header=0, dtype=None, usecols=['HD/KIC', 'Fe/H', 'd/R'], sep=';', engine='python').as_matrix()
    for [star, FeH, dR] in data:
        star = star.upper()
        star_FeH_dR[star] = [FeH, dR]        
    return star_FeH_dR
    
star_FeH_dR = read_FeH_dR("brandenburg2017table.csv")
rot_periods = mw_utils.load_rot_periods()

###############################################################################
if plot_ro:
    fig1, ((ax111, ax112), (ax121, ax122), (ax131, ax132)) = plt.subplots(nrows=3, ncols=2, sharex=True)
    fig1.set_size_inches(12, 18)
    ax131.set_xlabel(r'${\rm log} \langle R^\prime_{\rm HK}\rangle$', fontsize=axis_label_fs)
    ax132.set_xlabel(r'${\rm log}{\rm Ro}^{-1}$', fontsize=axis_label_fs)
    ax111.text(0.95, 0.9,'(a)', horizontalalignment='center', transform=ax111.transAxes, fontsize=panel_label_fs)
    ax112.text(0.95, 0.9,'(d)', horizontalalignment='center', transform=ax112.transAxes, fontsize=panel_label_fs)
    ax121.text(0.95, 0.9,'(b)', horizontalalignment='center', transform=ax121.transAxes, fontsize=panel_label_fs)
    ax122.text(0.95, 0.9,'(e)', horizontalalignment='center', transform=ax122.transAxes, fontsize=panel_label_fs)
    ax131.text(0.95, 0.9,'(c)', horizontalalignment='center', transform=ax131.transAxes, fontsize=panel_label_fs)
    ax132.text(0.95, 0.9,'(f)', horizontalalignment='center', transform=ax132.transAxes, fontsize=panel_label_fs)
    #ax111.set_aspect('equal', 'datalim')
    #ax112.set_aspect('equal', 'datalim')
    #ax121.set_aspect('equal', 'datalim')
    #ax122.set_aspect('equal', 'datalim')
    #ax131.set_aspect('equal', 'datalim')
    #ax132.set_aspect('equal', 'datalim')
else:
    fig1, ((ax11, ax12), (ax13, ax1a)) = plt.subplots(nrows=2, ncols=2, sharex=False)
    #fig1.set_size_inches(6, 18)
    fig1.set_size_inches(12, 12)
    ax13.set_xlabel(r'${\rm log} \langle R^\prime_{\rm HK}\rangle$', fontsize=axis_label_fs)
    ax11.text(0.95, 0.9,'(a)', horizontalalignment='center', transform=ax11.transAxes, fontsize=panel_label_fs)
    ax12.text(0.95, 0.9,'(b)', horizontalalignment='center', transform=ax12.transAxes, fontsize=panel_label_fs)
    ax13.text(0.95, 0.9,'(c)', horizontalalignment='center', transform=ax13.transAxes, fontsize=panel_label_fs)

    ax1a.text(0.95, 0.9,'(d)', horizontalalignment='center', transform=ax1a.transAxes, fontsize=panel_label_fs)
    #ax11.set_aspect('equal', 'datalim')
    #ax12.set_aspect('equal', 'datalim')
    #ax13.set_aspect('equal', 'datalim')
    
    if include_non_ms:
        ax11.set_ylim(-3.0, -1.0)
        ax11.set_xlim(-5.15, -4.4)

        ax12.set_xlim(-5.15, -4.4)

        ax13.set_xlim(-5.15, -4.4)
        ax13.set_ylim(-3.0, -0.75)

        ax1a.set_xlim(-5.3, -4.3)
        ax1a.set_ylim(-3.0, -1.0)
        ax11.set_ylabel(r'${\rm log}P_{\rm rot}/P_{\rm cyc}$', fontsize=axis_label_fs)
        ax13.set_ylabel(r'${\rm log}P_{\rm rot}/P_{\rm cyc}$', fontsize=axis_label_fs)
    #else:
    #    ax11.set_ylim(-2.9, -1.7)
    #    ax11.set_xlim(-5.1, -4.4)        
    

#fig1a, ax1a = plt.subplots(nrows=1, ncols=1, sharex=False)
#fig1a.set_size_inches(6, 6)
ax1a.set_xlabel(r'${\rm log} \langle R^\prime_{\rm HK}\rangle$', fontsize=axis_label_fs)

fig1b, ax1b = plt.subplots(nrows=1, ncols=1, sharex=False)
fig1b.set_size_inches(8, 6)
ax1b.set_xlabel(r'${\rm log} \langle R^\prime_{\rm HK}\rangle$', fontsize=axis_label_fs)

fig2, (ax21, ax22, ax23) = plt.subplots(nrows=3, ncols=1, sharex=False)
fig2.set_size_inches(6, 18)
ax21.text(0.95, 0.9,'(a)', horizontalalignment='center', transform=ax21.transAxes, fontsize=panel_label_fs)
ax22.text(0.95, 0.9,'(b)', horizontalalignment='center', transform=ax22.transAxes, fontsize=panel_label_fs)
ax23.text(0.95, 0.9,'(c)', horizontalalignment='center', transform=ax23.transAxes, fontsize=panel_label_fs)
#ax23.set_xlabel(r'$P_{\rm rot}$ [d]', fontsize=axis_label_fs)
ax23.set_xlabel(r'$\log P_{\rm rot}$ [$\log$ d]', fontsize=axis_label_fs)

fig3, (ax31, ax32) = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
fig3.set_size_inches(12, 4)
ax31.text(0.95, 0.9,'(a)', horizontalalignment='center', transform=ax31.transAxes, fontsize=panel_label_fs)
ax32.text(0.95, 0.9,'(b)', horizontalalignment='center', transform=ax32.transAxes, fontsize=panel_label_fs)
ax31.set_xlabel(r'$d/R$', fontsize=axis_label_fs)
ax32.set_xlabel(r'[Fe/H] (dex)', fontsize=axis_label_fs)

fig4, (ax41, ax42, ax43) = plt.subplots(nrows=3, ncols=1, sharex=False)
fig4.set_size_inches(6, 18)
ax41.text(0.95, 0.9,'(a)', horizontalalignment='center', transform=ax21.transAxes, fontsize=panel_label_fs)
ax42.text(0.95, 0.9,'(b)', horizontalalignment='center', transform=ax22.transAxes, fontsize=panel_label_fs)
ax43.text(0.95, 0.9,'(c)', horizontalalignment='center', transform=ax23.transAxes, fontsize=panel_label_fs)
ax43.set_xlabel(r'$\log \Omega $' + ' [' + r'$d^{-1}$]', fontsize=axis_label_fs)

def fit_data(data, ax):
    xs = list()
    ys = list()
    for star in data.keys():
        if star == "73350":
            # This was marked as inactive in Jyris data
            print "Omitting", star
        data_star = data[star]
        for [r_hk, y, err1, err2, r, g, b, alpha, ro, sym, p_rot, p_cyc, delta_i, cyc_err, size, p_rot_err, label] in data_star:
            if not label == "Non-active" and not label == "Non-active Baliunas":
                xs.append(r_hk)
                ys.append(y)
                if r_hk < -4.7:
                    print "Active star under inactive branch:", star
            if label == "Active Baliunas":
                color = [1.0, 0.5, 0.0, alpha]
                #color = [1.0, 1.0, 0, 1.0]
                ax.scatter(r_hk, y, marker=markers.MarkerStyle(sym, fillstyle=None), lw=1.5, facecolors='None', color=color, s=size, edgecolors=color)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    xs_fit = np.linspace(min(xs), max(xs), 10)
    max_params = None

    if fit_with_baliunas:
        sig_vars = [5, 6, 7]
        noise_vars = [0.08, 0.09, 0.1]
        length_scales = [0.5, 0.6, 0.7]
    else:
        sig_vars = [5, 6, 7]
        noise_vars = [0.1, 0.11, 0.12]
        length_scales = [1.5, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
    
    for sig_var in sig_vars:
        for noise_var in noise_vars:
            for length_scale in length_scales:
                gpr = GPR_QP(sig_var=sig_var, length_scale=length_scale, freq = 0, noise_var=noise_var, rot_freq=0.0, rot_amplitude=0.0, trend_var=0.0, c=0.0)
                gpr.init(xs, ys)
                (ys_fit, ys_var, log_lik) = gpr.fit(xs_fit)
                if max_params is None or log_lik > max_params[0]:
                    max_params = (log_lik, sig_var, noise_var, length_scale)
    print "Best params:", max_params
    (log_lik, sig_var, noise_var, length_scale) = max_params
    gpr = GPR_QP(sig_var=sig_var, length_scale=length_scale, freq = 0, noise_var=noise_var, rot_freq=0.0, rot_amplitude=0.0, trend_var=0.0, c=0.0)
    gpr.init(xs, ys)
    xs_fit = np.linspace(-5.1, -4.0, 500)
    (ys_fit, ys_var, log_lik) = gpr.fit(xs_fit)
    ax.plot(xs_fit, ys_fit, 'k--')
    #ys_err = 2.0 * np.sqrt(ys_var)
    #ax.fill_between(xs_fit, ys_fit + ys_err, ys_fit - ys_err, alpha=0.1, facecolor='gray', interpolate=True)
    ax.plot([-4.46, -4.46], [-3.7, -1.2], 'k-.')


def plot_data(data, save, ax11, ax12, ax2, ax31, ax32, ax4):
    activity_ls_1 = []
    activity_ls_2 = []
    handles = []
    labels = []
    used_labels = dict()
    for star in data.keys():
        is_ms = star_is_ms(star)
        data_star = data[star]

        data_star_arr = np.asarray(data_star)
        #data_star_arr = data_star_arr[np.where(data_star_arr[:,0] != None)]
        r0 = float(data_star_arr[0,4])
        g0 = float(data_star_arr[0,5])
        b0 = float(data_star_arr[0,6])
        alpha0 = float(data_star_arr[0,7])
        if len(data[star]) > 1:
            print star
            print data_star_arr[:,0]
            print data_star_arr[:,1]
        ax11.plot(data_star_arr[:,0], data_star_arr[:,1], linestyle=':', color=(r0, g0, b0, alpha0), lw=1.5)
        #inds = np.where(data_star_arr[:,11])[0] # is_ms
        if is_ms and not ax2 is None:
            p = data_star_arr[:,10].astype(float)
            c = data_star_arr[:,11].astype(float)
            ax2.plot(np.log10(p), np.log10(c), linestyle=':', color=(r0, g0, b0, alpha0), lw=1.5)
        if plot_ro and not ax12 is None:
            ax12.plot(data_star_arr[:,5], data_star_arr[:,1], linestyle=':', color=(r0, g0, b0, alpha0), lw=1.5)
        for [r_hk, y, err1, err2, r, g, b, alpha, ro, sym, p_rot, p_cyc, delta_i, cyc_err, size, p_rot_err, label] in data_star:
            err1 *= 2.0
            err2 *= 2.0
            cyc_err *= 2.0
            p_rot_err *= 2.0
            activity_ls_1.append([r_hk, y])
            activity_ls_2.append([ro, y])
            fillstyles = [None]
            syms = [sym]
            facecolors = [[r, g, b, alpha]]
            sizes = [size]
            if star == "SUN":
                fillstyles = [None, 'full']
                facecolors = ['none', [r, g, b, alpha]]
                syms = ['o', 'o']
                sizes = [size, 1]
            elif sym == 'd' or sym == 's' or sym == 'p' or sym == '*' or sym == "^" or sym == ".":
                facecolors = ['none']
            first_time = True
            for fillstyle, sym, facecolor, size in zip(fillstyles, syms, facecolors, sizes):
                handle = None
                if not first_time or err1 < 0.02 and err2 < 0.02:
                    handle = ax11.scatter(r_hk, y, marker=markers.MarkerStyle(sym, fillstyle=fillstyle), lw=1.5, facecolors=facecolor, color=[r, g, b, alpha], s=size, edgecolors=[r, g, b, alpha])
                    if is_ms and not ax2 is None: # omit non MS
                        ax2.scatter(np.log10(p_rot), np.log10(p_cyc), marker=markers.MarkerStyle(sym, fillstyle=fillstyle), lw=1.5, facecolors=facecolor, color=[r, g, b, alpha], s=size, edgecolors=[r, g, b, alpha])
                    if is_ms and not ax4 is None: # omit non MS
                        ax4.scatter(np.log(1.0/p_rot), y, marker=markers.MarkerStyle(sym, fillstyle=fillstyle), lw=1.5, facecolors=facecolor, color=[r, g, b, alpha], s=size, edgecolors=[r, g, b, alpha])
                    if plot_ro and not ax12 is None:
                        ax12.scatter(ro, y, marker=markers.MarkerStyle(sym, fillstyle=fillstyle), lw=1.5, facecolors=facecolor, color=[r, g, b, alpha], size=size, edgecolors=[r, g, b, alpha])
                else:
                    ax11.errorbar(r_hk, y, yerr=[[err1], [err2]], fmt=sym, lw=1.5, capsize=3, capthick=1.5, color=[r, g, b, alpha], markersize=np.sqrt(size), mew=1.5, mfc=facecolor, fillstyle=fillstyle, mec=[r, g, b, alpha])
                    if is_ms and not ax2 is None: # omit non MS
                        ax2.errorbar(np.log10(p_rot), np.log10(p_cyc), yerr=[[cyc_err/p_cyc/np.log(10.0)], [cyc_err/p_cyc/np.log(10.0)]], fmt=sym, lw=1.5, capsize=3, capthick=1.5, color=[r, g, b, alpha], markersize=np.sqrt(size), mew=1.5, mfc=facecolor, fillstyle=fillstyle, mec=[r, g, b, alpha])
                    if is_ms and not ax4 is None: # omit non MS
                        ax4.errorbar(np.log(1.0/p_rot), y, yerr=[[err1], [err2]], fmt=sym, lw=1.5, capsize=3, capthick=1.5, color=[r, g, b, alpha], markersize=np.sqrt(size), mew=1.5, mfc=facecolor, fillstyle=fillstyle, mec=[r, g, b, alpha])
                    if plot_ro and not ax12 is None:
                        ax12.errorbar(ro, y, yerr=[[err1], [err2]], fmt=sym, lw=1.5, capsize=3, capthick=1.5, color=[r, g, b, alpha], markersize=np.sqrt(size), mew=1.5, mfc=facecolor, fillstyle=fillstyle, mec=[r, g, b, alpha])
                if (star != "SUN" and handle is not None and not used_labels.has_key(label)):
                    used_labels[label] = None
                    handles.append(handle)
                    labels.append(label)
                if not first_time or (cyc_err/p_cyc/np.log(10.0) < 0.02 and p_rot_err/p_rot/np.log(10.0) < 0.02):
                    if is_ms and not ax2 is None: # omit non MS
                        ax2.scatter(np.log10(p_rot), np.log10(p_cyc), marker=markers.MarkerStyle(sym, fillstyle=fillstyle), lw=1.5, facecolors=facecolor, color=[r, g, b, alpha], s=size, edgecolors=[r, g, b, alpha])
                else:
                    if is_ms and not ax2 is None: # omit non MS
                        ax2.errorbar(np.log10(p_rot), np.log10(p_cyc), xerr=[[p_rot_err/p_rot/np.log(10.0)], [p_rot_err/p_rot/np.log(10.0)]], yerr=[[cyc_err/p_cyc/np.log(10.0)], [cyc_err/p_cyc/np.log(10.0)]], fmt=sym, lw=1.5, capsize=3, capthick=1.5, color=[r, g, b, alpha], markersize=np.sqrt(size), mew=1.5, mfc=facecolor, fillstyle=fillstyle, mec=[r, g, b, alpha])


                if star_FeH_dR.has_key(star) and not ax31 is None and not ax32 is None:
                    ax31.scatter(star_FeH_dR[star][1], delta_i, marker=markers.MarkerStyle(sym, fillstyle=fillstyle), lw=1, facecolors=facecolors, color=[r, g, b, alpha], s=size, edgecolors=[r, g, b, alpha])
                    ax32.scatter(star_FeH_dR[star][0], delta_i, marker=markers.MarkerStyle(sym, fillstyle=fillstyle), lw=1, facecolors=facecolors, color=[r, g, b, alpha], s=size, edgecolors=[r, g, b, alpha])
                first_time = False
    if save:
        np.savetxt("activity_" + type + "_rhk.txt", activity_ls_1, fmt='%f')
        if plot_ro:
            np.savetxt("activity_" + type +"_rho.txt", activity_ls_2, fmt='%f')
    
    if plot_ro:
        ax12.set_ylabel(r'${\rm log}P_{\rm rot}/P_{\rm cyc}$', fontsize=axis_label_fs)
    
    if not ax2 is None:
        #ax2.set_ylabel(r'$P_{\rm cyc}$ [yr]', fontsize=axis_label_fs)
        ax2.set_ylabel(r'$\log P_{\rm cyc}$ [$\log$ yr]', fontsize=axis_label_fs)
        #ax2.set_xlim(2.0, 4.0)
        #ax2.set_ylim(3, 30)
        #ax2.loglog()

    if not ax4 is None:
        ax4.set_ylabel(r'${\rm log}P_{\rm rot}/P_{\rm cyc}$', fontsize=axis_label_fs)

    if star_FeH_dR.has_key(star) and not ax31 is None and not ax32 is None:
        ax31.set_ylabel(r'$\Delta_i$', fontsize=axis_label_fs)
        ax32.set_ylabel(r'$\Delta_i$', fontsize=axis_label_fs)
        
        std1 = np.sqrt((1.0 - s1[0, 1]*s1[0, 1]/s1[0, 0]/s1[1, 1])*s1[1, 1])
        std2 = np.sqrt((1.0 - s2[0, 1]*s2[0, 1]/s2[0, 0]/s2[1, 1])*s2[1, 1])
        xmin, xmax = ax31.get_xlim()
        ax31.plot([xmin, xmax], [std1, std1], color='blue', linestyle='--', linewidth=1)
        ax31.plot([xmin, xmax], [-std1, -std1], color='blue', linestyle='--', linewidth=1)
        ax31.plot([xmin, xmax], [std2, std2], color='red', linestyle='-.', linewidth=1)
        ax31.plot([xmin, xmax], [-std2, -std2], color='red', linestyle='-.', linewidth=1)
        ax31.set_xlim(xmin, xmax)

        xmin, xmax = ax32.get_xlim()
        ax32.plot([xmin, xmax], [std1, std1], color='blue', linestyle='--', linewidth=1)
        ax32.plot([xmin, xmax], [-std1, -std1], color='blue', linestyle='--', linewidth=1)
        ax32.plot([xmin, xmax], [std2, std2], color='red', linestyle='-.', linewidth=1)
        ax32.plot([xmin, xmax], [-std2, -std2], color='red', linestyle='-.', linewidth=1)
        ax32.set_xlim(xmin, xmax)
    
    return (handles, labels)
    #fig1.subplots_adjust(left=0.1, right=0.97, top=0.98, bottom=0.05, hspace=0.1)
    
for type in ["BGLST", "GP_P", "GP_QP"]:

    if type == "BGLST":
        if plot_ro:
            ax11 = ax111
            ax12 = ax112
        else:
            ax11 = ax11            
        ax2 = ax21
        ax4 = ax41
        input_path = "BGLST_BIC_6/results.txt"
        bglst_or_gp = True
    elif type == "GP_P":
        if plot_ro:
            ax11 = ax121
            ax12 = ax122
        else:
            ax11 = ax12            
        ax2 = ax22
        ax4 = ax42
        input_path = "GP_periodic/results_combined.txt"
        bglst_or_gp = False
    elif type == "GP_QP":
        if plot_ro:
            ax11 = ax131
            ax12 = ax132
        else:
            ax11 = ax13            
        ax2 = ax23
        ax4 = ax43
        input_path = "GP_quasiperiodic/results_combined.txt"
        bglst_or_gp = False
    else:
        assert(False)    


    if bglst_or_gp:
        min_bic, max_bic, cycles = mw_utils.read_bglst_cycles(input_path)
    else:
        min_bic, max_bic, cycles = read_gp_cycles(input_path)

    clustered = False
    ###############################################################################
    suffix = ""
    if use_secondary_clusters:
        suffix = "_secondary"
    if os.path.isfile("clusters_" + type + suffix + ".txt"):
        clustered = True
        dat = np.loadtxt("clusters_" + type + suffix + ".txt", usecols=(0,1), skiprows=0)
    
        m1 = dat[0,:]
        m2 = dat[1,:]
        s1 = dat[2:4,:]
        s2 = dat[4:6,:]
        w1, v1 = LA.eig(s1)
        w2, v2 = LA.eig(s2)
        
        # Just swapping the color of custers if incorrect
        if type == "BGLST":# or type == "GP_QP":
            m_temp = m2
            s_temp = s2
            w_temp = w2
            v_temp = v2
            m2 = m1
            s2 = s1
            w2 = w1
            v2 = v1
            m1 = m_temp
            s1 = s_temp
            w1 = w_temp
            v1 = v_temp            
            
        #print w1
        #print v1
        #print w2
        #print v2
        cos1= v1[0,0]
        sin1= v1[1,0]
        angle1 = np.arccos(cos1)
        if sin1 < 0:
            angle1 = -angle1
        
        e1 = Ellipse(xy=m1, width=2*np.sqrt(w1[0]), height=2*np.sqrt(w1[1]), angle=angle1*180/np.pi, linestyle=None, linewidth=0)
        ax11.add_artist(e1)
        #e1.set_clip_box(ax1.bbox)
        e1.set_alpha(0.25)
        e1.set_facecolor('blue')
        #ax11.plot([m1[0], m1[0]+0.1], [m1[1], m1[1]+s1[0,1]/s1[0,0]*(0.1)], color='k', linestyle='-', linewidth=1)
        #ax11.plot([m1[0], m1[0]+v1[0,0]*0.1], [m1[1], m1[1]+v1[1,0]*0.1], color='k', linestyle='--', linewidth=1)
        #ax11.plot([m1[0], m1[0]+v1[0,1]*0.1], [m1[1], m1[1]+v1[1,1]*0.1], color='k', linestyle='--', linewidth=1)
    
        cos2= v2[0,0]
        sin2= v2[1,0]
        angle2 = np.arccos(cos2)
        if sin2 < 0:
            angle2 = -angle2
        
        e2 = Ellipse(xy=m2, width=2*np.sqrt(w2[0]), height=2*np.sqrt(w2[1]), angle=angle2*180/np.pi, linestyle=None, linewidth=0)
        ax11.add_artist(e2)
        #e2.set_clip_box(ax1.bbox)
        e2.set_alpha(0.25)
        e2.set_facecolor('red')
        #ax11.plot([m2[0], m2[0]+0.1], [m2[1], m2[1]+s2[0,1]/s2[0,0]*(0.1)], color='k', linestyle='-', linewidth=1)
        #ax11.plot([m2[0], m2[0]+v2[0,0]*0.1], [m2[1], m2[1]+v2[1,0]*0.1], color='k', linestyle='--', linewidth=1)
        #ax11.plot([m2[0], m2[0]+v2[0,1]*0.1], [m2[1], m2[1]+v2[1,1]*0.1], color='k', linestyle='--', linewidth=1)
    
        if angle1 > np.pi/2:
            angle1 -= np.pi/2
        if angle2 > np.pi/2:
            angle2 -= np.pi/2
            
        a1 = s1[0,1]/s1[0,0]
        b1 = m1[1] - s1[0,1]/s1[0,0]*m1[0]
        a2 = s2[0,1]/s2[0,0]
        b2 = m2[1] - s2[0,1]/s2[0,0]*m2[0]
        #ax11.plot([m1[0], m1[0]+0.1], [m1[1], a1*(m1[0]+0.1)+b1], color='k', linestyle='-', linewidth=1)
        #ax11.plot([m2[0], m2[0]+0.1], [m2[1], a2*(m2[0]+0.1)+b2], color='k', linestyle='-', linewidth=1)
        
        
        print "y1=" + str(a1) + "x+" + str(b1)
        print "y2=" + str(a2) + "x+" + str(b2)
    ###############################################################################
    
    i = 0
    data = dict()
    data_baliunas = dict()
    plot_ro = False
    
    active_cyc_mean = 0.0
    num_active = 0.0
    inactive_cyc_mean = 0.0
    num_inactive = 0.0
    active = []
    inactive = []
    with open("mwo-rhk.dat", "r") as ins:
        for line in ins:
            if i < 5:
                i += 1
                continue
            fields = parse(line)
            star = fields[0].strip()
            star = star.replace(' ', '')
            star = star.upper()
            try:
                bmv = float(fields[4].strip())
            except ValueError:
                bmv = 0.0
            try:
                r_hk = float(fields[6].strip())
            except ValueError:
                r_hk = None
            if (rot_periods.has_key(star)):
                (p_rot, p_rot_err) = rot_periods[star]
            else: 
                p_rot = None
            #try:
            #    p_rot = float(fields[7].strip())
            #except ValueError:
            #    p_rot = None
            if p_rot != None and r_hk != None and bmv != None:
                if bmv >= 1.0:
                    tau = 25.0
                else:
                    tau = np.power(10.0, -3.33 + 15.382*bmv - 20.063*bmv**2 + 12.540*bmv**3 - 3.1466*bmv**4)
                    #ro = np.log10(4*np.pi*tau/p_rot)
                ro = np.log10(4*np.pi*tau/p_rot)
                #print star, tau, bmv
                dark_color =  "black"
                light_color =  "gray"
                is_ms = star_is_ms(star)
                    
                for (cycs, baliunas) in [(cycles, False), (baliunas_cycles, True)]:
                    alpha = 0.9
                    sym = "d"
                    if is_ms:
                        sym = "+"
                    if star == "SUN":
                        sym = "*"
                    if cycs.has_key(star) and (is_ms or include_non_ms):
                        data_star = []
                        primary_cycle = True
                        for (p_cyc, std, bic) in cycs[star]:
                            label = "None"
                            exclude = False
                            for (p_cyc_2, std_2, bic_2) in cycs[star]:
                                if p_cyc != p_cyc_2:
                                    for ii in [2.0, 3.0]:
                                        if p_cyc_2 + std_2 > (p_cyc - 3.0*std) * ii and p_cyc_2 - std_2 < (p_cyc + 3.0*std) * ii:
                                            #print p_cyc, p_cyc_2
                                            #exclude = True
                                            break
                            if exclude:
                                continue
                            #r_hks_ls.append(r_hk)
                            val = np.log10(p_rot/p_cyc)
                            err = (p_rot_err/p_rot+std/p_cyc)/np.log(10)#std/p_cyc/np.log(10)
                            #val1 = np.log10(p_rot/(p_cyc + std))
                            #val2 = np.log10(p_rot/(p_cyc - std))
                            #print val - val1, val2 - val, err
                            delta_i = None
                            if primary_cycle:
                                size = 100
                            else:
                                size = 50
                            primary_cycle = False
                            if baliunas:
                                alpha = 0.7
                                c = 0.5*(1.0 - (bic - min_baliunas_grade)/(max_baliunas_grade - min_baliunas_grade))
                                #if bic > 1.0:
                                #    r = c
                                #    g = 1.0
                                #    b = c
                                #else:                                    
                                r = c
                                g = c
                                b = c
                                sym = "."
                                if clustered and is_ms:
                                    point = np.array([r_hk, val])
                                    dist1 = np.dot(point - m1, np.dot(LA.inv(s1), point - m1))
                                    dist2 = np.dot(point - m2, np.dot(LA.inv(s2), point - m2))
                                    if dist1 < dist2:
                                        label = "Active Baliunas"
                                    else:
                                        label = "Non-active Baliunas"
                            else:
                                if clustered and is_ms:
                                    point = np.array([r_hk, val])
                                    dist1 = np.dot(point - m1, np.dot(LA.inv(s1), point - m1))
                                    dist2 = np.dot(point - m2, np.dot(LA.inv(s2), point - m2))
                                    if bic > 100:
                                        c = 0.0
                                    else:
                                        c = 0.5 - 0.5 * (bic - min_bic)/(max_bic - min_bic)
                                    if dist1 < dist2:
                                        label = "Active"
                                        sym = "+"
                                        delta_i = val - (a1 * r_hk + b1)
                                        r = c
                                        g = c
                                        b = 1.0
                                        active_cyc_mean += p_cyc/365.25
                                        num_active += 1
                                        active.append([r_hk, val, err])
                                    else:
                                        label = "Non-active"
                                        sym = "x"
                                        delta_i = val - (a2 * r_hk + b2)
                                        r = 1.0
                                        g = c
                                        b = c
                                        inactive_cyc_mean += p_cyc/365.25
                                        num_inactive += 1
                                        inactive.append([r_hk, val, err])
                                else:                            
                                    if bic > 100:
                                        r = 0.0
                                        g = 0.0
                                        b = 0.0
                                    else:
                                        c = 0.5*(1.0 - (bic - min_bic)/(max_bic - min_bic))
                                        r = c
                                        g = c
                                        b = c
                                #size *= 1.0 - c
                            data_star.append([r_hk, val, err, err, r, g, b, alpha, ro, sym, p_rot, p_cyc/365.25, delta_i, std/365.25, size, p_rot_err, label])
                        if baliunas:
                            data_baliunas[star] = data_star
                        else:
                            data[star] = data_star
            #print star, bmv, r_hk, p_rot
    if type == "BGLST":
        plot_data(data, True, ax11, ax12, ax2, ax31, ax32, ax4)
        # Comparison to Baliunas
        plot_data(data, False, ax1a, None, None, None, None, None)        
        plot_data(data_baliunas, False, ax1a, None, None, None, None, None)
    else:
        # don't plot the resudue plot
        plot_data(data, True, ax11, ax12, ax2, None, None, ax4)
        if type == "GP_QP":
            # Comparison to Simulations and Jyri's results
            handles1, labels1 = plot_data(data, False, ax1b, None, None, None, None, None)
            handles2, labels2 = plot_data(ext_data_for_plot, False, ax1b, None, None, None, None, None)
            ax1b.legend(handles1 + handles2, labels1 + labels2,
                        numpoints = 1,
                        scatterpoints=1,
                        loc='upper right', ncol=1,
                        fontsize=8, labelspacing=1)
            for star in data.keys():
                if data_for_fit.has_key(star):
                    print "Duplicate star:", star 
                else:
                    if star_is_ms(star):
                        data_for_fit[star] = data[star]
            if fit_with_baliunas:
                for star in data_baliunas.keys():
                    if data_for_fit.has_key(star):
                        print "Duplicate Baliunas star:", star 
                    else:
                        data_for_fit[star] = data_baliunas[star]
            fit_data(data_for_fit, ax1b)
    ###########################################################################
    # Calculate trend lines for the branches
    
    active = np.asarray(active)
    inactive = np.asarray(inactive)
    ((mu_alpha, mu_beta), (sigma_alpha, sigma_beta), y_model, loglik) = bayes_lin_reg(active[:,0], active[:,1], np.ones(len(active[:,2]))/active[:,2])
    rhks = np.linspace(min(active[:,0]), max(active[:,0]), 100)
    print "active branch slope:", mu_alpha, 2.0*sigma_alpha
    #ax11.plot(rhks, rhks * mu_alpha + mu_beta, color='k', linestyle='-', linewidth=1)
    ((mu_alpha, mu_beta), (sigma_alpha, sigma_beta), y_model, loglik) = bayes_lin_reg(inactive[:,0], inactive[:,1], np.ones(len(inactive[:,2]))/inactive[:,2])
    print "inactive branch slope:", mu_alpha, 2.0*sigma_alpha
    rhks = np.linspace(min(inactive[:,0]), max(inactive[:,0]), 100)
    #ax11.plot(rhks, rhks * mu_alpha + mu_beta, color='k', linestyle='-', linewidth=1)
    
    #np.savetxt("active_" + type + ".dat", np.asarray(active), fmt='%f')
    #np.savetxt("inactive_" + type + ".dat", np.asarray(inactive), fmt='%f')
    ###########################################################################
    
    print "active_cyc_mean: ", active_cyc_mean/num_active
    print "inactive_cyc_mean: ", inactive_cyc_mean/num_inactive

fig1.savefig("activity_diagram.pdf")
plt.close(fig1)

#fig1a.savefig("activity_diagram_cmp.pdf")
#plt.close(fig1a)


ax1b.set_ylim(-3.7, -1.2)
ax1b.set_xlim(-5.1, -4.0)
ax1b.set_ylabel(r'${\rm log}P_{\rm rot}/P_{\rm cyc}$', fontsize=axis_label_fs)
fig1b.savefig("activity_diagram_cmp2.pdf")
plt.close(fig1b)

fig2.savefig("activity_diagram_2.pdf")
plt.close(fig2)

#fig3.savefig("residues.pdf")
#plt.close(fig3)

fig4.savefig("activity_diagram_3.pdf")
plt.close(fig4)
