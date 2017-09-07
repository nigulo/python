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

include_non_ms = True

#type = "BGLST"
#type = "GP_P"
type = "GP_QP"

if type == "BGLST":
    input_path = "BGLST_BIC_6/results.txt"
    bglst_or_gp = True
elif type == "GP_P":
    input_path = "GP_periodic/processed_with_cycles.txt"
    bglst_or_gp = False
elif type == "GP_QP":
    input_path = "GP_quasiperiodic/processed_with_cycles.txt"
    bglst_or_gp = False
else:
    assert(False)    

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

def read_bglst_cycles(file):
    max_bic = None
    min_bic = None
    all_cycles = dict()
    data = pd.read_csv(file, names=['star', 'f', 'sigma', 'normality', 'bic'], header=None, dtype=None, sep='\s+', engine='python').as_matrix()
    
    #data = np.genfromtxt(file, dtype=None, skip_header=1)
    for [star, f, std, normality, bic] in data:
        #if star == 'SUNALL':
        #    star = 'SUN'
        #print star, cyc, std_2
        if not np.isnan(f):
            if not all_cycles.has_key(star):
                all_cycles[star] = []
            cycles = all_cycles[star]
            log_bic = np.log(bic)
            if max_bic is None or log_bic > max_bic:
                max_bic = log_bic
            if min_bic is None or log_bic < min_bic:
                min_bic = log_bic
                
            cyc = 1.0/f
            
            f_samples = np.random.normal(loc=f, scale=std, size=1000)
            cyc_std = np.std(np.ones(len(f_samples))/f_samples)
            if cyc_std < cyc:
                cycles.append((cyc*365.25, cyc_std*3*365.25, log_bic)) # three sigma
                all_cycles[star] = cycles
    return min_bic, max_bic, all_cycles

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
                cycles.append((cyc*365.25, std*3*365.25, log_bic)) # three sigma
                all_cycles[star] = cycles
    return min_bic, max_bic, all_cycles

if bglst_or_gp:
    min_bic, max_bic, cycles = read_bglst_cycles(input_path)
else:
    min_bic, max_bic, cycles = read_gp_cycles(input_path)


i = 0
data = []
plot_ro = False

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
        try:
            p_rot = float(fields[7].strip())
        except ValueError:
            p_rot = None
        if p_rot != None and r_hk != None and bmv != None:
            data_star = None
            if bmv >= 1.0:
                tau = 25.0
            else:
                tau = np.power(10.0, -3.33 + 15.382*bmv - 20.063*bmv**2 + 12.540*bmv**3 - 3.1466*bmv**4)
                #ro = np.log10(4*np.pi*tau/p_rot)
                ro = np.log10(4*np.pi*tau/p_rot)
            #print star, tau, bmv
            dark_color =  "black"
            light_color =  "gray"
            sym = "o"
            is_ms = False
            if len(np.where(ms_stars == star.upper())[0] > 0):
                is_ms = True
            if is_ms:
                sym = "+"
            if star == "SUN":
                sym = "*"
                #dark_color = "gold"
                #light_color = "lemonchiffon"
            #if star == "10780" or star == "155886":
            #    dark_color = "red"
            #    light_color = "lightsalmon"
            if cycles.has_key(star) and (is_ms or include_non_ms):
                data_star = []
                for (p_cyc, std, bic) in cycles[star]:
                    exclude = False
                    for (p_cyc_2, std_2, bic_2) in cycles[star]:
                        if p_cyc != p_cyc_2:
                            for i in [2.0, 3.0]:
                                if p_cyc_2 + std_2 > (p_cyc - std) * i and p_cyc_2 - std_2 < (p_cyc + std) * i:
                                    print p_cyc, p_cyc_2
                                    exclude = True
                                    break
                    if exclude:
                        continue
                    #r_hks_ls.append(r_hk)
                    val = np.log10(p_rot/p_cyc)
                    err = std/p_cyc/np.log(10)
                    if err < 0.02:
                        err = 0.0
                    #val1 = np.log10(p_rot/(p_cyc + std))
                    #val2 = np.log10(p_rot/(p_cyc - std))
                    #print val - val1, val2 - val, err
                    if bic > 100:
                        r = 0.0
                        g = 0.0
                        b = 0.0
                    else:
                        c = 0.5*(1.0 - (bic - min_bic)/(max_bic - min_bic))
                        r = c
                        g = c
                        b = c
                    data_star.append([r_hk, val, err, err, r, g, b, ro, sym])
                data.append(data_star)
        #print star, bmv, r_hk, p_rot



###############################################################################
if plot_ro:
    fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=False)
    fig1.set_size_inches(6, 8)
else:
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, sharex=False)
    fig1.set_size_inches(6, 5)

###############################################################################
if plot_ro:
    ax1.text(0.9, 0.9,'(a)', horizontalalignment='center', transform=ax1.transAxes)
    ax2.text(0.9, 0.9,'(b)', horizontalalignment='center', transform=ax2.transAxes)

activity_ls_1 = []
activity_ls_2 = []
for data_star in data:
    data_star_arr = np.asarray(data_star)
    #data_star_arr = data_star_arr[np.where(data_star_arr[:,0] != None)]
    ax1.plot(data_star_arr[:,0], data_star_arr[:,1], linestyle=':', color='gray', lw=1.0)
    if plot_ro:
        ax2.plot(data_star_arr[:,5], data_star_arr[:,1], linestyle=':', color='gray', lw=1.0)
    for [r_hk, y, err1, err2, r, g, b, ro, sym] in data_star:
        activity_ls_1.append([r_hk, y])
        activity_ls_2.append([ro, y])
        if err1 == 0 and err2 == 0:
            ax1.scatter(r_hk, y, marker=markers.MarkerStyle(sym, fillstyle='full'), lw=1, color=[r, g, b], s=36, edgecolors=[r, g, b])
            if plot_ro:
                ax2.scatter(ro, y, marker=markers.MarkerStyle(sym, fillstyle='full'), lw=1, color=[r, g, b], size=36, edgecolors=[r, g, b])
        else:
            ax1.errorbar(r_hk, y, yerr=[[err1], [err2]], fmt=sym, lw=1, capsize=3, capthick=1, color=[r, g, b], markersize=6, fillstyle='full', markeredgecolor=[r, g, b])
            if plot_ro:
                ax2.errorbar(ro, y, yerr=[[err1], [err2]], fmt=sym, lw=1, capsize=3, capthick=1, color=[r, g, b], markersize=6, fillstyle='full', markeredgecolor=[r, g, b])
            
np.savetxt("activity_1.txt", activity_ls_1, fmt='%f')
np.savetxt("activity_2.txt", activity_ls_2, fmt='%f')

ax1.set_xlabel(r'${\rm log} \langle R\prime_{\rm HK}\rangle$')
ax1.set_ylabel(r'${\rm log}P_{\rm rot}/P_{\rm cyc}$')
if plot_ro:
    ax2.set_xlabel(r'${\rm log}{\rm Ro}^{-1}$')
    ax2.set_ylabel(r'${\rm log}P_{\rm rot}/P_{\rm cyc}$')
#fig1.subplots_adjust(left=0.1, right=0.97, top=0.98, bottom=0.05, hspace=0.1)

###############################################################################
if os.path.isfile("clusters.txt"):
    dat = np.loadtxt("clusters.txt", usecols=(0,1), skiprows=0)

    m1 = dat[0,:]
    m2 = dat[1,:]
    s1 = dat[2:4,:]
    s2 = dat[4:6,:]
    w1, v1 = LA.eig(s1)
    w2, v2 = LA.eig(s2)
    
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
    ax1.add_artist(e1)
    #e1.set_clip_box(ax1.bbox)
    e1.set_alpha(0.25)
    e1.set_facecolor('blue')
    #ax1.plot([m1[0], m1[0]+np.cos(angle1)], [m1[1], m1[1]+np.sin(angle1)], color='k', linestyle='-', linewidth=1)

    cos2= v2[0,0]
    sin2= v2[1,0]
    angle2 = np.arccos(cos2)
    if sin2 < 0:
        angle2 = -angle2
    
    e2 = Ellipse(xy=m2, width=2*np.sqrt(w2[0]), height=2*np.sqrt(w2[1]), angle=angle2*180/np.pi, linestyle=None, linewidth=0)
    ax1.add_artist(e2)
    #e2.set_clip_box(ax1.bbox)
    e2.set_alpha(0.25)
    e2.set_facecolor('red')
    #ax1.plot([m2[0], m2[0]+np.cos(angle2)], [m2[1], m2[1]+np.sin(angle2)], color='k', linestyle='-', linewidth=1)

    if angle1 > np.pi/2:
        angle1 -= np.pi/2
    if angle2 > np.pi/2:
        angle2 -= np.pi/2
    print "y1=" + str(np.tan(angle1)) + "x+" + str(m1[1] - np.tan(angle1) * m1[0])
    print "y2=" + str(np.tan(angle2)) + "x+" + str(m2[1] - np.tan(angle2) * m2[0])

fig1.savefig("activity_diagram.pdf")
plt.close(fig1)
