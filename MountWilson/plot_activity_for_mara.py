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

def read_cycles(file):
    all_cycles = dict()
    data = pd.read_csv(file, names=['star', 'cyc', '2sigma'], header=0, dtype=None, sep='\s+', engine='python').as_matrix()
    
    #data = np.genfromtxt(file, dtype=None, skip_header=1)
    for [star, cyc, std_2] in data:
        if star == 'SUNALL':
            star = 'Sun'
        #print star, cyc, std_2
        if not np.isnan(cyc):
            if not all_cycles.has_key(star):
                all_cycles[star] = []
            cycles = all_cycles[star]
            #cycles.append((cyc*365.25, std_2/2*365.25)) # one sigma
            cycles.append((cyc*365.25, std_2*1.5*365.25)) # three sigma
            all_cycles[star] = cycles
    return all_cycles


def read_bglst_cycles(file):
    max_bic = None
    min_bic = None
    all_cycles = dict()
    data = pd.read_csv(file, names=['star', 'cyc', 'sigma', 'normality', 'bic'], header=0, dtype=None, sep='\s+', engine='python').as_matrix()
    
    #data = np.genfromtxt(file, dtype=None, skip_header=1)
    for [star, cyc, std, normality, bic] in data:
        if star == 'SUNALL':
            star = 'Sun'
        #print star, cyc, std_2
        if not np.isnan(cyc):
            if not all_cycles.has_key(star):
                all_cycles[star] = []
            cycles = all_cycles[star]
            log_bic = np.log(bic)
            if max_bic == None or log_bic > max_bic:
                max_bic = log_bic
            if min_bic == None or log_bic < min_bic:
                min_bic = log_bic
                
            cycles.append((cyc*365.25, std*3*365.25, log_bic)) # three sigma
            all_cycles[star] = cycles
    return min_bic, max_bic, all_cycles

min_bic, max_bic, bglst_cycles = read_bglst_cycles("BGLST/results.txt")

#ls_cycles001 = read_cycles("ls_results0.01.txt")
#ls_cycles005 = read_cycles("ls_results0.05.txt")
d2_cycles001 = read_cycles("d2_results0.01.txt")
d2_cycles005 = read_cycles("d2_results0.05.txt")

i = 0
data_bglst = []
data_d2 = []

with open("mwo-rhk.dat", "r") as ins:
    for line in ins:
        if i < 5:
            i += 1
            continue
        fields = parse(line)
        star = fields[0].strip()
        star = star.replace(' ', '')
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
            data_ls_star = None
            if bmv >= 1.0:
                tau = 25.0
            else:
                tau = np.power(10.0, -3.33 + 15.382*bmv - 20.063*bmv**2 + 12.540*bmv**3 - 3.1466*bmv**4)
                #ro = np.log10(4*np.pi*tau/p_rot)
                ro = np.log10(4*np.pi*tau/p_rot)
            dark_color =  "blue"
            light_color =  "lightblue"
            #if star == "Sun":
            #    dark_color = "gold"
            #    light_color = "lemonchiffon"
            if bglst_cycles.has_key(star):
                data_bglst_star = []
                for (p_cyc, std, bic) in bglst_cycles[star]:
                    val = np.log10(p_rot/p_cyc)
                    err = std/p_cyc/np.log(10)
                    #if star == "Sun":
                    #    r = 1.0
                    #    g = 215.0/255.0
                    #    b = 0.0
                    if bic > 100:
                        r = 0.0
                        g = 0.0
                        b = 0.0
                    else:
                        c = 0.5*(1.0 - (bic - min_bic)/(max_bic - min_bic))
                        r = c
                        g = c
                        b = c
                    data_bglst_star.append([r_hk, val, err, err, r, g, b, ro, "o"])
                data_bglst.append(data_bglst_star)
            data_d2_star = None
            if d2_cycles001.has_key(star):
                data_d2_star = []
                for (p_cyc, std) in d2_cycles001[star]:
                    #r_hks_d2.append(r_hk)
                    val = np.log10(p_rot/p_cyc)
                    err = std/p_cyc/np.log(10)
                    #p_rot_div_cyc_d2.append([val, err, err])
                    data_d2_star.append([r_hk, val, err, err, dark_color, ro, "s"])
                data_d2.append(data_d2_star)
            if False:#d2_cycles005.has_key(star):
                if data_d2_star == None:
                    data_d2_star = []
                for (p_cyc, std) in d2_cycles005[star]:
                    #r_hks_d2.append(r_hk)
                    val = np.log10(p_rot/p_cyc)
                    err = std/p_cyc/np.log(10)
                    #p_rot_div_cyc_d2.append([val, err, err])
                    data_d2_star.append([r_hk, val, err, err, light_color, ro, "s"])
                data_d2.append(data_d2_star)
        #print star, bmv, r_hk, p_rot



###############################################################################

#data = pd.read_csv("saar_brandenburg.dat", names=["star", "rhk", "tc", "p_rot", "p_cyc1", "p_cyc2"], header=0, dtype=None, sep='\s+', engine='python').as_matrix()

#for [star, rhk, tc, p_rot, p_cyc1, p_cyc2] in data:
#    if not np.isnan(p_cyc1):
#        ro = np.log10(4*np.pi*tc/p_rot)
#        data_d2_star = []
#        data_ls_star = []
#        val = np.log10(p_rot/p_cyc1/365.25)
#        color =  "blue"
#        if star == "4628" or star == "18256":
#            color = "green"
#        data_ls_star.append([rhk, val, 0, 0, color, ro, "s"])
#        data_d2_star.append([rhk, val, 0, 0, color, ro, "s"])
#        if not np.isnan(p_cyc2):
#            val = np.log10(p_rot/p_cyc2/365.25)
#            data_ls_star.append([rhk, val, 0, 0, color, ro, "s"])
#            data_d2_star.append([rhk, val, 0, 0, color, ro, "s"])
#        data_d2.append(data_d2_star)
#        data_ls.append(data_ls_star)


###############################################################################
fig1, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True)
fig1.set_size_inches(6, 5)
#fig.tight_layout(pad=2.5)


#ax1.text(0.9, 0.9,'(a)', horizontalalignment='center', transform=ax1.transAxes)
#ax2.text(0.9, 0.9,'(b)', horizontalalignment='center', transform=ax2.transAxes)


#for data_star in data_d2:
#    data_star_arr = np.asarray(data_star)
#    #data_star_arr = data_star_arr[np.where(data_star_arr[:,0] != None)]
#    ax1.plot(data_star_arr[:,0], data_star_arr[:,1], linestyle='--', color='blue', lw=1.0)
#    for [r_hk, y, err1, err2, color, ro, sym] in data_star:
#        if err1 == 0 and err2 == 0:
#            ax1.scatter(r_hk, y, marker=markers.MarkerStyle(sym, fillstyle='full'), lw=1, color=color, s=10, edgecolors=color)
#        else:
#            ax1.scatter(r_hk, y, marker=markers.MarkerStyle(sym, fillstyle='full'), lw=1, color=color, s=10, edgecolors=color)
#            #ax1.errorbar(r_hk, y, yerr=[[err1], [err2]], fmt=sym, lw=1, capsize=3, capthick=1, color=color, markersize=3, fillstyle='full', markeredgecolor=color)

for data_star in data_bglst:
    data_star_arr = np.asarray(data_star)
    #data_star_arr = data_star_arr[np.where(data_star_arr[:,0] != None)]
    ax1.plot(data_star_arr[:,0], data_star_arr[:,1], linestyle='--', color='silver', lw=1.0)
    for [r_hk, y, err1, err2, r, g, b, ro, sym] in data_star:
        if err1 == 0 and err2 == 0:
            ax1.scatter(r_hk, y, marker=markers.MarkerStyle(sym, fillstyle='full'), lw=1, color=[r, g, b], s=10, edgecolors=[r, g, b])
        else:
            ax1.scatter(r_hk, y, marker=markers.MarkerStyle(sym, fillstyle='full'), lw=1, color=[r, g, b], s=10, edgecolors=[r, g, b])
            #ax1.errorbar(r_hk, y, yerr=[[err1], [err2]], fmt=sym, lw=1, capsize=3, capthick=1, color=[r, g, b], markersize=3, fillstyle='full', markeredgecolor=[r, g, b])
           

            
ax1.set_ylabel(r'${\rm log}P_{\rm rot}/P_{\rm cyc}$')
ax1.set_xlabel(r'${\rm log}R\prime_{\rm HK}$')
#fig1.subplots_adjust(left=0.1, right=0.97, top=0.98, bottom=0.05, hspace=0.1)
ax1.set_xlim([-5.4,-4.0])
ax1.set_ylim([-3.5,-0.5])


fig1.savefig("activity_diagram_for_mara.eps")
plt.close(fig1)
