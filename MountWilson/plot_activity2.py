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


def read_FeH_dR(file):
    star_FeH_dR = dict()
    data = pd.read_csv(file, header=0, dtype=None, usecols=['HD/KIC', 'Fe/H', 'd/R'], sep=';', engine='python').as_matrix()
    for [star, FeH, dR] in data:
        star = star.upper()
        star_FeH_dR[star] = [FeH, dR]        
    return star_FeH_dR
    
star_FeH_dR = read_FeH_dR("brandenburg2017table.csv")

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
    fig1, (ax11, ax12, ax13) = plt.subplots(nrows=3, ncols=1, sharex=False)
    fig1.set_size_inches(6, 18)
    ax13.set_xlabel(r'${\rm log} \langle R^\prime_{\rm HK}\rangle$', fontsize=axis_label_fs)
    ax11.text(0.95, 0.9,'(a)', horizontalalignment='center', transform=ax11.transAxes, fontsize=panel_label_fs)
    ax12.text(0.95, 0.9,'(b)', horizontalalignment='center', transform=ax12.transAxes, fontsize=panel_label_fs)
    ax13.text(0.95, 0.9,'(c)', horizontalalignment='center', transform=ax13.transAxes, fontsize=panel_label_fs)
    #ax11.set_aspect('equal', 'datalim')
    #ax12.set_aspect('equal', 'datalim')
    #ax13.set_aspect('equal', 'datalim')

fig2, (ax21, ax22, ax23) = plt.subplots(nrows=3, ncols=1, sharex=False)
fig2.set_size_inches(6, 18)
ax21.text(0.95, 0.9,'(a)', horizontalalignment='center', transform=ax21.transAxes, fontsize=panel_label_fs)
ax22.text(0.95, 0.9,'(b)', horizontalalignment='center', transform=ax22.transAxes, fontsize=panel_label_fs)
ax23.text(0.95, 0.9,'(c)', horizontalalignment='center', transform=ax23.transAxes, fontsize=panel_label_fs)
ax23.set_xlabel(r'$P_{\rm rot}$ [d]', fontsize=axis_label_fs)

fig3, (ax31, ax32) = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
fig3.set_size_inches(12, 4)
ax31.text(0.95, 0.9,'(a)', horizontalalignment='center', transform=ax31.transAxes, fontsize=panel_label_fs)
ax32.text(0.95, 0.9,'(b)', horizontalalignment='center', transform=ax32.transAxes, fontsize=panel_label_fs)
ax31.set_xlabel(r'$d/R$', fontsize=axis_label_fs)
ax32.set_xlabel(r'[Fe/H] (dex)', fontsize=axis_label_fs)

for type in ["BGLST", "GP_P", "GP_QP"]:

    if type == "BGLST":
        if plot_ro:
            ax11 = ax111
            ax12 = ax112
        else:
            ax11 = ax11            
        ax2 = ax21
        input_path = "BGLST_BIC_6/results.txt"
        bglst_or_gp = True
    elif type == "GP_P":
        if plot_ro:
            ax11 = ax121
            ax12 = ax122
        else:
            ax11 = ax12            
        ax2 = ax22
        input_path = "GP_periodic/results_combined.txt"
        bglst_or_gp = False
    elif type == "GP_QP":
        if plot_ro:
            ax11 = ax131
            ax12 = ax132
        else:
            ax11 = ax13            
        ax2 = ax23
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
    if os.path.isfile("clusters_" + type + ".txt"):
        clustered = True
        dat = np.loadtxt("clusters_" + type + ".txt", usecols=(0,1), skiprows=0)
    
        m1 = dat[0,:]
        m2 = dat[1,:]
        s1 = dat[2:4,:]
        s2 = dat[4:6,:]
        w1, v1 = LA.eig(s1)
        w2, v2 = LA.eig(s2)
        
        # Just swapping the color of custers if incorrect
        if type == "GP_P" or type == "GP_QP":
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
                sym = "d"
                is_ms = star_is_ms(star)
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
                                        #print p_cyc, p_cyc_2
                                        #exclude = True
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
                        delta_i = None
                        if clustered and is_ms:
                            point = np.array([r_hk, val])
                            dist1 = np.dot(point - m1, np.dot(LA.inv(s1), point - m1))
                            dist2 = np.dot(point - m2, np.dot(LA.inv(s2), point - m2))
                            if bic > 100:
                                c = 0.0
                            else:
                                c = 0.5 - 0.5 * (bic - min_bic)/(max_bic - min_bic)
                            if dist1 < dist2:
                                sym = "+"
                                delta_i = val - (a1 * r_hk + b1)
                                r = c
                                g = c
                                b = 1.0
                            else:
                                sym = "x"
                                delta_i = val - (a2 * r_hk + b2)
                                r = 1.0
                                g = c
                                b = c
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
                        data_star.append([r_hk, val, err, err, r, g, b, ro, sym, p_rot, p_cyc/365.25, delta_i])
                    data[star] = data_star
            #print star, bmv, r_hk, p_rot

    activity_ls_1 = []
    activity_ls_2 = []
    for star in data.keys():
        is_ms = star_is_ms(star)
        data_star = data[star]
        data_star_arr = np.asarray(data_star)
        #data_star_arr = data_star_arr[np.where(data_star_arr[:,0] != None)]
        r0 = float(data_star_arr[0,4])
        g0 = float(data_star_arr[0,5])
        b0 = float(data_star_arr[0,6])
        print "Color", r0, g0, b0
        ax11.plot(data_star_arr[:,0], data_star_arr[:,1], linestyle=':', color=(r0, g0, b0), lw=1.5)
        #inds = np.where(data_star_arr[:,11])[0] # is_ms
        if is_ms:
            ax2.plot(data_star_arr[:,9], data_star_arr[:,10], linestyle=':', color=(r0, g0, b0), lw=1.5)
        if plot_ro:
            ax12.plot(data_star_arr[:,5], data_star_arr[:,1], linestyle=':', color=(r0, g0, b0), lw=1.5)
        for [r_hk, y, err1, err2, r, g, b, ro, sym, p_rot, p_cyc, delta_i] in data_star:
            activity_ls_1.append([r_hk, y])
            activity_ls_2.append([ro, y])
            fillstyles = [None]
            syms = [sym]
            facecolors = [[r, g, b]]
            sizes = [50]
            if sym == 'd':
                facecolors = ['none']
            elif star == "SUN":
                fillstyles = [None, 'full']
                facecolors = ['none', [r, g, b]]
                syms = ['o', 'o']
                sizes = [50, 1]
            first_time = True
            for fillstyle, sym, facecolor, size in zip(fillstyles, syms, facecolors, sizes):
                if star == "SUN":
                    print fillstyle
                if not first_time or err1 == 0 and err2 == 0:
                    ax11.scatter(r_hk, y, marker=markers.MarkerStyle(sym, fillstyle=fillstyle), lw=1.5, facecolors=facecolor, color=[r, g, b], s=size, edgecolors=[r, g, b])
                    if is_ms: # omit non MS
                        ax2.scatter(p_rot, p_cyc, marker=markers.MarkerStyle(sym, fillstyle=fillstyle), lw=1.5, facecolors=facecolor, color=[r, g, b], s=size, edgecolors=[r, g, b])
                    if plot_ro:
                        ax12.scatter(ro, y, marker=markers.MarkerStyle(sym, fillstyle=fillstyle), lw=1.5, facecolors=facecolor, color=[r, g, b], size=size, edgecolors=[r, g, b])
                else:
                    ax11.errorbar(r_hk, y, yerr=[[err1], [err2]], fmt=sym, lw=1.5, capsize=3, capthick=1.5, color=[r, g, b], markersize=np.sqrt(size), mew=1.5, mfc=facecolor, fillstyle=fillstyle, mec=[r, g, b])
                    if is_ms: # omit non MS
                        ax2.errorbar(p_rot, p_cyc, yerr=[[err1], [err2]], fmt=sym, lw=1.5, capsize=3, capthick=1.5, color=[r, g, b], markersize=np.sqrt(size), mew=1.5, mfc=facecolor, fillstyle=fillstyle, mec=[r, g, b])
                    if plot_ro:
                        ax12.errorbar(ro, y, yerr=[[err1], [err2]], fmt=sym, lw=1.5, capsize=3, capthick=1.5, color=[r, g, b], markersize=np.sqrt(size), mew=1.5, mfc=facecolor, fillstyle=fillstyle, mec=[r, g, b])
                if type == "BGLST" and star_FeH_dR.has_key(star):
                    ax31.scatter(star_FeH_dR[star][1], delta_i, marker=markers.MarkerStyle(sym, fillstyle=fillstyle), lw=1, facecolors=facecolors, color=[r, g, b], s=size, edgecolors=[r, g, b])
                    ax32.scatter(star_FeH_dR[star][0], delta_i, marker=markers.MarkerStyle(sym, fillstyle=fillstyle), lw=1, facecolors=facecolors, color=[r, g, b], s=size, edgecolors=[r, g, b])
                first_time = False
    np.savetxt("activity_" + type + "_rhk.txt", activity_ls_1, fmt='%f')
    if plot_ro:
        np.savetxt("activity_" + type +"_rho.txt", activity_ls_2, fmt='%f')
    
    ax11.set_ylabel(r'${\rm log}P_{\rm rot}/P_{\rm cyc}$', fontsize=axis_label_fs)
    if plot_ro:
        ax12.set_ylabel(r'${\rm log}P_{\rm rot}/P_{\rm cyc}$', fontsize=axis_label_fs)
    
    ax2.set_ylabel(r'$P_{\rm cyc}$ [yr]', fontsize=axis_label_fs)

    if type == "BGLST" and star_FeH_dR.has_key(star):
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
    
    #fig1.subplots_adjust(left=0.1, right=0.97, top=0.98, bottom=0.05, hspace=0.1)
    

fig1.savefig("activity_diagram.pdf")
plt.close(fig1)

fig2.savefig("activity_diagram_2.pdf")
plt.close(fig2)

fig3.savefig("residues.pdf")
plt.close(fig2)
