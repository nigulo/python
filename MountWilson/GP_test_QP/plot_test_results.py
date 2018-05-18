# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 20:20:51 2017

@author: nigul
"""

import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import mw_utils

axis_label_fs = 15
panel_label_fs = 15

dat = np.loadtxt("test/results.txt")

ids = dat[:, 0].astype(int)

num_iters = 400
freq_row = 2
n_eff_col = 9

indices = list()
i = 0
for id in ids:
    details = np.genfromtxt("test/detailed_results_"+str(id)+".txt", usecols=(n_eff_col), skip_header=5, skip_footer=5)
    if (details[freq_row]) >= float(num_iters) / 10:
        #print np.where(ids == id)[0][0]
        #indices.append(np.where(ids == id)[0][0])
        indices.append(i)
    i += 1

indices = np.asarray(indices)
print float(len(indices)) / len(ids)
print len(indices)

f_orig = dat[indices, 1]
f_ls_orig = dat[indices, 2]
f_bglst_orig = dat[indices, 3]
f_gp_orig = dat[indices, 4]
l_orig = dat[indices, 6]
l_gp_orig = dat[:, 7]
sig_var_orig = dat[indices, 8]
sig_var_gp_orig = dat[indices, 9]
trend_var_orig = dat[indices, 10]
trend_var_gp_orig = dat[indices, 11]

coh_lens = np.arange(0.6, 5, 0.1)
quality1 = np.ones(len(coh_lens))
std1 = np.zeros(len(coh_lens))
quality2 = np.ones(len(coh_lens))
std2 = np.zeros(len(coh_lens))
quality31 = np.ones(len(coh_lens))
quality32 = np.ones(len(coh_lens))
std31 = np.zeros(len(coh_lens))
std32 = np.zeros(len(coh_lens))

#quality41 = abs(f_orig-f_ls_orig)/f_orig        
#quality42 = abs(f_orig-f_gp_orig)/f_orig
#indices41 = np.where(quality41 < 1)
#indices42 = np.where(quality42 < 1)

i = 0
for coh_len in coh_lens:
    #indices = np.where(np.abs(f-f_gp) < np.abs(f-f_ls))
    #indices = np.arange(0, len(f))
    indices = np.where(l_orig < np.ones(len(f_orig))/f_orig*coh_len)
    if len(indices[0]) > 0:
        print len(indices[0])
        #indices = np.intersect1d(indices1, indices2)    
        
        f = f_orig[indices]
        f_ls = f_ls_orig[indices]
        f_bglst = f_bglst_orig[indices]
        f_gp = f_gp_orig[indices]
        l = l_orig[indices]
        l_gp = l_gp_orig[indices]
        sig_var = sig_var_orig[indices]
        #print np.shape(np.where(np.abs(f-f_gp) < np.abs(f-f_ls))), len(f)
        #print np.mean((l_gp - l)/l)
    
        (a, b) = mw_utils.estimate_with_se(np.column_stack((f,f_bglst,f_gp)), lambda x: sum(abs(x[:,0]-x[:,1])/x[:,0])/sum(abs(x[:,0]-x[:,2])/x[:,0]))
        quality1[i] = a#sum(abs(f-f_ls)/f)/sum(abs(f-f_gp)/f)
        std1[i] = b#sum(abs(f-f_ls)/f)/sum(abs(f-f_gp)/f)
        
        
        diff = abs(f-f_bglst) - abs(f-f_gp)
        
        (a, b) = mw_utils.estimate_with_se(diff, lambda diff: float(len(np.where(diff > 0)[0]))/len(indices[0]))
        quality2[i] = a#float(len(np.where(diff > 0)[0]))/len(indices[0])
        std2[i] = b#float(len(np.where(diff > 0)[0]))/len(indices[0])
        
        indices1 = np.where(sig_var > 0)
        f = f[indices1]
        f_ls = f_ls[indices1]
        f_bglst = f_bglst[indices1]
        f_gp = f_gp[indices1]
        
        (a, b) = mw_utils.mean_with_se(abs(f-f_bglst)/f)
        
        quality31[i] = a
        std31[i] = b

        (a, b) = mw_utils.mean_with_se(abs(f-f_gp)/f)
        quality32[i] = a
        std32[i] = b
        
    i += 1    
    

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
fig.set_size_inches(6, 10)
#fig.tight_layout(pad=2.5)

ax1.text(0.95, 0.9,'(a)', horizontalalignment='center', transform=ax1.transAxes, fontsize=panel_label_fs)
ax1.set_ylabel(r'$\overline{\Delta}_{\rm H}/\overline{\Delta}_{\rm GP}$', fontsize=axis_label_fs)#,fontsize=20)
ax2.text(0.95, 0.9,'(b)', horizontalalignment='center', transform=ax2.transAxes, fontsize=panel_label_fs)
ax2.set_ylabel(r'# $\Delta_{\rm GP} < \Delta_{\rm H}$', fontsize=axis_label_fs)#,fontsize=20)
ax3.text(0.95, 0.9,'(c)', horizontalalignment='center', transform=ax3.transAxes, fontsize=panel_label_fs)

ax1.plot(coh_lens, quality1, 'k-')
ax1.fill_between(coh_lens, quality1 + std1, quality1 - std1, alpha=0.1, facecolor='gray', interpolate=True)
ax2.plot(coh_lens, quality2, 'k-')
ax2.fill_between(coh_lens, quality2 + std2, quality2 - std2, alpha=0.1, facecolor='gray', interpolate=True)
handles1 = ax3.plot(coh_lens, quality31, 'b--', alpha=0.9)
handles2 = ax3.plot(coh_lens, quality32, 'r-', alpha=0.9)
ax3.fill_between(coh_lens, quality31 + std31, quality31 - std31, alpha=0.1, facecolor='lightblue', interpolate=True)
ax3.fill_between(coh_lens, quality32 + std32, quality32 - std32, alpha=0.1, facecolor='lightsalmon', interpolate=True)

ax3.legend(handles1 + handles2, ["Harmonic", "Quasiperiodic GP"],
            numpoints = 1,
            scatterpoints=1,
            bbox_to_anchor=(0., 0.8, 1., .1),
            #loc='upper right', 
            ncol=1,
            fontsize=10, labelspacing=0.7)

ax3.set_ylabel(r'$\overline{\Delta}$', fontsize=axis_label_fs)#,fontsize=20)
ax3.set_xlabel(r'$\ell/P_{\rm true}$', fontsize=axis_label_fs)#,fontsize=20)

fig.savefig("exp_diag.pdf")


#fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=False)
#fig.set_size_inches(6, 8)

#ax1.text(0.9, 0.9,'(a)', horizontalalignment='center', transform=ax1.transAxes)
#ax2.text(0.9, 0.9,'(b)', horizontalalignment='center', transform=ax2.transAxes)

#ax1.scatter(f_orig[indices41], quality41[indices41], color = 'b', s=1)
#ax1.scatter(f_orig[indices42], quality42[indices42], color = 'r', s=1)
#ax2.scatter(sig_var_orig[indices41], quality41[indices41], color = 'b', s=1)
#ax2.scatter(sig_var_orig[indices42], quality42[indices42], color = 'r', s=1)
#fig.savefig("test/diagnostics2.png")
