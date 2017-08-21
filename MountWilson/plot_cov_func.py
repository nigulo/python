# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 20:20:51 2017

@author: nigul
"""

import numpy as np
import matplotlib.pyplot as plt

freq = 1.0
sig_var = 1.0
start_time = -10.0
end_time = 10.0
count = 10000
time_step = (end_time-start_time) / count
dt = np.linspace(start_time, end_time, count)
cov1 = sig_var * np.exp(-0.5 * freq * freq * pow(dt,2)) * np.cos(2*np.pi*freq*dt)
cov_11 = cov1[0:count/2]
cov_12 = cov1[count/2:]
spec1 = np.fft.rfft(np.concatenate([cov_12, cov_11]))

cov2 = sig_var * np.cos(2*np.pi*freq*dt)
cov_21 = cov2[0:count/2]
cov_22 = cov2[count/2:]
spec2 = np.fft.rfft(np.concatenate([cov_22, cov_21]))

freqs = np.fft.rfftfreq(count, d=time_step)
#freqs = np.fft.fftshift(freqs)


fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=False)
fig.set_size_inches(6, 8)
#fig.tight_layout(pad=2.5)

ax1.text(0.95, 0.9,'(a)', horizontalalignment='center', transform=ax1.transAxes)
ax1.set_ylabel(r'Covariance')#,fontsize=20)
ax2.text(0.95, 0.9,'(b)', horizontalalignment='center', transform=ax2.transAxes)
ax2.set_ylabel(r'Power')#,fontsize=20)

ax1.plot(dt, cov1, 'b-')
ax1.plot(dt, cov2, 'r--')



ax2.plot(freqs, np.real(spec1)/max(np.real(spec2)), 'b-')
ax2.plot(freqs, np.real(spec2)/max(np.real(spec2)), 'r--')

ax1.set_xlabel(r'Time lag')#,fontsize=20)
ax1.set_xlim(-5, 5)
ax2.set_xlabel(r'Frequency')#,fontsize=20)
ax2.set_xlim(0, 5)
ax2.set_ylim(0, 1)

fig.savefig("cov_func.eps")


#fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=False)
#fig.set_size_inches(6, 8)

#ax1.text(0.9, 0.9,'(a)', horizontalalignment='center', transform=ax1.transAxes)
#ax2.text(0.9, 0.9,'(b)', horizontalalignment='center', transform=ax2.transAxes)

#ax1.scatter(f_orig[indices41], quality41[indices41], color = 'b', s=1)
#ax1.scatter(f_orig[indices42], quality42[indices42], color = 'r', s=1)
#ax2.scatter(sig_var_orig[indices41], quality41[indices41], color = 'b', s=1)
#ax2.scatter(sig_var_orig[indices42], quality42[indices42], color = 'r', s=1)
#fig.savefig("test/diagnostics2.png")
