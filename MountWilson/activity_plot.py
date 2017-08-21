# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 12:58:15 2016

@author: nigul
"""

import numpy as np
import matplotlib.pyplot as plt

dat = np.loadtxt("periods_cycles2.txt", delimiter='\t')
dat=dat[np.where(dat[:,0] > 0)]
dat=dat[np.where(dat[:,1] < 25)]

cycles1 = dat[:,0:2]
cycles2 = dat[:,0:3]
cycles3 = dat[:,0:4]
cycles4 = dat[:,0:5]
cycles2 = cycles2[np.where(cycles2[:,2] > 0)]
cycles3 = cycles3[np.where(cycles3[:,3] > 0)]
cycles4 = cycles4[np.where(cycles4[:,4] > 0)]

print np.shape(dat)

fig = plt.gcf()
fig.set_size_inches(18, 6)
#plt.plot(1/cycles1[:,0], cycles1[:,1]/cycles1[:,0], 'b+')
#plt.plot(1/cycles2[:,0], cycles2[:,2]/cycles2[:,0], 'rx')
#plt.loglog()
plt.plot(cycles1[:,0], cycles1[:,1], 'b+')
plt.plot(cycles2[:,0], cycles2[:,2], 'rx')
plt.plot(cycles3[:,0], cycles3[:,3], 'go', fillstyle='none')
plt.plot(cycles4[:,0], cycles4[:,4], 'y^', fillstyle='none')
#plt.loglog()
plt.xlabel("Period [d]")
plt.ylabel("Cycle length [yr]")
plt.savefig("activity.png")
plt.close()
