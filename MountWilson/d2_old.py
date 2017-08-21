# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 12:58:15 2016

@author: nigul
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import LombScargle
import os
import os.path

p_value = 0.01
max_iters = 100

for root, dirs, files in os.walk("d2_res"):
    for file in files:
        if file[-8:] == "_min.csv":
            star = file[:-8]
            if "_" in star:
                continue
            data = np.loadtxt("detrended/"+star+".dat", usecols=(0,1))
            t = data[:,0]
            y = data[:,1]
            n = len(y)
            var = np.var(y)

            data = np.loadtxt("d2_res/"+file, usecols=(0,1))
            freqs = data[:,0]
            disps = data[:,1]

            z0 = -np.log(1-np.power(1-p_value, 1.0/len(freqs))) / n

            fig = plt.gcf()
            fig.set_size_inches(18, 6)
            plt.plot(freqs, np.ones(len(freqs)) - disps, 'b-', freqs, np.ones(len(freqs))*z0, 'k--')

            best_freqs = list()
            min_disps = list()

            min_disp_ind = np.argmin(disps[1:-1])
            if min_disp_ind >= 0:
                min_disp = disps[min_disp_ind+1]
                if min_disp < 1-z0:
                    best_freq = freqs[min_disp_ind+1]
                    best_freqs.append(best_freq)
                    min_disps.append(min_disp)

                    y_fit = LombScargle(t, y).model(t, best_freq)
                    y -= y_fit
                    y -= np.mean(y)
                    detrended = np.column_stack((t, y))
                    np.savetxt("detrended/" + star + "_0.dat", detrended, fmt='%f')

            i = 0

            while os.path.isfile("d2_res/"+star + "_" + str(i)+"_min.csv"):
                data = np.loadtxt("detrended/"+star + "_" + str(i)+".dat", usecols=(0,1))
                t = data[:,0]
                y = data[:,1]
                n = len(y)
                var = np.var(y)

                data = np.loadtxt("d2_res/"+star + "_" + str(i)+"_min.csv", usecols=(0,1))
                freqs = data[:,0]
                disps = data[:,1]

                z0 = -np.log(1-np.power(1-p_value, 1.0/len(freqs))) / n

                min_disp_ind = np.argmin(disps[1:-1])
                if min_disp_ind >= 0:
                    min_disp = disps[min_disp_ind+1]
                    if min_disp < 1-z0:
                        best_freq = freqs[min_disp_ind+1]
                        best_freqs.append(best_freq)
                        min_disps.append(min_disp)
                        y_fit = LombScargle(t, y).model(t, best_freq)
                        y -= y_fit
                        y -= np.mean(y)
                        detrended = np.column_stack((t, y))
                        np.savetxt("detrended/" + star + "_" + str(i+1) + ".dat", detrended, fmt='%f')
                i += 1

            if len(best_freqs) > 0:
                plt.stem(best_freqs, np.ones(len(best_freqs)) - min_disps)

            plt.xlabel("Frequency")
            plt.ylabel("Dispersion")
            plt.savefig("d2_res/" + star + ".png")
            plt.close()
