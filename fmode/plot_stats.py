#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))
sys.path.append('..')
import matplotlib as mpl

import numpy as np

import os.path
from astropy.io import fits
import plot
import misc


path = "output"
path2 = "../../fits"
all_files = list()


for root, dirs, files in os.walk(path):
    for file in files:
        all_files.append(file)
all_files.sort()

stats = None
stats2 = None
n = 0
for file in all_files[::24]:
    try:
        hdul = fits.open(path + "/" + file)
        date = file[:10]
        hdul2 = fits.open(path2 + "/" + date + "_hmi.M_720s.fits")
        time = file[11:19]
        for i in range(1, len(hdul2)):
            if hdul2[i].header['T_REC'][11:19] == time:
                input_data = hdul2[i].data
                max_val = np.nanmax(input_data)
                min_val = np.nanmin(input_data)
                fltr = np.isnan(input_data)
                input_data[fltr] = max_val

                input_data = misc.sample_image(input_data, 0.25)

                fltr = input_data > .9*max_val
                input_data[fltr] = np.nan
                fltr = input_data < .9*min_val
                input_data[fltr] = np.nan

                break
        if len(hdul) > 0:
            if stats is None:
                stats = np.zeros_like(hdul[0].data)
                stats2 = np.zeros_like(hdul[0].data)
            for i in range(len(hdul)):
                stats += hdul[i].data
                stats2 += hdul[i].data**2
                n += 1
                n_cols = hdul[i].data.shape[0]//2
                test_plot = plot.plot(nrows=3, ncols=n_cols, size=plot.default_size(hdul[i].data.shape[1]*10, hdul[i].data.shape[2]*10))
                row = 0
                col = 0
                for j in range(hdul[i].data.shape[0]):
                    test_plot.colormap(hdul[i].data[j, ::-1, ::-1].T, ax_index=[row, col], show_colorbar=True)
                    col += 1
                    if col == n_cols:
                        col = 0
                        row += 1
                for col in range(n_cols):
                    test_plot.colormap(input_data, ax_index=[2, col], cmap="bwr", show_colorbar=True, vmin=-100, vmax=100)
                test_plot.save(f"stats_{file}.png")
                test_plot.close()
        hdul.close()
        hdul2.close()
    except Exception as e:
        print(e)
        pass
            
mean = stats/n
sq_mean = stats2/n
std = np.sqrt(sq_mean - mean**2)

for i in range(stats.shape[0]):
    test_plot = plot.plot(nrows=1, ncols=2, size=plot.default_size(stats.shape[1]*10, stats.shape[2]*10))
    test_plot.colormap(mean[i, ::-1, ::-1].T, ax_index=0, show_colorbar=True)
    test_plot.colormap(std[i, ::-1, ::-1].T, ax_index=1, show_colorbar=True)
    test_plot.save(f"stats{i}.png")
    test_plot.close()

            
