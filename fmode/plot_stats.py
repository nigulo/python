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


path = "output"
all_files = list()


for root, dirs, files in os.walk(path):
    for file in files:
        all_files.append(file)
all_files.sort()

stats = None
stats2 = None
n = 0
for file in all_files:
    hdul = fits.open(path + "/" + file)
    if len(hdul) > 0:
        if stats is None:
            stats = np.zeros_like(hdul[0].data)
            stats2 = np.zeros_like(hdul[0].data)
        for i in range(len(hdul)):
            stats += hdul[i].data
            stats2 += hdul[i].data**2
            n += 1
            
mean = stats/n
sq_mean = stats2/n
std = np.sqrt(sq_mean - mean**2)

for i in range(stats.shape[2]):
    test_plot = plot.plot(nrows=1, ncols=1, size=plot.default_size(stats.shape[0]//8, stats.shape[1]//8))
    test_plot.colormap(stats[:, :, i], show_colorbar=True)
    test_plot.save(f"stats{i}.png")
    test_plot.close()

            