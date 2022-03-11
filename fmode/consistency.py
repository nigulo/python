import sys
import os
from scipy.io import readsav
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))
import plot

stats = []
for root, dirs, files in os.walk("data_full"):
    for file in files:
        time_index = file[11:13]
        dat = readsav(os.path.join(root, file), idict=None, python_dict=False, uncompressed_file_name=None, verbose=False)
        levels = np.linspace(np.min(np.log(dat.p_kyom_kx0))+2, np.max(np.log(dat.p_kyom_kx0))-2, 42)
        
        stats.append([float(time_index)/4, np.sum(dat.p_kyom_kx0)])
        '''
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.contour(dat.k_y, dat.nu, np.log(dat.p_kyom_kx0), levels=levels)
        
        x = np.asarray(dat.nu, dtype='float')
        k_index = np.min(np.where(dat.k_y >= 750)[0])
        # Average over 3 neigbouring slices to reduce noise
        y = np.log(dat.p_kyom_kx0[:, k_index-1] + dat.p_kyom_kx0[:, k_index] + dat.p_kyom_kx0[:, k_index+1]) - np.log(3)
        y -= np.mean(y)
        inds = np.where(x > 1.5)[0]
        x = x[inds]
        y = y[inds]
        inds = np.where(x < 8.)[0]
        x = x[inds]
        y = y[inds]
        '''
stats.sort()
stats = np.asarray(stats)
fig = plot.plot(nrows=1, ncols=1, size=plot.default_size(300, 200))
fig.plot(stats[:, 0], stats[:, 1], "k-")
fig.save(f"integral.png")
