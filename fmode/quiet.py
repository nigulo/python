import sys
sys.path.append('../utils')
sys.path.append('..')
import config
import matplotlib as mpl

mpl.use('Agg')
mpl.rcParams['figure.figsize'] = (20, 30)
#import pickle
import numpy as np
import numpy.random as random
import time
import os.path
from astropy.io import fits

def fetch_next(t):
    for file in all_files:
        if condition:
            hdul = fits.open(file)
            data = hdul[0].data
            hdul.close()
            return data
    raise "No more files"

def is_quiet(data_at_t):
    return True

if (__name__ == '__main__'):

    path = '.'

    i = 1
    
    if len(sys.argv) > 1:
        path = sys.argv[i]
    i += 1
    
    x = 0
    y = 0
    nx = 100
    ny = 100
    
    t_start = 0
    t_end = 100
    
    nt = 10 # The duration of quet segment requested
    
    all_files = list()
    quiet_times = list()
    
    for root, dirs, files in os.walk(path)
        all_files.extend(files)

    t = t_start
    data = fetch_next(t)
    t0 = 0
    while t < t_end:
        try:
            quiet = True
            for t1 in np.arange(t, t + nt):
                if t0 >= len(data):
                    data = fetch_next(t)
                    t0 = 0                
                if not is_quiet(data[t0]):
                    quiet = False
                    t = t1 + 1
                    break
                t0 += 1
            if quiet:
                quiet_times.append(t)
                t += nt
        except:
            break
        
        