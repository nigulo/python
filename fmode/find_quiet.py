import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))
sys.path.append('..')
import matplotlib as mpl

import numpy as np
import numpy.random as random
import time
import os.path
from astropy.io import fits
import track

import astropy.units as u
from astropy.coordinates import SkyCoord
import sunpy.coordinates
from datetime import datetime, timedelta

from filelock import FileLock



if (__name__ == '__main__'):
    
    input_path = 'output'
    output_path = '.'
    quiet_level = 6
    
    i = 1
    if len(sys.argv) > i:
        quiet_level = float(sys.argv[i])
        
    indices = None

    time = ""
    if os.path.isfile("quiet.txt"):      
        with open("quiet.txt", "rb") as file:
            file.seek(-2, os.SEEK_END)
            while file.read(1) != b'\n':
                file.seek(-2, os.SEEK_CUR) 
            line = file.readline().decode()
            print("Last quiet patch", line)
    
            time, _, _, _, _ = line.split()
    #year, month, day, hrs, mins, secs = track.parse_t_rec(time)
    #start_time = datetime(int(year), int(month), int(day), int(hrs), int(mins), int(secs))
    
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file[-5:] == ".fits" and file[:len(time)] >= time:
                try:
                    hdul = fits.open(input_path + "/" + file)
                    for i in range(len(hdul)):
                        
                        mean = hdul[i].data[4, :, :]
                        std = hdul[i].data[5, :, :]
                        print("mean", mean)
                        print("std", std)
                        
                        if indices is None:
                            lon_inds = np.arange(mean.shape[0])
                            lat_inds = np.arange(mean.shape[1])
                            indices = np.transpose([np.repeat(lon_inds, len(lat_inds)), np.tile(lat_inds, len(lon_inds))])
                            #print("Indices", indices)
                        quiet_indices = indices[mean + std <= quiet_level]
                        print("Quiet indices", quiet_indices)
                        with open("quiet.txt", "a+") as f:
                            start_time = hdul[i].header["START_T"]
                            for lon_index, lat_inde in quiet_indices:
                                f.write(f"{start_time} {lon_index} {lat_index} {mean[lon_index, lat_index]} {std[lon_index, lat_index]}\n")
                        
                    hdul.close()
                except:
                    # Corrupted fits file
                    pass
    