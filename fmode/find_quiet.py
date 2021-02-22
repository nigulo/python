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


def get_last():
    time = ""
    time2 = ""
    lon_index = None
    lat_index = None
    if os.path.isfile("quiet.txt") and os.path.getsize("quiet.txt") > 0:
        with open("quiet.txt", "rb") as file:
            file.seek(-2, os.SEEK_END)
            while file.read(1) != b'\n':
                file.seek(-2, os.SEEK_CUR) 
            line = file.readline().decode()
            print("Last quiet patch", line)
    
            date, time,  lon_index, lat_index, _, _ = line.split()
            time = date + "_" + time
            time2 = date + " " + time
    return time, time2, lon_index, lat_index

if (__name__ == '__main__'):
    
    input_path = 'output'
    output_path = '.'
    quiet_level = 6
    
    i = 1
    if len(sys.argv) > i:
        quiet_level = float(sys.argv[i])
    
    print("Quiet level", quiet_level)    
    indices = None

    time, time2, _, _ = get_last()
    
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file[-5:] == ".fits" and file[:len(time)] >= time:
                try:
                    hdul = fits.open(input_path + "/" + file)
                    for i in range(len(hdul)):
                        mean = hdul[i].data[4, :, :]
                        std = hdul[i].data[5, :, :]
                        #print("mean", mean)
                        #print("std", std)
                        
                        if indices is None:
                            lon_inds = np.arange(mean.shape[0])
                            lat_inds = np.arange(mean.shape[1])
                            indices = np.reshape(np.transpose([np.repeat(lon_inds, len(lat_inds)), np.tile(lat_inds, len(lon_inds))]), (len(lon_inds), len(lat_inds), 2))
                            #print("Indices", indices)
                        #print(mean + std <= quiet_level)
                        quiet_indices = indices[mean + std <= quiet_level]
                        #print("Quiet indices", quiet_indices)
                        with FileLock("quiet.txt"):
                            start_time = hdul[i].header["START_T"]
                            time, time2, last_lon_index, last_lat_index = get_last()
                            with open("quiet.txt", "a+") as f:
                                for lon_index, lat_index in quiet_indices:
                                    if start_time > time2 or (start_time == time2 and (lon_index > last_lon_index or lon_index == last_lon_index and lat_index > last_lat_index)):
                                        f.write(f"{start_time} {lon_index} {lat_index} {mean[lon_index, lat_index]} {std[lon_index, lat_index]}\n")
                        
                    hdul.close()
                except Exception as e:
                    print(e)
                    # Corrupted fits file
                    pass
    
