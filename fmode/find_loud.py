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

def get_lon_lat(lon_index, lat_index, num_lon, num_lat, header):
    min_lon = header["MIN_LON"]
    max_lon = header["MAX_LON"]
    min_lat = header["MIN_LAT"]
    max_lat = header["MAX_LAT"]
    patch_size = header["PATCH_SZ"]
    #print(min_lon, max_lon, min_lat, max_lat)
    lon_step = (max_lon - min_lon)/ (num_lon - 1)
    lat_step = (max_lat - min_lat)/ (num_lat - 1)
    
    start_lon = min_lon + lon_step*lon_index
    start_lat = min_lat + lat_step*lat_index
    
    return start_lon, start_lon + patch_size, start_lat, start_lat + patch_size

def get_last():
    time = ""
    time2 = ""
    lon_index = None
    lat_index = None
    if os.path.isfile("loud.txt") and os.path.getsize("loud.txt") > 0:
        with open("loud.txt", "r") as file:
            line = file.readlines()[-1]
            #file.seek(-2, os.SEEK_END)
            #while file.read(1) != b'\n':
            #    file.seek(-2, os.SEEK_CUR) 
            #line = file.readline().decode()
            print("Last loud patch", line)
    
            date, time, _, _, _, _, _, lon_index, lat_index, _, _ = line.split(",")
            time = date + "_" + time
            time2 = date + "," + time
    return time, time2, lon_index, lat_index

if (__name__ == '__main__'):
    
    input_path = 'output'
    output_path = '.'
    
    indices = None

    time, time2, _, _ = get_last()
    all_files = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file[-5:] == ".fits":
                all_files.append(file)
    all_files.sort()
    for file in all_files:
            if file[:len(time)] >= time:
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
                        max_index = np.unravel_index(np.argmax(mean), mean.shape)
                        #print("max_index", max_index)
                        with FileLock("loud.lock"):
                            start_time = hdul[i].header["START_T"]
                            start_time = start_time[:10] + "," + start_time[11:]
                            time, time2, last_lon_index, last_lat_index = get_last()
                            with open("loud.txt", "a") as f:
                                lon_index, lat_index = max_index
                                if start_time > time2 or (start_time == time2 and (lon_index > last_lon_index or lon_index == last_lon_index and lat_index > last_lat_index)):
                                    start_lon, end_lon, start_lat, end_lat = get_lon_lat(lon_index, lat_index, mean.shape[0], mean.shape[1], hdul[i].header)
                                    carr_lon = hdul[i].header["CARR_LON"]
                                    f.write(f"{start_time},{carr_lon},{start_lon},{end_lon},{start_lat},{end_lat},{lon_index},{lat_index},{mean[lon_index, lat_index]},{std[lon_index, lat_index]}\n")
                        
                    hdul.close()
                except Exception as e:
                    print(e)
    
