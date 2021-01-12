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
import plot

A = 14.713
B = -2.396
C = -1.787

def diff_rot(lat):
    lat *= np.pi / 180
    return A + B*np.sin(lat)**2 + C*np.sin(lat)**4

class quiet_sun:
    
    def __init__(self):
        self.path = '.'
        start_date = '2013.02.14'
        self.num_days = -1
        
        self.lat = 0
        self.long = 0
        self.size = 15
    
        i = 1
        
        if len(sys.argv) > i:
            self.path = sys.argv[i]
        i += 1
        if len(sys.argv) > i:
            start_date = sys.argv[i]
        i += 1
        if len(sys.argv) > i:
            self.num_days = int(sys.argv[i])
        i += 1
        if len(sys.argv) > i:
            self.lat = float(sys.argv[i])
        i += 1
        if len(sys.argv) > i:
            self.long = float(sys.argv[i])
        i += 1
        if len(sys.argv) > i:
            self.size = float(sys.argv[i])
        

        self.nt = 10 # The duration of quet segment requested
        
        self.all_files = list()
        self.quiet_times = list()
        
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file >= start_date:
                    self.all_files.append(file)
            #self.all_files.extend(files)
        self.all_files.sort()
        self.current_day = 0

    def fetch_next(self):
        if self.num_days <= 0 or self.current_day <= self.num_days:    
            file = self.path + "/" + self.all_files[self.current_day]
            hdul = fits.open(file)
            hdul.info()
            f = 1./(len(hdul) - 1)
            nx_full, ny_full = hdul[1].shape
            x_coef = nx_full/180.
            y_coef = ny_full/180.
            x = nx_full // 2 + (self.long - self.size/2)*x_coef
            nx = int(x_coef*self.size)
            ny = int(y_coef*self.size)
            print(nx, ny)
            self.current_day += 1
            self.data = np.empty((len(hdul) - 1, nx, ny), dtype=np.float32)
            for i in np.arange(1, len(hdul)):
                lat = self.lat + self.size/2
                y = lat*y_coef
                for j in np.arange(ny):
                    lat = y/y_coef
                    print(lat)
                    x_shift = diff_rot(lat)*i*f*x_coef
                    x1 = int(x + x_shift)
                    y1 = int(y + ny_full // 2)
                    self.data[i-1, :, j] = fits.getdata(file, i)[x1:x1+nx, y1]
                    y -= 1
                test_plot = plot.plot(nrows=1, ncols=1)
                test_plot.colormap(self.data[i-1], cmap_name="bwr")
                test_plot.save(f"test{i-1}.png")
                test_plot.close()
                print(i)
            print(self.data.shape)
            hdul.close()
        else:
            raise "No more files"
    
    def is_quiet(self, t0):        
        return True
    
    def search(self):
        self.fetch_next()
        t = 0
        t0 = 0
        while True:
            try:
                quiet = True
                for t1 in np.arange(t, t + self.nt):
                    if t0 >= len(data):
                        self.fetch_next()
                        t0 = 0                
                    if not is_quiet(t0):
                        quiet = False
                        t = t1 + 1
                        break
                    t0 += 1
                if quiet:
                    self.quiet_times.append(self.all_files[self.current_day] + ":" + str(t0))
                    t += self.nt
            except:
                break

if (__name__ == '__main__'):
    
    qs = quiet_sun()
    qs.search()

        
        