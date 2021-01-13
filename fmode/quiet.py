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

def center_and_radius(snapshot):
    nx_full, ny_full = snapshot.shape
    xc = nx_full//2
    yc = ny_full//2
    for xl in np.arange(nx_full):
        if not np.isnan(snapshot[xl, yc]):
            break
    for xr in np.arange(nx_full - 1, 0, -1):
        if not np.isnan(snapshot[xr, yc]):
            break
    for yb in np.arange(ny_full):
        if not np.isnan(snapshot[xc, yb]):
            break
    for yt in np.arange(ny_full - 1, 0, -1):
        if not np.isnan(snapshot[xc, yt]):
            break
    xc = (xl + xr)/2
    yc = (yb + yt)/2
    r = ((xr - xl) + (yt - yb))/4
    print(xl, xr, yb, yt)
    print(xc, yc, r)
    return xc, yc, r

class quiet_sun:
    
    def __init__(self):
        self.path = '.'
        start_date = '2013.02.14'
        self.num_days = -1
        
        self.lat = 0
        self.long = 0
        self.size = 15
        self.num_hrs = 8 # The duration of quet segment requested
    
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
        i += 1
        if len(sys.argv) > i:
            self.num_hrs = float(sys.argv[i])
        

        
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
            full_snapshot = fits.getdata(file, 1)
            xc, yc, r = center_and_radius(full_snapshot)
            test_plot = plot.plot(nrows=1, ncols=1)
            test_plot.colormap(full_snapshot, cmap_name="bwr")
            test_plot.save(f"full_snapshot.png")
            test_plot.close()
            
            snapshots_per_day = len(hdul) - 1
            f = 1./snapshots_per_day
            self.nt = int(snapshots_per_day*self.num_hrs/24)
            #nx_full, ny_full = hdul[1].shape
            coef = r/90.
            x = xc + (self.long - self.size/2)*coef
            nx = int(coef*self.size)
            #ny = int(y_coef*self.size)
            print(nx)
            self.current_day += 1
            self.data = np.empty((len(hdul) - 1, nx, nx), dtype=np.float32)
            for i in np.arange(1, len(hdul)):
                lat = self.lat + self.size/2
                y = lat*coef
                for j in np.arange(nx):
                    lat = y/coef
                    print(lat)
                    x_shift = -diff_rot(lat)*(i-1)*f*coef
                    x1 = int(x + x_shift)
                    y1 = int(y + yc)
                    self.data[i-1, j] = fits.getdata(file, i)[y1, x1:x1+nx]
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

        
        