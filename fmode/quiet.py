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

class quiet_sun:
    
    def __init__(self):
        self.path = '.'
        start_date = '2013.02.14'
        self.num_days = -1
    
        i = 1
        
        if len(sys.argv) > i:
            self.path = sys.argv[i]
        i += 1
        if len(sys.argv) > i:
            start_date = sys.argv[i]
        i += 1
        if len(sys.argv) > i:
            self.num_days = int(sys.argv[i])
        
        
        self.x = 0
        self.y = 0
        self.nx = 100
        self.ny = 100
                
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
            self.current_day += 1
            self.data = np.empty((len(hdul) - 1, self.nx, self.ny), dtype=np.int16)
            for i in np.arange(1, len(hdul)):
                self.data[i-1] = fits.getdata(file, i)[self.x:self.x+self.nx, self.y:self.y+self.ny]
                print(i)
            print(self.data.shape)
            hdul.close()
        else:
            raise "No more files"
    
    def is_quiet(self, t):
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
                    if not is_quiet(data[t0]):
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

        
        