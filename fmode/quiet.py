import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))
sys.path.append('..')
import config
import matplotlib as mpl

import numpy as np
import numpy.random as random
import time
import os.path
from astropy.io import fits
import plot
import misc

import astropy.units as u
from astropy.coordinates import SkyCoord
import sunpy.coordinates
from sunpy.coordinates import frames

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d

A = 14.713
B = -2.396
C = -1.787

radius_km = 695700

def diff_rot(lat):
    #lat *= np.pi / 180
    return (A + B*np.sin(lat)**2 + C*np.sin(lat)**4)*np.pi/180

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

def _plot_segment(longs, lats, ax, cmap, alpha):
    phi, theta = np.meshgrid(longs*np.pi/180, (90-lats)*np.pi/180)
    X = np.sin(theta) * np.cos(phi)
    Y = np.sin(theta) * np.sin(phi)
    Z = np.cos(theta)
    plot = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cmap, linewidth=0, antialiased=False, alpha=alpha)

def plot_segment(longs, lats):
    longs_all = np.linspace(0, 360, 100)
    lats_all = np.linspace(-90, 90, 50)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    _plot_segment(longs_all, lats_all, ax, None, alpha=0.2)#plt.get_cmap('Greys'))
    _plot_segment(longs, lats, ax, plt.get_cmap('jet'), alpha=0.8)
    fig.savefig("segment.png")
    plt.close(fig)

def do_plot(xs, ys):
    print(xs.shape, ys.shape)
    test_plot = plot.plot(nrows=1, ncols=1, size=plot.default_size(100, 100))
    r = 900.
    phi = np.linspace(0, np.pi*2, 100)
    xs1 = r*np.cos(phi)
    ys1 = r*np.sin(phi)
    test_plot.plot(xs1, ys1, params="k-")
    test_plot.plot(xs, ys, params="k.")
    test_plot.save(f"segment_ortho.png")
    test_plot.close()


class quiet_sun:
    
    def __init__(self):
        self.path = '.'
        start_date = '2013-02-14'
        self.num_days = -1
        
        self.lat = 0
        self.long = 0
        self.size = 30
        self.num_hrs = 8 # The duration of quet segment requested
        self.track = True
    
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
        i += 1
        if len(sys.argv) > i:
            self.track = int(sys.argv[i])

        print(self.path)
        #self.lat *= np.pi/180
        #self.long *= np.pi/180
        #self.size *= np.pi/180
        
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
            day = self.all_files[self.current_day][:10]            
            file = self.path + "/" + self.all_files[self.current_day]
            print(day)
            hdul = fits.open(file)
            from pprint import pprint
            pprint(vars(hdul))
            #hdul.info()
            print(self.track)
            full_snapshot = fits.getdata(file, 1)
            #test_plot = plot.plot(nrows=1, ncols=1)
            #indices = np.argwhere(np.isnan(full_snapshot))
            #print(indices)
            #full_snapshot[indices] = 0.
            #print(np.max(full_snapshot), np.min(full_snapshot))
            #test_plot.colormap(misc.trunc(full_snapshot, 1e-3), cmap_name="bwr")
            #test_plot.save(f"full_snapshot.png")
            #test_plot.close()
            xc, yc, r = center_and_radius(full_snapshot)
            
            snapshots_per_day = len(hdul) - 1
            f = 1./snapshots_per_day
            self.nt = int(snapshots_per_day*self.num_hrs/24)
            #nx_full, ny_full = hdul[1].shape
            #coef1 = r/90.
            #x = xc + (self.long - self.size/2)*coef

            #c = SkyCoord((self.long - self.size/2)*u.deg, (self.lat - self.size/2)*u.deg, frame=frames.Helioprojective, obstime="2013-02-14 00:00:00", observer="earth")
            
            lat1 = self.lat - self.size/2
            lat2 = lat1 + self.size
            
            long1 = self.long - self.size/2
            long2 = long1 + self.size
            num_long = int(r*self.size/90)
            num_lat = num_long
            longs = np.linspace(long1, long2, num_long)
            lats = np.linspace(lat1, lat2, num_lat)
            plot_segment(longs, lats)
            
            long_lat = np.transpose([np.tile(longs, len(lats)), np.repeat(lats, len(longs))])

            c0 = SkyCoord(long_lat[:, 0]*u.deg, long_lat[:, 1]*u.deg, frame=frames.HeliographicStonyhurst)
            hpc_out = sunpy.coordinates.Helioprojective(observer="earth", obstime=f"{day} 00:00:00")
            c1 = c0.transform_to(hpc_out)
            
            xs = c1.Tx.value
            ys = c1.Ty.value
            dists = c1.distance.value
            r_arcsec = np.arctan(radius_km / dists[0])*180/np.pi*3600
            coef = r/r_arcsec
            print(r_arcsec, coef)
            #xys = np.transpose([np.tile(xs, len(ys)), np.repeat(ys, len(xs))])
            nx = len(longs)
            ny = len(lats)
            do_plot(xs, ys)

            '''
            c0 = SkyCoord(long1*u.deg, lat1*u.deg, frame=frames.HeliographicStonyhurst)
            hpc_out = sunpy.coordinates.Helioprojective(observer="earth", obstime=f"{day} 00:00:00")
            c1 = c0.transform_to(hpc_out)
            
            x1 = c1.Tx.value
            y1 = c1.Ty.value
            
            c0 = SkyCoord(long2*u.deg, lat2*u.deg, frame=frames.HeliographicStonyhurst)
            print(c0)
            hpc_out = sunpy.coordinates.Helioprojective(observer="earth", obstime=f"{day} 00:00:00")
            c1 = c0.transform_to(hpc_out)
            
            x2 = c1.Tx.value
            y2 = c1.Ty.value
            
            nx = int((x2-x1)*coef)
            ny = int((y2-y1)*coef)

            print(x1, x2, y1, y2)
            '''
            
            print(nx, ny)
            self.current_day += 1
            self.data = np.empty((len(hdul) - 1, ny, nx), dtype=np.float32)
            for i in np.arange(1, len(hdul)):
                
                hrs = (i - 1) *24/ (len(hdul) - 1)
                mins = (hrs - int(hrs))*60
                hrs = int(hrs)
                secs = int((mins - int(mins))*60)
                mins = int(mins)
                mins1 = secs/60
                mins = int(mins)
                data = fits.getdata(file, i)
                obstime = f"{day} {hrs}:{mins}:{secs}"
                print(obstime)
                
                #ys = np.linspace(y1, y2, ny)
                #xs = np.linspace(x1, x2, nx)
                
                #xys = np.transpose([np.tile(xs, len(ys)), np.repeat(ys, len(xs))])

                c1 = SkyCoord(xs*u.arcsec, ys*u.arcsec, frame=frames.Helioprojective, obstime=f"{day} 00:00:00", observer="earth")
                c2 = c1.transform_to(frames.HeliographicCarrington)
                c3 = SkyCoord(c2.lon, c2.lat, frame=frames.HeliographicCarrington, obstime=obstime, observer="earth")
                c4 = c3.transform_to(frames.Helioprojective)
                #print(c4)
                
                x_pix = (c4.Tx.value*coef + xc).astype(int)
                y_pix = (c4.Ty.value*coef + yc).astype(int)
                #print(x_pix)
                
                l = 0
                for j in np.arange(ny):
                    #y = y1+(y2-y1)*j/ny
                    
                    for k in np.arange(nx):
                        #print(c4.Tx[l], c4.Ty[l])
                        #x = x1+(x2-x1)*k/nx
                        #c1 = SkyCoord(x*u.arcsec, y*u.arcsec, frame=frames.Helioprojective, obstime=obstime, observer="earth")
                        #c2 = c1.transform_to(frames.HeliographicCarrington)
                        #c3 = SkyCoord(c2.lon, c2.lat, frame=frames.HeliographicCarrington, obstime=f"{day} 00:00:00", observer="earth")
                        #c4 = c3.transform_to(frames.Helioprojective)
                        
                        #x_ = c4.Tx[l].value
                        #y_ = c4.Ty[l].value
                        #l += 1
                        
                        #x_pix = int(x_*r/r_arcsec + xc)
                        #y_pix = int(y_*r/r_arcsec + yc)
                        #print(x_pix, y_pix)
                
                        self.data[i-1, j, k] = data[y_pix[l], x_pix[l]]
                        l += 1
                test_plot = plot.plot(nrows=1, ncols=1, size=plot.default_size(self.data[i-1].shape[1], self.data[i-1].shape[0]))
                test_plot.colormap(self.data[i-1], cmap_name="bwr")
                test_plot.save(f"test{i-1}.png")
                test_plot.close()
                print(i)
            print(self.data.shape)
            hdul.close()
            self.current_day += 1
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

        
        