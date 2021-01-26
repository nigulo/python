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
import floodfill

import astropy.units as u
from astropy.coordinates import SkyCoord
import sunpy.coordinates
from sunpy.coordinates import frames

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


class track:
    
    def __init__(self, path='.', start_date='2013-02-14', num_days=-1, num_frames=8*5, step=10):
        self.path = path
        self.start_date = start_date
        self.num_days = num_days
        self.num_frames = num_frames
        self.step = step
        
        self.cube = None

        print(self.path)
        
        self.all_files = list()
        self.quiet_times = list()
        
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file >= start_date:
                    self.all_files.append(file)
        self.all_files.sort()


    def save_stats(self):
        if self.stats is not None:
            output_file = f"{self.stats_time}.fits"
            self.stats.writeto(output_file, overwrite=True)
            self.stats = None

    def calc_stats(self):
        abs_data = np.abs(self.data)
        stats = np.array([np.mean(abs_data), np.std(abs_data))])
        cards = list()
        cards.append(fits.Card(keyword="TIME", value=self.get_obs_time(), comment="Observation time")
        header = fits.Header(cards)
        hdu = fitsImageHDU(data=stats, header=header, name='Statistics')
        if self.stats is None:
            self.stats = fits.HDUList()
            self.stats_time = self.get_obs_start_time()
        self.stats.append(hdu)
        if len(self.stats) >= self.num_frames:
            self.save_stats()
    
    def set_start_time(self):
        self.set_time()
        self.hrs_start = self.hrs
        self.mins_start = self.mins
        self.secs_start = self.secs
    
    def get_obs_start_time(self):
        return f"{self.day}_{self.hrs_start}__{self.mins_start}_{self.secs_start}"
            
    def set_time(self):
        hrs = self.frame_index*24/self.num_frames_per_day
        mins = (hrs - int(hrs))*60
        hrs = int(hrs)
        secs = int(round((mins - int(mins))*60))
        mins = int(mins)
        self.hrs = format(hrs, "02")
        self.mins = format(mins, "02")
        self.secs = format(secs, "02")
        
    def get_obs_time():
        return f"{self.day} {self.hrs}:{self.mins}:{self.secs}"

    def get_obs_time2():
        return f"{self.day}_{self.hrs}_{self.mins}_{self.secs}"
        
    def process_frames(self):
        if self.num_days <= 0 or self.day_index <= self.num_days:
            self.day = self.all_files[self.day_index][:10]            
            file = self.path + "/" + self.all_files[self.day_index]
            print(self.day)

            hdul = fits.open(file)
            print(hdul[1].header)

            self.num_frames_per_day = len(hdul) - 1
            self.set_time()
            
            coef_x = 1./hdul[1].header['CDELT2']
            coef_y = 1./hdul[1].header['CDELT1']
            xc = hdul[1].header['CRPIX2']
            yc = hdul[1].header['CRPIX1']
            sdo_lon1 = hdul[1].header['CRLN_OBS']
            sdo_lat1 = hdul[1].header['CRLT_OBS']
            sdo_dist = hdul[1].header['DSUN_OBS']
            #r_sun = hdul[1].header['RSUN_REF']
            observer_1 = frames.HeliographicStonyhurst(0.*u.deg, sdo_lat1*u.deg, radius=sdo_dist*u.m, obstime=self.get_obs_time())

            #full_snapshot = fits.getdata(file, 1)
            print(xc, yc)

            #r_arcsec = np.arctan(r_sun/sdo_dist)*180/np.pi*3600
            #r_pix = r_arcsec*coef_x
            
            #snapshots_per_day = len(hdul) - 1
            #f = 1./snapshots_per_day
            #self.nt = int(snapshots_per_day*self.num_hrs/24)

            #c0 = SkyCoord(long_lat[:, 0]*u.deg, long_lat[:, 1]*u.deg, frame=frames.HeliographicStonyhurst)
            #hpc_out = sunpy.coordinates.Helioprojective(observer=observer_1)#"earth", obstime=f"{day} 00:00:00")
            #c1 = c0.transform_to(hpc_out)
            

            arcsecs_per_pix_x = hdul[1].header['CDELT2']
            arcsecs_per_pix_y = hdul[1].header['CDELT1']
            #self.data = np.empty((len(hdul) - 1, ny, nx), dtype=np.float32)
            
            
            for i in np.arange(self.frame_index+1, min(self.num_frames, self.num_frames_per_day)+1):
                                
                self.set_time()
                
                obstime = self.get_obs_time()
                print(obstime)

                data = fits.getdata(file, i)

                a = hdul[i].header['CROTA2']*np.pi/180
                nx = hdul[i].header['NAXIS1']
                ny = hdul[i].header['NAXIS2']
                dx = hdul[i].header['CRVAL1']
                dy = hdul[i].header['CRVAL2']
                arcsecs_per_pix_x = hdul[i].header['CDELT1']
                arcsecs_per_pix_y = hdul[i].header['CDELT2']
                coef_x = 1./arcsecs_per_pix_x
                coef_y = 1./arcsecs_per_pix_y
                xc = hdul[i].header['CRPIX1']
                yc = hdul[i].header['CRPIX2']
                
                sin_a = np.sin(a)
                cos_a = np.cos(a)
                
                def pix_to_image(xs, ys):
                    #nx2 = (nx+1)/2
                    #ny2 = (ny+1)/2
                    #xc_arcsec = dx + arcsecs_per_pix_x*cos_a*(nx2 - xc) - arcsecs_per_pix_y*sin_a*(ny2 - yc)
                    #yc_arcsec = dy + arcsecs_per_pix_x*sin_a*(nx2 - xc) + arcsecs_per_pix_y*cos_a*(ny2 - yc)
                    xs = xs - xc
                    ys = ys - yc
                    
                    xs_arcsec = dx + arcsecs_per_pix_x*cos_a*xs - arcsecs_per_pix_y*sin_a*ys
                    ys_arcsec = dy + arcsecs_per_pix_x*sin_a*xs + arcsecs_per_pix_y*cos_a*ys
                                        
                    return xs_arcsec, ys_arcsec

                def image_to_pix(xs_arcsec, ys_arcsec):
                    xs_arcsec = xs_arcsec - dx
                    ys_arcsec = ys_arcsec - dy
                    #nx2 = (nx+1)/2
                    #ny2 = (ny+1)/2
                    #xc_arcsec = dx + arcsecs_per_pix_x*cos_a*(nx2 - xc) - arcsecs_per_pix_y*sin_a*(ny2 - yc)
                    #yc_arcsec = dy + arcsecs_per_pix_x*sin_a*(nx2 - xc) + arcsecs_per_pix_y*cos_a*(ny2 - yc)
                    
                    xs = xc + coef_x*cos_a*(xs_arcsec) + coef_y*sin_a*(ys_arcsec)
                    ys = yc - coef_x*sin_a*(xs_arcsec) + coef_y*cos_a*(ys_arcsec)
                                        
                    return xs, ys
                    
                print(coef_x, coef_y, xc, yc)
                
                xs = (np.arange(1, nx + 1)).astype(float)
                ys = (np.arange(1, ny + 1)).astype(float)
                xs_arcsec, ys_arcsec = pix_to_image(xs, ys)
    
    
                grid = np.transpose([np.tile(xs_arcsec, ny), np.repeat(ys_arcsec, nx)])
                

                sdo_lon = hdul[i].header['CRLN_OBS']
                sdo_lat = hdul[i].header['CRLT_OBS']
                sdo_dist = hdul[i].header['DSUN_OBS']
                #observer_i = frames.HeliographicStonyhurst((sdo_lon-sdo_lon1)*u.deg, sdo_lat*u.deg, radius=sdo_dist*u.m, obstime=obstime)
                observer_i = frames.HeliographicStonyhurst(0.*u.deg, sdo_lat*u.deg, radius=sdo_dist*u.m, obstime=obstime)
                
                c1 = SkyCoord(grid[:, 0]*u.arcsec, grid[:, 1]*u.arcsec, frame=frames.Helioprojective, observer=observer_1)#observer="earth", obstime=f"{day} 00:00:00")
                c2 = c1.transform_to(frames.HeliographicCarrington)
                c3 = SkyCoord(c2.lon, c2.lat, frame=frames.HeliographicCarrington, observer=observer_i, obstime=obstime)#, observer="earth")
                c4 = c3.transform_to(frames.Helioprojective)
                
                x_pix, y_pix = image_to_pix(c4.Tx.value, c4.Ty.value)
                x_pix = np.round(x_pix)
                y_pix = np.round(y_pix)

                #######################
                # No tracking
                c3 = SkyCoord(c2.lon, c2.lat, frame=frames.HeliographicCarrington, observer=observer_1, obstime=obstime)#, observer="earth")
                c4 = c3.transform_to(frames.Helioprojective)
                x_pix_nt, y_pix_nt = image_to_pix(c4.Tx.value, c4.Ty.value)
                x_pix_nt = np.round(x_pix_nt)
                y_pix_nt = np.round(y_pix_nt)

                #######################
                    
                self.data = np.empty((ny, nx), dtype=np.float32)
                data_nt = np.empty((ny, nx), dtype=np.float32)
                l = 0
                for j in np.arange(ny):
                    #print("--------")
                    for k in np.arange(nx):
                        #print("y, x", y_pix[l], x_pix[l])
                        if np.isnan(y_pix[l]) or np.isnan(x_pix[l]):
                            self.data[j, k] = np.nan
                        else:
                            self.data[j, k] = data[int(y_pix[l]), int(x_pix[l])]
                        if np.isnan(y_pix_nt[l]) or np.isnan(x_pix_nt[l]):
                            data_nt[j, k] = np.nan
                        else:
                            data2_nt[j, k] = data[int(y_pix_nt[l]), int(x_pix_nt[l])]
                        l += 1
                test_plot = plot.plot(nrows=1, ncols=1, size=plot.default_size(self.data.shape[1]//8, self.data.shape[0]//8))
                test_plot.colormap(self.data, cmap_name="bwr", show_colorbar=True)
                suffix = self.get_obs_time2()
                test_plot.save(f"frame_{suffix}.png")
                test_plot.close()
                test_plot = plot.plot(nrows=1, ncols=1, size=plot.default_size(data_nt.shape[1]//8, data_nt.shape[0]//8))
                test_plot.colormap(data_nt, cmap_name="bwr", show_colorbar=True)
                test_plot.save(f"frame_nt_{suffix}.png")
                test_plot.close()
                print(i)
                self.calc_stats()
                sys.stdout.flush()
                self.frame_index += 1
            hdul.close()
            if self.frame_index >= self.num_frames_per_day:
                self.frame_index = 0
                self.day_index += 1
        else:
            raise "No more files"
        
        
    def track(self):
        self.day_index = 0
        self.frame_index = 0
        while True:
            self.process_frames()
            if self.num_days > 0 and self.day_index >= self.num_days:
                break


if (__name__ == '__main__'):
    
    path = '.'
    start_date = '2013-02-14'
    num_days = -1 # How many days to track
    num_frames = 8*5
    step = 10 # Step in number of frames between tracked sequences of num_hrs length
    
    i = 1
    
    if len(sys.argv) > i:
        path = sys.argv[i]
    i += 1
    if len(sys.argv) > i:
        start_date = sys.argv[i]
    i += 1
    if len(sys.argv) > i:
        num_days = int(sys.argv[i])
    i += 1
    if len(sys.argv) > i:
        num_frames = int(sys.argv[i])
    i += 1
    if len(sys.argv) > i:
        step = int(sys.argv[i])
    
    tr = track(path=path, start_date=start_date, num_days=num_days, num_frames=num_frames, step=step)
    tr.track()

        
        
