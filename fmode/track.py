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

DEBUG = True

stats_file_mode = "daily"

radius_km = 695700

def diff_rot(lat):
    #lat *= np.pi / 180
    return (A + B*np.sin(lat)**2 + C*np.sin(lat)**4)*np.pi/180


class stats_header:
    
    def __init__(self):
        self.cards = list()
        self.data = None
     
    def add_card(self, card):
        self.cards.append(card)
                
    def get_cards(self):
        return self.cards
    
#------------------------------------------------------------------------------

class stats:

    def __init__(self, date, patch_lons, patch_lats, patch_size):
        self.storage = fits.open(f"{date}.fits", mode="append")
        self.header = None
        self.date = date
        self.num_frames = 0
        self.patch_lons = patch_lons
        self.patch_lats = patch_lats
        self.patch_lons0 = patch_lons[0]
        self.patch_lats0 = patch_lats[0]
        self.patch_size = patch_size
        self.patch_step = patch_lons[1] - patch_lons[0]
        self.num_patches = len(patch_lons)
        self.data = np.zeros((self.num_patches, self.num_patches, 3))

    def get_date(self):
        return self.date
    
    def get_num_frames(self):
        return self.num_frames
    
    def set_header(self, header):
        assert(self.header is None)
        self.header = fits.Header(header.get_cards())
    
    '''
    def get_indices(self, lon, lat):
        lon_end1 = (lon - self.patch_lons0)/self.patch_step
        lon_end = int(lon_end1)
        lon_delta = (lon_end1 - lon_end)*self.patch_step

        lon_start = max(0, lon_end - int((self.patch_size - lon_delta)/self.patch_step))
        
        lat_end1 = (lat - self.patch_lats0)/self.patch_step
        lat_end = int(lat_end1)
        lat_delta = (lat_end1 - lat_end)*self.patch_step

        lat_start = max(0, lat_end - int((self.patch_size - lat_delta)/self.patch_step))
        
        return lon_start, lon_end, lat_start, lat_end

    def process_pixel(self, lon, lat, value):
        
        lon_start, lon_end, lat_start, lat_end = self.get_indices(lon, lat)
        lon_end += 1
        lat_end += 1
        
        abs_value = np.abs(value)
        self.data[lon_start:lon_end, lat_start:lat_end] += [abs_value, abs_value**2, 1]
    '''
    
    def process_frame(self, lons, lats, x_pix, y_pix, data, obs_time=None):
        if DEBUG:
            test_plot = plot.plot(nrows=1, ncols=1, size=plot.default_size(1000, 1000))
        for i in range(len(self.patch_lons)):
            patch_lon = self.patch_lons[i]
            lon_filter = (lons >= patch_lon) * (lons < patch_lon + self.patch_size)
            lons1 = lons[lon_filter]
            lats1 = lats[lon_filter]
            x_pix1 = x_pix[lon_filter]
            y_pix1 = y_pix[lon_filter]
            for j in range(len(self.patch_lats)):
                patch_lat = self.patch_lats[j]
                lat_filter = (lats1 >= patch_lat) * (lats1 < patch_lat + self.patch_size)
                lons2 = lons1[lat_filter]
                lats2 = lats1[lat_filter]
                x_pix2 = x_pix1[lat_filter]
                y_pix2 = y_pix1[lat_filter]
                abs_data = np.abs(data[y_pix2.astype(int), x_pix2.astype(int)])
                self.data[i, j] += [np.sum(abs_data), np.sum(abs_data**2), np.product(abs_data.shape)]
                if DEBUG:
                    k = i*len(self.patch_lats)+j
                    color = "rb"[((k // self.num_patches) % 2 + k % 2) % 2]
                    test_plot.plot(x_pix2, -y_pix2 + int(np.sqrt(len(lons))), params=f"{color}.")
        if DEBUG:
            test_plot.save(f"patches{obs_time}.png")
            test_plot.close()
        self.num_frames += 1
    
        
    #def frame_processed(self):
    #    self.num_frames += 1

    def save(self):
        means = self.data[:, :, 0]/self.data[:, :, 2]
        stds = np.sqrt((self.data[:, :, 1] - self.data[:, :, 0]**2)/self.data[:, :, 2])
        hdu = fits.ImageHDU(data=np.array([means, stds]), header=self.header, name='Statistics')
        self.storage.append(hdu)
        self.storage.flush()
        self.data = np.zeros((self.num_patches, self.num_patches, 3))
        self.header = None
        self.num_frames = 0
        
    def close(self):
        self.storage.close()

#------------------------------------------------------------------------------

class state:
    
    def __init__(self, step, num_days, num_frames, path, files):
        self.step = step
        self.num_days = num_days
        self.num_frames = num_frames
        
        self.path = path
        self.files = files
        
        self.hdul = fits.open(self.path + "/" + self.files[0])
        self.num_frames_per_day = len(self.hdul) - 1
        
        self.start_day_index = 0
        self.start_frame_index = 0
        self.day_index = 0
        self.frame_index = 0
        self.observer = None
        self.metadata = self.hdul[1].header
        self.file = None
        
    def get_num_frames_per_day(self):
        return self.num_frames_per_day
    
    def get_frame_index(self):
        return self.frame_index

    def next(self):    
        file_date = self.files[self.day_index][:10]
        file = self.path + "/" + self.files[self.day_index]
        
        if self.file is None or file != self.file:
            if self.hdul is not None:
                self.hdul.close()
                self.file = file
                self.hdul = fits.open(self.file)
        
        assert(self.num_frames_per_day == len(hdul) - 1)
        
        self.metadata = self.hdul[self.frame_index + 1].header

        t_rec = self.metadata['T_REC']
        
        self.date = t_rec[:10]
        assert(file_date == self.date)
        
        self.hrs = t_rec[11:13]
        self.mins = t_rec[14:16]
        self.secs = t_rec[17:19]
        self.obstime = t_rec[:19]
        
        self.obs_time = f"{self.date} {self.hrs}:{self.mins}:{self.secs}"
        self.obs_time2 = f"{self.date}_{self.hrs}:{self.mins}:{self.secs}"
        
        if DEBUG:
            hrs = self.frame_index*24/self.num_frames_per_day
            mins = (hrs - int(hrs))*60
            hrs = int(hrs)
            secs = int(round((mins - int(mins))*60))
            mins = int(mins)
            if secs == 60:
                secs = 0
                mins += 1
                if mins == 60:
                    mins = 0
                    hrs += 1
            hrs = format(hrs, "02")
            mins = format(mins, "02")
            secs = format(secs, "02")
            
            assert(self.hrs == hrs)
            assert(self.mins == mins)
            assert(self.secs == secs)

        self.sdo_lon = header['CRLN_OBS']
        self.sdo_lat = header['CRLT_OBS']
        self.sdo_dist = header['DSUN_OBS']

        if not self.is_tracking():
            self.start_tracking()
    
    def set_stats(self, stats):
        self.stats = stats
        
    def get_stats(self):
        return self.stats
        
    def get_metadata(self):
        return self.metadata
    
    def get_data(self):
        return fits.getdata(self.file, self.frame_index + 1)

    def get_date(self):
        return self.date

    def get_obs_time(self):
        return self.obs_time

    def get_obs_time2(self):
        return self.obs_time2
    
    def get_observer(self):
        return self.observer
    
    def get_sdo_lon(self):
        return self.sdo_lon
    
    def is_tracking():
        return self.observer is not None
    
    def start_tracking(self):
        assert(self.observer is None)
        self.observer = frames.HeliographicStonyhurst(0.*u.deg, self.sdo_lat*u.deg, radius=self.sdo_dist*u.m, obstime=self.obs_time)

        header = stats_header()
        header.add_card(fits.Card(keyword="TIME", value=self.get_obs_time(), comment="Observation time"))
        header.add_card(fits.Card(keyword="CLON", value=self.get_sdo_lon(), comment="Carrington longitude"))
        self.stats.set_header(header)

    def end_tracking(self):
        self.observer = None
        self.start_frame_index += self.step
        if self.start_frame_index >= self.num_frames_per_day:
            self.start_day_index += self.start_frame_index//self.num_frames_per_day
            self.start_frame_index = self.start_frame_index % self.num_frames_per_day
        self.frame_index = self.start_frame_index
        self.day_index = self.start_day_index
        
    def frame_processed(self):
        self.frame_index += 1
        if self.frame_index >= self.num_frames_per_day:
            self.frame_index = 0
            self.day_index += 1

        if self.stats.get_num_frames() >= self.num_frames:
            self.stats.save()
            self.state.end_tracking()
            
    def is_done(self):
        return day_index >= len(self.files) or (self.num_days > 0 and self.day_index >= self.num_days)
    
    def close():
        self.hdul.close()
        self.stats.close()

#------------------------------------------------------------------------------
def pix_to_image(xs, ys):
    xs = xs - xc
    ys = ys - yc
    
    xs_arcsec = dx + arcsecs_per_pix_x*cos_a*xs - arcsecs_per_pix_y*sin_a*ys
    ys_arcsec = dy + arcsecs_per_pix_x*sin_a*xs + arcsecs_per_pix_y*cos_a*ys
                        
    return xs_arcsec, ys_arcsec

def image_to_pix(xs_arcsec, ys_arcsec):
    xs_arcsec = xs_arcsec - dx
    ys_arcsec = ys_arcsec - dy
    
    xs = xc + coef_x*cos_a*(xs_arcsec) + coef_y*sin_a*(ys_arcsec)
    ys = yc - coef_x*sin_a*(xs_arcsec) + coef_y*cos_a*(ys_arcsec)
                        
    return xs, ys
#------------------------------------------------------------------------------

class track:
    
    def __init__(self, path='.', start_date='2013-02-14', num_days=-1, num_frames=8*5, step=10, num_patches=100, patch_size=15):
        self.start_date = start_date
        self.num_patches = num_patches
        self.patch_size = patch_size
        
        # Here we already assume that tracked sequences are 8hrs
        assert(self.patch_size < 160)
        assert(self.patch_size >= 160/num_patches)
        self.patch_lons = np.linspace(-80, 80 - patch_size, num_patches)
        self.patch_lats = self.patch_lons
        
        print(path)
        
        all_files = list()
        
        for root, dirs, files in os.walk(path):
            for file in files:
                if file >= start_date:
                    all_files.append(file)
        all_files.sort()

        self.state = state(step, num_days, num_frames, path, all_files)

        metadata = self.state.get_metadata()
        #print(hdul[1].header)
        self.nx = metadata['NAXIS1']
        self.ny = metadata['NAXIS2']
        
        #print(coef_x, coef_y, xc, yc)
        
        xs = (np.arange(1, self.nx + 1)).astype(float)
        ys = (np.arange(1, self.ny + 1)).astype(float)
        xs_arcsec, ys_arcsec = pix_to_image(xs, ys)

        self.grid = np.transpose([np.tile(xs_arcsec, ny), np.repeat(ys_arcsec, nx)])
        self.state.set_stats(stats(self.state.get_obs_time2(), self.patch_lons, self.patch_lats, self.patch_size))


    '''
    def is_tracking():
        return self.observer is not None
    
    def start_tracking(self, sdo_lat, sdo_dist, obs_time):
        self.observer = frames.HeliographicStonyhurst(0.*u.deg, sdo_lat*u.deg, radius=sdo_dist*u.m, obstime=obs_time)

    def end_tracking(self):
        assert(self.stats is not None)
        self.stats.save()
        self.observer = None
        self.start_frame_index += self.step
        if self.start_frame_index >= self.num_frames_per_day:
            self.start_day_index += self.start_frame_index//self.num_frames_per_day
            self.start_frame_index = self.start_frame_index % self.num_frames_per_day
        self.frame_index = self.start_frame_index
        self.day_index = self.start_day_index
    '''    
        

    '''
    def calc_stats_patch(self, lon, lat, plt = None, color = None):
        #print("calc_stats_patch", lon, lat, np.nanmin(self.lons), np.nanmax(self.lons), np.nanmin(self.lats), np.nanmax(self.lats))
        lon_filter = (self.lons >= lon) * (self.lons < lon + self.patch_size)
        lat_filter = (self.lats >= lat) * (self.lats < lat + self.patch_size)
        fltr = lon_filter * lat_filter
        if DEBUG:
            print(self.x_pix[fltr], self.y_pix[fltr])
            plt.plot(self.x_pix[fltr], -self.y_pix[fltr] + self.ny, params=f"{color}.")
        patch = self.data[self.y_pix[fltr].astype(int), self.x_pix[fltr].astype(int)]
        abs_patch = np.abs(patch)
        return np.array([np.sum(abs_patch), np.sum(abs_patch**2), len(abs_patch)])
    '''

    '''
    def calc_stats(self):
        if self.stats.is_new():
            header = stats_header()
            header.add_card(fits.Card(keyword="TIME", value=self.get_obs_time(), comment="Observation time"))
            header.add_card(fits.Card(keyword="CLON", value=self.sdo_lon, comment="Carrington longitude"))
            self.stats.set_header(header)
        #######################################################################
        #for i in range(len(self.lons)):
        #    if (not np.isnan(self.lons[i])) and (not np.isnan(self.lats[i])):
        #        if np.isnan(self.x_pix[i]) or np.isnan(self.y_pix[i]):
        #            value = np.nan
        #        else:
        #            value = self.data[int(self.y_pix[i]), int(self.x_pix[i])]
        #        self.stats.process_pixel(self.lons[i], self.lats[i], value)
        #self.stats.frame_processed()
        #######################################################################
        #sums_counts = np.empty((self.num_patches**2, 3))
        #if DEBUG:
        #    test_plot = plot.plot(nrows=1, ncols=1, size=plot.default_size(1000, 1000))
        #    colors = "rb"
        #i = 0
        #for lon in self.patch_lons:
        #    for lat in self.patch_lats:
        #        if DEBUG:
        #            color = colors[((i // self.num_patches) % 2 + i % 2) % 2]
        #            sums_counts[i] = self.calc_stats_patch(lon, lat, test_plot, color)
        #        else:
        #            sums_counts[i] = self.calc_stats_patch(lon, lat)
        #        i += 1
        #if DEBUG:
        #    suffix = self.get_obs_time2()
        #    test_plot.save(f"patches{suffix}.png")
        #    test_plot.close()
        #######################################################################
        self.stats.process_frame(self.lons, self.lats, self.x_pix, self.y_pix, self.data, obs_time=self.get_obs_time2())
    '''        

    '''
    def process_frame(self):
        #self.frame_index += 1
        #if self.frame_index >= self.num_frames_per_day:
        #    self.frame_index = 0
        #    self.day_index += 1
        
        if self.stats.is_new():
            header = stats_header()
            header.add_card(fits.Card(keyword="TIME", value=self.state.get_obs_time(), comment="Observation time"))
            header.add_card(fits.Card(keyword="CLON", value=self.state.get_sdo_lon(), comment="Carrington longitude"))
            self.stats.set_header(header)
        self.stats.process_frame(self.lons, self.lats, self.x_pix, self.y_pix, self.data, obs_time=self.state.get_obs_time2())
        
        self.state.frame_processed()

        print(f"time 6: {time.perf_counter()}")
        if self.stats.get_num_frames() >= self.num_frames:
            assert(self.stats is not None)
            self.stats.save()
            self.state.end_tracking()
            print(f"time 7: {time.perf_counter()}")
            return True
        print(f"time 7: {time.perf_counter()}")
        return False
    '''

    '''
    def set_time(self, header):
        t_rec = header['T_REC']
        
        self.date = t_rec[:10]
        self.hrs = t_rec[11:13]
        self.mins = t_rec[14:16]
        self.secs = t_rec[17:19]
        self.obstime = t_rec[:19]
        
        self.obs_time = f"{self.date} {self.hrs}:{self.mins}:{self.secs}"
        self.obs_time2 = f"{self.date}_{self.hrs}:{self.mins}:{self.secs}"
        
        if DEBUG:
            hrs = self.frame_index*24/self.num_frames_per_day
            mins = (hrs - int(hrs))*60
            hrs = int(hrs)
            secs = int(round((mins - int(mins))*60))
            mins = int(mins)
            if secs == 60:
                secs = 0
                mins += 1
                if mins == 60:
                    mins = 0
                    hrs += 1
            hrs = format(hrs, "02")
            mins = format(mins, "02")
            secs = format(secs, "02")
            
            assert(self.hrs = hrs)
            assert(self.mins = mins)
            assert(self.secs = secs)
    '''
        
    def process_frame(self):
        print(self.state.get_date())
        self.create_stats()

        print(f"time 1: {time.perf_counter()}")
        metadata = self.state.get_metadata()#hdul[i].metadata
        
        obs_time = self.state.get_obs_time()
        print(obs_time)

        data = self.state.get_data()
        
        ctype1 = metadata['CTYPE1']
        ctype2 = metadata['CTYPE2']
        assert(ctype1 == "HPLN-TAN" and ctype2 == "HPLT-TAN")

        cunit1 = metadata['CUNIT1']
        cunit2 = metadata['CUNIT2']
        assert(cunit1 == "arcsec" and cunit2 == "arcsec")

        a = metadata['CROTA2']*np.pi/180
        nx = metadata['NAXIS1']
        ny = metadata['NAXIS2']
        assert(nx == self.nx and ny == self.ny)
        
        dx = metadata['CRVAL1']
        dy = metadata['CRVAL2']
        arcsecs_per_pix_x = metadata['CDELT1']
        arcsecs_per_pix_y = metadata['CDELT2']
        coef_x = 1./arcsecs_per_pix_x
        coef_y = 1./arcsecs_per_pix_y
        xc = metadata['CRPIX1']
        yc = metadata['CRPIX2']
        
        #sdo_lon = metadata['CRLN_OBS']
        #sdo_lat = metadata['CRLT_OBS']
        #sdo_dist = metadata['DSUN_OBS']
        
        if DEBUG:
            r_sun = metadata['RSUN_REF']
            self.r_sun_pix = int(round(np.arctan(r_sun/sdo_dist)*180/np.pi*3600*coef_x))
            self.xc, self.yc = xc, yc        
        
        #if self.observer is None:
        #    self.observer = frames.HeliographicStonyhurst(0.*u.deg, sdo_lat*u.deg, radius=sdo_dist*u.m, obstime=obstime)
        #    #self.sdo_lon0 = self.sdo_lon
        
        sin_a = np.sin(a)
        cos_a = np.cos(a)
        
        
        print(f"time 2: {time.perf_counter()}")        

        observer = self.state.get_observer()
        #observer_i = frames.HeliographicStonyhurst((sdo_lon-sdo_lon1)*u.deg, sdo_lat*u.deg, radius=sdo_dist*u.m, obstime=obstime)
        observer_i = frames.HeliographicStonyhurst(0.*u.deg, sdo_lat*u.deg, radius=sdo_dist*u.m, obstime=obs_time)
        
        c1 = SkyCoord(self.grid[:, 0]*u.arcsec, self.grid[:, 1]*u.arcsec, frame=frames.Helioprojective, observer=observer)#observer="earth", obstime=f"{day} 00:00:00")
        c2 = c1.transform_to(frames.HeliographicCarrington)
        lons = c2.lon.value - self.state.get_sdo_lon()
        #np.savetxt("lons.csv", self.lons, delimiter=",")
        lats = c2.lat.value
        c3 = SkyCoord(c2.lon, c2.lat, frame=frames.HeliographicCarrington, observer=observer_i, obstime=obs_time)#, observer="earth")
        c4 = c3.transform_to(frames.Helioprojective)

        print(f"time 3: {time.perf_counter()}")
        
        x_pix, y_pix = image_to_pix(c4.Tx.value, c4.Ty.value)
        x_pix = np.round(x_pix)
        y_pix = np.round(y_pix)

        #######################
        # No tracking
        if DEBUG:
            c3 = SkyCoord(c2.lon, c2.lat, frame=frames.HeliographicCarrington, observer=observer, obstime=obs_time)#, observer="earth")
            c4 = c3.transform_to(frames.Helioprojective)
            x_pix_nt, y_pix_nt = image_to_pix(c4.Tx.value, c4.Ty.value)
            x_pix_nt = np.round(x_pix_nt)
            y_pix_nt = np.round(y_pix_nt)

        #######################
        if DEBUG:
            data_for_plot = np.empty((ny, nx), dtype=np.float32)
            data_nt = np.empty((ny, nx), dtype=np.float32)
        
        print(f"time 4: {time.perf_counter()}")

        l = 0
        for j in np.arange(ny):
            #print("--------")
            for k in np.arange(nx):
                #print("y, x", y_pix[l], x_pix[l])
                if np.isnan(y_pix[l]) or np.isnan(x_pix[l]):
                    if DEBUG:
                        data_for_plot[j, k] = np.nan
                else:
                    if DEBUG:
                        data_for_plot[j, k] = data[int(y_pix[l]), int(x_pix[l])]
                if DEBUG:
                    if np.isnan(y_pix_nt[l]) or np.isnan(x_pix_nt[l]):
                        data_nt[j, k] = np.nan
                    else:
                        data_nt[j, k] = data[int(y_pix_nt[l]), int(x_pix_nt[l])]
                l += 1

        print(f"time 5: {time.perf_counter()}")

        if DEBUG:
            test_plot = plot.plot(nrows=1, ncols=1, size=plot.default_size(data_for_plot.data.shape[1]//8, data_for_plot.data.shape[0]//8))
            test_plot.colormap(data_for_plot, cmap_name="bwr", show_colorbar=True)
            suffix = self.state.get_obs_time2()
            test_plot.save(f"frame_{suffix}.png")
            test_plot.close()
            test_plot = plot.plot(nrows=1, ncols=1, size=plot.default_size(data_nt.shape[1]//8, data_nt.shape[0]//8))
            test_plot.colormap(data_nt, cmap_name="bwr", show_colorbar=True)
            test_plot.save(f"frame_nt_{suffix}.png")
            test_plot.close()
        sys.stdout.flush()

        self.state.get_stats().process_frame(lons, lats, x_pix, y_pix, data, obs_time=self.state.get_obs_time2())
        
        self.state.frame_processed()

        print(f"time 6: {time.perf_counter()}")
        print(f"time 7: {time.perf_counter()}")        
        sys.stdout.flush()
        
        
    def track(self):
        while not self.state.is_done():
            self.state.next()
            create_new_stats = False
            date = self.state.get_date()
            stats_date = self.state.set_stats().get_date()
            if stats_file_mode == "daily":
                if date > stats_date:
                    create_new_stats = True
            if stats_file_mode == "monthly":
                if date[:7] > stats_date[:7]:
                    create_new_stats = True
            if stats_file_mode == "yearly":
                if date[:4] > stats_date[:4]:
                    create_new_stats = True
            if create_new:
                self.state.get_stats().close()
                self.state.set_stats(stats(self.state.get_obs_time2(), self.patch_lons, self.patch_lats, self.patch_size))
            self.process_frame()
        self.state.close()


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
    i += 1
    if len(sys.argv) > i:
        num_patches = int(sys.argv[i])
    i += 1
    if len(sys.argv) > i:
        patch_size = float(sys.argv[i])
    
    tr = track(path=path, start_date=start_date, num_days=num_days, num_frames=num_frames, step=step, 
               num_patches=num_patches, patch_size=patch_size)
    tr.track()

        
        
