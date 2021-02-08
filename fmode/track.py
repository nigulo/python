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
import plot

import astropy.units as u
from astropy.coordinates import SkyCoord
import sunpy.coordinates
from sunpy.coordinates import frames
from datetime import datetime, timedelta

A = 14.713
B = -2.396
C = -1.787

DEBUG = False

stats_file_mode = "daily"

radius_km = 695700

def diff_rot(lat):
    return (A + B*np.sin(lat)**2 + C*np.sin(lat)**4)*np.pi/180


class stats_header:
    
    def __init__(self):
        self.cards = list()
        self.data = None
     
    def add_card(self, card):
        self.cards.append(card)
                
    def get_cards(self):
        return self.cards
    

class stats:

    def __init__(self, patch_lons, patch_lats, patch_size):
        self.header = None
        self.num_frames = 0
        self.patch_lons = patch_lons
        self.patch_lats = patch_lats
        self.patch_lons0 = patch_lons[0]
        self.patch_lats0 = patch_lats[0]
        self.patch_size = patch_size
        self.patch_step = patch_lons[1] - patch_lons[0]
        self.num_patches = len(patch_lons)
        self.data = np.zeros((self.num_patches, self.num_patches, 3))

    def init(self, date):
        self.date = date
        self.storage = fits.open(f"{date}.fits", mode="append")
        self.tracked_times = set()
        for entry in self.storage:
            self.tracked_times.add(entry.header["TIME"])

    def get_date(self):
        return self.date
    
    def get_num_frames(self):
        return self.num_frames
    
    def set_header(self, header):
        assert(self.header is None)
        self.header = fits.Header(header.get_cards())
        
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
                self.data[i, j] += [np.nansum(abs_data), np.nansum(abs_data**2), np.product(abs_data.shape)]
                if DEBUG:
                    color = "rb"[(i % 2 + k % 2) % 2]
                    test_plot.plot(x_pix2, -y_pix2 + int(np.sqrt(len(lons))), params=f"{color}.")
        if DEBUG:
            test_plot.save(f"patches{obs_time}.png")
            test_plot.close()
        self.num_frames += 1

    def save(self):
        assert(self.header is not None)
        means = self.data[:, :, 0]/self.data[:, :, 2]
        stds = np.sqrt((self.data[:, :, 1] - self.data[:, :, 0]**2)/self.data[:, :, 2])
        hdu = fits.ImageHDU(data=np.array([means, stds]), header=self.header, name='Statistics')
        if hdu.header["TIME"] not in self.tracked_times:
            self.tracked_times.add(hdu.header["TIME"])
            self.storage.append(hdu)
            self.storage.flush()
        self.data = np.zeros((self.num_patches, self.num_patches, 3))
        self.header = None
        self.num_frames = 0
        
    def close(self):
        self.storage.close()


def parse_t_rec(t_rec):
    year = t_rec[:4] 
    month = t_rec[5:7]
    day = t_rec[8:10]
    
    hrs = t_rec[11:13]
    mins = t_rec[14:16]
    secs = t_rec[17:19]
    
    return year, month, day, hrs, mins, secs

def filter_files(files, time):
    i = 0
    for f in files:
        if f >= time:
            break
        i += 1
    return files[i:]


class state:
    
    def __init__(self, step, num_days, num_hrs, path, files):
        self.step = step
        self.num_hrs = num_hrs
        
        self.path = path
        self.files = files
        
        hdul = fits.open(self.path + "/" + self.files[0])
        #self.num_frames_per_day = len(hdul) - 1
        t_rec = hdul[1].header['T_REC']
        year, month, day, hrs, mins, secs = parse_t_rec(t_rec)
        self.start_time = datetime(int(year), int(month), int(day), int(hrs), int(mins), int(secs))
        self.end_time = self.start_time + timedelta(hours=self.num_hrs)
        self.file_time = self.start_time
        
        if num_days > 0:
            self.done_time = self.start_time + timedelta(days=num_days)
        else:
            self.done_time = None
        
        hdul.close()
        
        self.frame_index = 0
        self.observer = None
        self.file = None
        self.hdul = None
        
        
    def get_num_frames_per_day(self):
        return self.num_frames_per_day
    
    def get_frame_index(self):
        return self.frame_index

    def next(self):
        files = filter_files(self.files, str(self.file_time)[:10])
        file_date = files[0][:10]
        file = self.path + "/" + files[0]
        
        if self.file is None or file != self.file:
            if self.hdul is not None:
                self.hdul.close()
            self.file = file
            self.hdul = fits.open(self.file)
        
        self.num_frames_per_day = len(self.hdul) - 1
        
        self.metadata = self.hdul[self.frame_index + 1].header

        t_rec = self.metadata['T_REC']
        
        year, month, day, hrs, mins, secs = parse_t_rec(t_rec)
        date = year + "-" + month + "-" + day
        assert(file_date == date)
        
        self.hrs = hrs
        self.mins = mins
        self.secs = secs

        self.obs_time = datetime(int(year), int(month), int(day), int(hrs), int(mins), int(secs))
        
        self.obs_time_str = f"{date} {self.hrs}:{self.mins}:{self.secs}"
        self.obs_time_str2 = f"{date}_{self.hrs}:{self.mins}:{self.secs}"
        
        self.sdo_lon = self.metadata['CRLN_OBS']
        self.sdo_lat = self.metadata['CRLT_OBS']
        self.sdo_dist = self.metadata['DSUN_OBS']

        if self.obs_time >= self.end_time:
            self.stats.save()
            self.end_tracking()

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

    def get_obs_time(self):
        return self.obs_time

    def get_obs_time_str(self):
        return self.obs_time_str

    def get_obs_time_str2(self):
        return self.obs_time_str2

    def get_end_time(self):
        return self.end_time

    def get_end_time_str(self):
        return str(self.end_time)[:19]
    
    def get_observer(self):
        return self.observer
    
    def get_sdo_lon(self):
        return self.sdo_lon

    def get_sdo_lat(self):
        return self.sdo_lat

    def get_sdo_dist(self):
        return self.sdo_dist
    
    def is_tracking(self):
        return self.observer is not None
    
    def start_tracking(self):
        assert(self.observer is None)
        self.observer = frames.HeliographicStonyhurst(0.*u.deg, self.sdo_lat*u.deg, radius=self.sdo_dist*u.m, obstime=self.get_obs_time_str())

        header = stats_header()
        header.add_card(fits.Card(keyword="START_TIME", value=self.get_obs_time_str(), comment="Tracking start time"))
        header.add_card(fits.Card(keyword="END_TIME", value=self.get_end_time_str(), comment="Tracking end time"))
        header.add_card(fits.Card(keyword="CLON", value=self.get_sdo_lon(), comment="Carrington longitude of start frame"))
        self.stats.set_header(header)

    def end_tracking(self):
        self.observer = None
        self.start_time = self.start_time + timedelta(hours=self.step)
        self.end_time = self.start_time + timedelta(hours=self.num_hrs)
        self.file_time = self.start_time
        self.files = filter_files(self.files, str(self.file_time)[:10])
        
    def frame_processed(self):
        self.frame_index += 1
        if self.frame_index >= self.num_frames_per_day:
            self.frame_index = 0
            self.file_time = self.file_time + timedelta(days=1)
            
    def is_done(self):
        return len(self.files) == 0 or (self.done_time is not None and self.file_time >= self.done_time)
    
    def close():
        self.hdul.close()
        self.stats.close()


def pix_to_image(xs, ys, dx, dy, xc, yc, cos_a, sin_a, arcsecs_per_pix_x, arcsecs_per_pix_y):
    xs = xs - xc
    ys = ys - yc
    
    xs_arcsec = dx + arcsecs_per_pix_x*cos_a*xs - arcsecs_per_pix_y*sin_a*ys
    ys_arcsec = dy + arcsecs_per_pix_x*sin_a*xs + arcsecs_per_pix_y*cos_a*ys
                        
    return xs_arcsec, ys_arcsec

def image_to_pix(xs_arcsec, ys_arcsec, dx, dy, xc, yc, cos_a, sin_a, coef_x, coef_y):
    xs_arcsec = xs_arcsec - dx
    ys_arcsec = ys_arcsec - dy
    
    xs = xc + coef_x*cos_a*(xs_arcsec) + coef_y*sin_a*(ys_arcsec) - 1
    ys = yc - coef_x*sin_a*(xs_arcsec) + coef_y*cos_a*(ys_arcsec) - 1
                        
    return xs, ys


class track:

    def __init__(self, path, files, num_days=-1, num_hrs=8, step=1, num_patches=100, patch_size=15):
        self.num_patches = num_patches
        self.patch_size = patch_size
        
        # Here we already assume that tracked sequences are 8hrs
        assert(self.patch_size < 160)
        self.patch_lons = np.linspace(-80, 80 - patch_size, num_patches)
        self.patch_lats = self.patch_lons
        
        print(path)
        
        self.state = state(step, num_days, num_hrs, path, files)
        sts = stats(self.patch_lons, self.patch_lats, self.patch_size)
        self.state.set_stats(sts)
        self.state.next()
        sts.init(self.state.get_obs_time_str2())

        metadata = self.state.get_metadata()

        self.nx = metadata['NAXIS1']
        self.ny = metadata['NAXIS2']
        
        self.xs = (np.arange(1, self.nx + 1)).astype(float)
        self.ys = (np.arange(1, self.ny + 1)).astype(float)


    def transform(self):

        print(f"time 1: {time.perf_counter()}")
        metadata = self.state.get_metadata()
        
        obs_time = self.state.get_obs_time_str()
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
                
        sin_a = np.sin(a)
        cos_a = np.cos(a)
        
        xs_arcsec, ys_arcsec = pix_to_image(self.xs, self.ys, dx, dy, xc, yc, cos_a, sin_a, arcsecs_per_pix_x, arcsecs_per_pix_y)
        grid = np.transpose([np.tile(xs_arcsec, ny), np.repeat(ys_arcsec, nx)])
        
        print(f"time 2: {time.perf_counter()}")        

        observer = self.state.get_observer()
        observer_i = frames.HeliographicStonyhurst(0.*u.deg, self.state.get_sdo_lat()*u.deg, radius=self.state.get_sdo_dist()*u.m, obstime=obs_time)
        
        c1 = SkyCoord(grid[:, 0]*u.arcsec, grid[:, 1]*u.arcsec, frame=frames.Helioprojective, observer=observer)
        c2 = c1.transform_to(frames.HeliographicCarrington)
        lons = c2.lon.value - self.state.get_sdo_lon()
        lats = c2.lat.value
        c3 = SkyCoord(c2.lon, c2.lat, frame=frames.HeliographicCarrington, observer=observer_i, obstime=obs_time)
        c4 = c3.transform_to(frames.Helioprojective)

        print(f"time 3: {time.perf_counter()}")
        
        x_pix, y_pix = image_to_pix(c4.Tx.value, c4.Ty.value, dx, dy, xc, yc, cos_a, sin_a, coef_x, coef_y)
        x_pix = np.round(x_pix)
        y_pix = np.round(y_pix)
        
        #######################
        # No tracking
        if DEBUG:
            c3 = SkyCoord(c2.lon, c2.lat, frame=frames.HeliographicCarrington, observer=observer, obstime=obs_time)#, observer="earth")
            c4 = c3.transform_to(frames.Helioprojective)
            x_pix_nt, y_pix_nt = image_to_pix(c4.Tx.value, c4.Ty.value, dx, dy, xc, yc, cos_a, sin_a, coef_x, coef_y)
            x_pix_nt = np.round(x_pix_nt)
            y_pix_nt = np.round(y_pix_nt)

        #######################
        if DEBUG:
            data_for_plot = np.empty((ny, nx), dtype=np.float32)
            data_nt = np.empty((ny, nx), dtype=np.float32)
        
        print(f"time 4: {time.perf_counter()}")

        l = 0
        for j in np.arange(ny):
            for k in np.arange(nx):
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
            suffix = self.state.get_obs_time_str2()
            test_plot.save(f"frame_{suffix}.png")
            test_plot.close()
            test_plot = plot.plot(nrows=1, ncols=1, size=plot.default_size(data_nt.shape[1]//8, data_nt.shape[0]//8))
            test_plot.colormap(data_nt, cmap_name="bwr", show_colorbar=True)
            test_plot.save(f"frame_nt_{suffix}.png")
            test_plot.close()
        sys.stdout.flush()
        
        return lons, lats, x_pix, y_pix, data

    def process_frame(self):

        lons, lats, x_pix, y_pix, data = self.transform()

        self.state.get_stats().process_frame(lons, lats, x_pix, y_pix, data, obs_time=self.state.get_obs_time_str2())
        
        self.state.frame_processed()

        print(f"time 6: {time.perf_counter()}")
        print(f"time 7: {time.perf_counter()}")        
        sys.stdout.flush()
        
        
    def track(self):
        while not self.state.is_done():
            self.state.next()
            self.process_frame()
            if not self.state.is_tracking():
                create_new_stats = False
                date = self.state.get_obs_time_str()[:10]
                stats_date = self.state.get_stats().get_date()
                if stats_file_mode == "daily":
                    if date > stats_date:
                        create_new_stats = True
                if stats_file_mode == "monthly":
                    if date[:7] > stats_date[:7]:
                        create_new_stats = True
                if stats_file_mode == "yearly":
                    if date[:4] > stats_date[:4]:
                        create_new_stats = True
                if create_new_stats:
                    self.state.get_stats().close()
                    sts = stats(self.patch_lons, self.patch_lats, self.patch_size)
                    self.state.set_stats(sts)
                    sts.init(self.state.get_obs_time_str2())
        self.state.close()


if (__name__ == '__main__'):
    
    path = '.'
    start_date = '2013-02-14'
    num_days = -1 # For how many days to run the script
    num_hrs = 8 # Duration of tracking
    step = 1 # Step in hours between tracked sequences of num_hrs length
    num_patches = 100
    patch_size = 15
    
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
        num_hrs = float(sys.argv[i])
    i += 1
    if len(sys.argv) > i:
        step = float(sys.argv[i])
    i += 1
    if len(sys.argv) > i:
        num_patches = int(sys.argv[i])
    i += 1
    if len(sys.argv) > i:
        patch_size = float(sys.argv[i])
    
    all_files = list()
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if file >= start_date:
                all_files.append(file)
    all_files.sort()    
    
    tr = track(path=path, files=all_files, num_days=num_days, num_hrs=num_hrs, step=step, 
               num_patches=num_patches, patch_size=patch_size)
    tr.track()

        
        
