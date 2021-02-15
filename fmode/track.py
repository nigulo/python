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

from filelock import FileLock

A = 14.713
B = -2.396
C = -1.787

DEBUG = False

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

def collect_stats_1(data):
    data2 = data**2
    data3 = data2*data
    data4 = data2**2
    abs_data = np.abs(data)
    abs_data3 = np.abs(data3)
    return [np.sum(data), np.sum(data2), np.sum(data3), np.sum(data4),
             np.sum(abs_data), np.sum(abs_data3), np.product(data.shape)]

def collect_stats_2(data):
    n = data[:, :, 6]
    mean = data[:, :, 0]/n
    data2_mean = data[:, :, 1]/n
    data3_mean = data[:, :, 2]/n
    data4_mean = data[:, :, 3]/n
    abs_mean = data[:, :, 4]/n
    abs_data3_mean = data[:, :, 5]/n
    var = data2_mean - mean**2
    abs_var = data2_mean - abs_mean**2
    mean3 = mean**3
    abs_mean3 = abs_mean**3
    std = np.sqrt(var)
    std3 = var * std
    abs_std = np.sqrt(abs_var)
    abs_std3 = abs_var * abs_std
    skew = (data3_mean - 3*mean*var - mean3)/std3
    kurt = (data4_mean - mean*(2*data3_mean - mean3))/(var**2) - 2*mean*skew - 3
    abs_skew = (abs_data3_mean - 3*abs_mean*abs_var - abs_mean3)/abs_std3
    abs_kurt = (data4_mean - abs_mean*(2*abs_data3_mean - abs_mean3))/(abs_var**2) - 2*abs_mean*abs_skew - 3
    
    return [mean, std, skew, kurt, abs_mean, abs_std, abs_skew, abs_kurt]


class stats:

    def __init__(self, patch_lons, patch_lats, patch_size, output_path="."):
        self.header = None
        self.num_frames = 0
        self.patch_lons = patch_lons
        self.patch_lats = patch_lats
        self.patch_lons0 = patch_lons[0]
        self.patch_lats0 = patch_lats[0]
        self.patch_size = patch_size
        self.patch_step = patch_lons[1] - patch_lons[0]
        self.num_patches = len(patch_lons)
        self.output_path = output_path
        self.storage = None

    def init(self, date):
        self.date = date
        with FileLock("track.lock"):
            f = f"{self.output_path}/{date}.fits"
            if os.path.isfile(f):      
                print("File exists", f)
                return
            self.storage = fits.open(f, mode="append")
        # Just to ensure that the tracking is done only once.
        # This is not needed anymore, as we use file lock and pass tracking
        # entirely
        self.tracked_times = set()
        for entry in self.storage:
            self.tracked_times.add(entry.header["START_TIME"])
        self.data = np.zeros((self.num_patches, self.num_patches, 7))

    def get_date(self):
        return self.date
    
    def get_num_frames(self):
        return self.num_frames
    
    def set_header(self, header):
        assert(self.header is None)
        self.header = fits.Header(header.get_cards())
        
    def process_frame(self, lons, lats, x_pix, y_pix, data, obs_time, plot_file=None):
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
                #assert(len(x_pix2) == len(y_pix2))
                filtered_data = data[y_pix2.astype(int), x_pix2.astype(int)]
                filtered_data = filtered_data[np.logical_not(np.isnan(filtered_data))]
                
                self.data[i, j] += collect_stats_1(filtered_data)
                if DEBUG:
                    color = "rb"[(i % 2 + j % 2) % 2]
                    test_plot.plot(x_pix2, -y_pix2 + int(np.sqrt(len(lons))), params=f"{color}.")
        if DEBUG:
            test_plot.save(f"patches_{plot_file}.png")
            test_plot.close()
        self.num_frames += 1

    def save(self):
        assert(self.header is not None)
        hdu = fits.ImageHDU(data=np.array(collect_stats_2(self.data)), header=self.header, name='Statistics')
        if hdu.header["START_TIME"] not in self.tracked_times:
            self.tracked_times.add(hdu.header["START_TIME"])
            self.storage.append(hdu)
            self.storage.flush()
        self.data = np.zeros((self.num_patches, self.num_patches, 7))
        self.header = None
        self.num_frames = 0
        
    def close(self):
        if self.storage is not None:
            self.storage.close()
            self.storage = None

    def is_open(self):
        return self.storage is not None

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

class state:
    
    def __init__(self, num_hrs, step, num_bursts, path, files, start_time):
        self.step = step
        self.num_hrs = num_hrs
        self.num_bursts = num_bursts
        
        self.path = path
        self.files = files
        
        hdul = fits.open(self.path + "/" + self.files[0])
        if start_time is None:
            t_rec = hdul[1].header['T_REC']
            year, month, day, hrs, mins, secs = parse_t_rec(t_rec)
            self.start_time = datetime(int(year), int(month), int(day), int(hrs), int(mins), int(secs))
        else:
            self.start_time = start_time
            year, month, day, hrs, mins, secs = parse_t_rec(str(start_time))
        self.end_time = self.start_time + timedelta(hours=self.num_hrs)
        self.file_time = self.start_time
 
        self.obs_time = self.start_time
        
        self.obs_time_str = f"{year}-{month}-{day} {hrs}:{mins}:{secs}"
        self.obs_time_str2 = f"{year}-{month}-{day}_{hrs}:{mins}:{secs}"
       
        self.metadata = hdul[1].header
        
        hdul.close()
        
        self.frame_index = -1
        self.observer = None
        self.file = None
        self.hdul = None
        
        self.obs_time = None
        self.last_obs_time = None
        
        self.nx = self.metadata['NAXIS1']
        self.ny = self.metadata['NAXIS2']
        
        self.xs = np.arange(1, self.nx + 1)
        self.ys = np.arange(1, self.ny + 1)
        
   
    def get_num_frames_per_day(self):
        return self.num_frames_per_day
    
    def get_frame_index(self):
        return self.frame_index

    def next_frame(self):
        print("frame_index 1", self.frame_index)
        self.frame_index += 1
        if self.frame_index >= self.num_frames_per_day:
            self.frame_index = -1
            self.file_time = self.file_time + timedelta(days=1)
            #print("next-file")
            print("next_frame False")
            return False
        print("next_frame True")
        return True

    def next(self):
        files = filter_files(self.files, str(self.file_time)[:10])
        if len(files) == 0:
            self.stats.save()
            self.end_tracking()
            self.num_bursts -= 1
            return False
        if not self.stats.is_open():
            self.end_tracking()
            return False
        file_date = files[0][:10]
        file = self.path + "/" + files[0]
        
        if self.file is None or file != self.file:
            if self.hdul is not None:
                self.hdul.close()
            self.file = file
            self.hdul = fits.open(self.file)
        
        self.num_frames_per_day = len(self.hdul) - 1

        print("frame_index 2", self.frame_index)
        if not self.next_frame():
            return False
        print("frame_index 3", self.frame_index)
        self.metadata = self.hdul[self.frame_index + 1].header

        t_rec = self.metadata['T_REC']
        
        year, month, day, hrs, mins, secs = parse_t_rec(t_rec)
        date = year + "-" + month + "-" + day
        assert(file_date == date)
        
        self.hrs = hrs
        self.mins = mins
        self.secs = secs

        # This field is only for debug purposes
        self.last_obs_time = self.obs_time
        self.obs_time = datetime(int(year), int(month), int(day), int(hrs), int(mins), int(secs))
        
        if self.get_obs_time() < self.get_start_time():
            print("Return 1")
            return False
        
        self.obs_time_str = f"{date} {self.hrs}:{self.mins}:{self.secs}"
        self.obs_time_str2 = f"{date}_{self.hrs}:{self.mins}:{self.secs}"
        
        self.sdo_lon = self.metadata['CRLN_OBS']
        self.sdo_lat = self.metadata['CRLT_OBS']
        self.sdo_dist = self.metadata['DSUN_OBS']

        if self.is_tracking() and self.obs_time >= self.end_time:
            self.stats.save()
            self.end_tracking()
            self.num_bursts -= 1
            print("Return 2")
            return False

        if not self.is_tracking():
            self.start_tracking()
        
        print("Return 3")
        return True
    
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

    def get_last_obs_time(self):
        return self.last_obs_time

    def get_start_time(self):
        return self.start_time

    def get_start_time_str(self):
        return str(self.start_time)[:19]

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
        print("start_tracking", self.get_start_time(), self.get_obs_time())
        assert(self.get_start_time() <= self.get_obs_time() and (self.get_last_obs_time() is None or self.get_start_time() > self.get_last_obs_time()))
        self.observer = frames.HeliographicStonyhurst(0.*u.deg, self.sdo_lat*u.deg, radius=self.sdo_dist*u.m, obstime=self.get_obs_time_str())

        header = stats_header()
        header.add_card(fits.Card(keyword="START_TIME", value=self.get_start_time_str(), comment="Tracking start time"))
        header.add_card(fits.Card(keyword="END_TIME", value=self.get_end_time_str(), comment="Tracking end time"))
        header.add_card(fits.Card(keyword="CLON", value=self.get_sdo_lon(), comment="Carrington longitude of start frame"))
        self.stats.set_header(header)
        
        metadata = self.get_metadata()
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
        
        fltr = np.logical_not(np.isnan(self.get_data())).flatten()
        grid = grid[fltr]
        
        self.xs_arcsec = grid[:, 0]*u.arcsec
        self.ys_arcsec = grid[:, 1]*u.arcsec
        
        grid = np.transpose([np.tile(self.xs, ny), np.repeat(self.ys, nx)])
        grid = grid[fltr]
        self.xys = grid
        
        self.dbg_stack = []
        self.num_added = 0
        
        
    def get_xs_ys_arcsec(self):
        return self.xs_arcsec, self.ys_arcsec

    def get_xys(self):
        return self.xys

    def transform_index(self, i):
        for indices_deleted, related_indices in self.dbg_stack:
            for k in range(len(indices_deleted)):
                if indices_deleted[k] == i:
                    i = related_indices[k]
                    break
            num = 0
            for ind in indices_deleted:
                if ind < i:
                    num += 1
            i -= num
        return i
        
    def get_nx_ny(self):
        return self.nx, self.ny

    def frame_processed(self, xs_arcsec, ys_arcsec, observer, dbg_info):
        self.xs_arcsec = xs_arcsec
        self.ys_arcsec = ys_arcsec
        self.observer = observer
        self.dbg_stack.append(dbg_info)

    def end_tracking(self):
        self.observer = None
        self.start_time = self.start_time + timedelta(hours=self.step)
        self.end_time = self.start_time + timedelta(hours=self.num_hrs)
        self.file_time = self.start_time
        print("end_tracking", self.file_time)
        self.files = filter_files(self.files, str(self.file_time)[:10])
        self.frame_index = -1
                    
    def is_done(self):
        return len(self.files) == 0 or self.num_bursts == 0
    
    def close(self):
        print("Close")
        if self.is_tracking():
            self.stats.save()
            self.end_tracking()
        if self.hdul is not None:
            self.hdul.close()
        self.stats.close()

def fix_sampling(x_pix, y_pix, xs_arcsec, ys_arcsec, lons, lats, xys, sdo_lon, observer, image_params):
        pix_dict = {(x_pix[i], y_pix[i]) : i for i in range(len(x_pix))}
        indices_to_delete = []
        related_indices = []
        for i in range(len(x_pix)):
            if not np.isnan(x_pix[i]) and not np.isnan(x_pix[i]):
                ind = pix_dict[(x_pix[i], y_pix[i])]
                if ind != i:
                    indices_to_delete.append(i)
                    related_indices.append(ind)
        x_pix = np.delete(x_pix, indices_to_delete)
        y_pix = np.delete(y_pix, indices_to_delete)
        xs_arcsec = np.delete(xs_arcsec, indices_to_delete)
        ys_arcsec = np.delete(ys_arcsec, indices_to_delete)
        lons = np.delete(lons, indices_to_delete)
        lats = np.delete(lats, indices_to_delete)
        print("Number of pixels removed", len(indices_to_delete))
        
        added_x_pix = []
        added_y_pix = []

        for x, y in xys:
            if (x, y) not in pix_dict:
                added_x_pix.append(x)
                added_y_pix.append(y)
                    
        print("Number of pixels added", len(added_x_pix))
        x_pix = np.append(x_pix, added_x_pix)
        y_pix = np.append(y_pix, added_y_pix)
        dx, dy, xc, yc, cos_a, sin_a, arcsecs_per_pix_x, arcsecs_per_pix_y = image_params
        added_xs_arcsec, added_ys_arcsec = pix_to_image(np.asarray(added_x_pix), np.asarray(added_y_pix), dx, dy, xc, yc, cos_a, sin_a, arcsecs_per_pix_x, arcsecs_per_pix_y)
        added_xs_arcsec, added_ys_arcsec = added_xs_arcsec*u.arcsec, added_ys_arcsec*u.arcsec
        xs_arcsec = np.append(xs_arcsec, added_xs_arcsec)
        ys_arcsec = np.append(ys_arcsec, added_ys_arcsec)
        
        c1 = SkyCoord(added_xs_arcsec, added_ys_arcsec, frame=frames.Helioprojective, observer=observer)
        c2 = c1.transform_to(frames.HeliographicCarrington)
        added_lons = c2.lon.value - sdo_lon
        added_lats = c2.lat.value
        
        lons = np.append(lons, added_lons)
        lats = np.append(lats, added_lats)
                
        assert(len(lons) == len(lats))
        assert(len(lons) == len(x_pix))
        assert(len(x_pix) == len(y_pix))
        assert(len(x_pix) == len(xs_arcsec))
        assert(len(xs_arcsec) == len(ys_arcsec))

        return x_pix, y_pix, xs_arcsec, ys_arcsec, lons, lats, (indices_to_delete, related_indices)
    

class track:

    def __init__(self, input_path, output_path, files, num_hrs=8, step=1, num_bursts=-1, num_patches=100, patch_size=15, 
                 stats_dbg = None, stats_file_mode="burst", start_time=None):
        assert(stats_file_mode == "burst" or stats_file_mode == "day" or stats_file_mode == "month" or stats_file_mode == "year")
        self.num_patches = num_patches
        self.patch_size = patch_size
        
        # Here we already assume that tracked sequences are 8hrs
        assert(self.patch_size < 160)
        self.patch_lons = np.linspace(-80, 80 - patch_size, num_patches)
        self.patch_lats = self.patch_lons
        
        print(f"Input path: {input_path}")
        print(f"Output path: {output_path}")
        
        self.state = state(num_hrs, step, num_bursts, input_path, files, start_time)
        if stats_dbg is None:
            sts = stats(self.patch_lons, self.patch_lats, self.patch_size, output_path)
        else:
            sts = stats_dbg
        self.state.set_stats(sts)
        sts.init(self.state.get_start_time_str())

        #metadata = self.state.get_metadata()
        #self.nx = metadata['NAXIS1']
        #self.ny = metadata['NAXIS2']
        
        #self.xs = (np.arange(1, self.nx + 1)).astype(float)
        #self.ys = (np.arange(1, self.ny + 1)).astype(float)

        self.stats_file_mode = stats_file_mode

    def transform(self):

        print(f"time 1: {time.perf_counter()}")
        metadata = self.state.get_metadata()
        
        obs_time = self.state.get_obs_time_str()
        print(obs_time)

        data = self.state.get_data()
        
        if DEBUG:
            ctype1 = metadata['CTYPE1']
            ctype2 = metadata['CTYPE2']
            assert(ctype1 == "HPLN-TAN" and ctype2 == "HPLT-TAN")
    
            cunit1 = metadata['CUNIT1']
            cunit2 = metadata['CUNIT2']
            assert(cunit1 == "arcsec" and cunit2 == "arcsec")

        xs_arcsec, ys_arcsec = self.state.get_xs_ys_arcsec()

        a = metadata['CROTA2']*np.pi/180
        nx = metadata['NAXIS1']
        ny = metadata['NAXIS2']
        assert((nx, ny) == self.state.get_nx_ny())
        
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
        
        #xs_arcsec, ys_arcsec = pix_to_image(self.xs, self.ys, dx, dy, xc, yc, cos_a, sin_a, arcsecs_per_pix_x, arcsecs_per_pix_y)
        #grid = np.transpose([np.tile(xs_arcsec, ny), np.repeat(ys_arcsec, nx)])
        
        print(f"time 2: {time.perf_counter()}")        

        observer = self.state.get_observer()
        observer_i = frames.HeliographicStonyhurst(0.*u.deg, self.state.get_sdo_lat()*u.deg, radius=self.state.get_sdo_dist()*u.m, obstime=obs_time)
        
        c1 = SkyCoord(xs_arcsec, ys_arcsec, frame=frames.Helioprojective, observer=observer)
        c2 = c1.transform_to(frames.HeliographicCarrington)
        lons = c2.lon.value - self.state.get_sdo_lon()
        lats = c2.lat.value
        c3 = SkyCoord(c2.lon, c2.lat, frame=frames.HeliographicCarrington, observer=observer_i, obstime=obs_time)
        c4 = c3.transform_to(frames.Helioprojective)
        
        xs_arcsec = c4.Tx
        ys_arcsec = c4.Ty

        print(f"time 3: {time.perf_counter()}")
        
        x_pix, y_pix = image_to_pix(xs_arcsec.value, ys_arcsec.value, dx, dy, xc, yc, cos_a, sin_a, coef_x, coef_y)
        x_pix = np.round(x_pix)
        y_pix = np.round(y_pix)
        
        print(f"time 4: {time.perf_counter()}")
        
        x_pix, y_pix, xs_arcsec, ys_arcsec, lons, lats, dbg_info = fix_sampling(x_pix, y_pix, xs_arcsec, ys_arcsec, lons, lats, 
                                                                      self.state.get_xys(), self.state.get_sdo_lon(), observer_i,
                                                                      (dx, dy, xc, yc, cos_a, sin_a, arcsecs_per_pix_x, arcsecs_per_pix_y))

        print(f"time 5: {time.perf_counter()}")
        
        #######################
        # No tracking
        if DEBUG:
            c3 = SkyCoord(c2.lon, c2.lat, frame=frames.HeliographicCarrington, observer=observer, obstime=obs_time)#, observer="earth")
            c4 = c3.transform_to(frames.Helioprojective)
            x_pix_nt, y_pix_nt = image_to_pix(c4.Tx.value, c4.Ty.value, dx, dy, xc, yc, cos_a, sin_a, coef_x, coef_y)
            x_pix_nt = np.round(x_pix_nt)
            y_pix_nt = np.round(y_pix_nt)

            data_for_plot = np.empty((ny, nx), dtype=np.float32)
            data_nt = np.empty((ny, nx), dtype=np.float32)
            data_for_plot[:, :] = np.nan
            data_nt[:, :] = np.nan
        
            xys = self.state.get_xys()
            for l in range(len(x_pix)):
                j = xys[l, 1]
                k = xys[l, 0]
                if np.isnan(y_pix[l]) or np.isnan(x_pix[l]):
                    data_for_plot[j, k] = np.nan
                else:
                    data_for_plot[j, k] = data[int(y_pix[l]), int(x_pix[l])]
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
        
        self.state.frame_processed(xs_arcsec, ys_arcsec, observer_i, dbg_info)
        
        return lons, lats, x_pix, y_pix, data

    def process_frame(self):

        lons, lats, x_pix, y_pix, data = self.transform()

        self.state.get_stats().process_frame(lons, lats, x_pix, y_pix, data, obs_time=self.state.get_obs_time(), plot_file=self.state.get_obs_time_str2())
        
        print(f"time 6: {time.perf_counter()}")
        #self.state.frame_processed()

        #print(f"time 7: {time.perf_counter()}")        
        sys.stdout.flush()
        
        
    def track(self):
        while not self.state.is_done():
            if self.state.next():
                self.process_frame()
            if self.state.is_done():
                break
            if not self.state.is_tracking():
                create_new_stats = False
                date = self.state.get_start_time_str()
                stats_date = self.state.get_stats().get_date()
                print("Change file", date, stats_date)
                if self.stats_file_mode == "burst":
                    if date > stats_date:
                        create_new_stats = True
                elif self.stats_file_mode == "day":
                    if date[:10] > stats_date[:10]:
                        create_new_stats = True
                elif self.stats_file_mode == "month":
                    if date[:7] > stats_date[:7]:
                        create_new_stats = True
                elif self.stats_file_mode == "year":
                    if date[:4] > stats_date[:4]:
                        create_new_stats = True
                if create_new_stats:
                    self.state.get_stats().close()
                    #sts = stats(self.patch_lons, self.patch_lats, self.patch_size)
                    #self.state.set_stats(sts)
                    self.state.get_stats().init(self.state.get_start_time_str())
        self.state.close()


if (__name__ == '__main__'):
    
    input_path = '.'
    output_path = '.'
    start_date = '2013-02-14'
    num_hrs = 8 # Duration of tracking
    step = 1 # Step in hours between tracked sequences of num_hrs length
    num_bursts = -1 # For how many days to run the script
    num_patches = 100
    patch_size = 15
    
    i = 1
    
    if len(sys.argv) > i:
        input_path = sys.argv[i]
    i += 1
    if len(sys.argv) > i:
        output_path = sys.argv[i]
    i += 1
    if len(sys.argv) > i:
        start_date = sys.argv[i]
    i += 1
    if len(sys.argv) > i:
        num_hrs = float(sys.argv[i])
    i += 1
    if len(sys.argv) > i:
        step = float(sys.argv[i])
    i += 1
    if len(sys.argv) > i:
        num_bursts = int(sys.argv[i])
    i += 1
    if len(sys.argv) > i:
        num_patches = int(sys.argv[i])
    i += 1
    if len(sys.argv) > i:
        patch_size = float(sys.argv[i])
    
    if len(start_date) < 4:
        last_file = ""
        for root, dirs, files in os.walk(output_path):
            for file in files:
                if file[-5:] == ".fits":
                    if file > last_file:
                        try:
                            hdul = fits.open(output_path + "/" + last_file)
                            start_time = hdul[-1].header["START_TIME"]
                            hdul.close()
                            last_file = file
                        except:
                            # Corrupted fits file
                            pass    

    
    year, month, day, hrs, mins, secs = parse_t_rec(start_time)
    start_time = datetime(int(year), int(month), int(day), int(hrs), int(mins), int(secs))
    start_time = start_time + timedelta(hours=step)
    
    print("Start time", str(start_time))
        
    all_files = list()
    
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file >= start_date:
                all_files.append(file)
    all_files.sort()
    
    tr = track(input_path=input_path, output_path=output_path, files=all_files, num_bursts=num_bursts, num_hrs=num_hrs, step=step, 
               num_patches=num_patches, patch_size=patch_size, start_time=start_time)
    tr.track()

        
        
