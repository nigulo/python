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
from calendar import monthrange

PROFILE = False
num_chunks = 20
added_pix_list_size = 10000*num_chunks//20

inf = float("inf")

if PROFILE:
    import tracemalloc
    import gc

A = 14.713
B = -2.396
C = -1.787

DEBUG = True
DEBUG2 = False

radius_km = 695700

def take_snapshot(title="main"):
    if PROFILE:
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        t = time.perf_counter()
        total = 0
        with open(f"memory{pid}.prof", "a") as f:
            f.write("=======================================\n")
            i = 0
            for stat in top_stats:
                stat = str(stat)
                if i < 20:
                    f.write(f"{t} {title}: {stat}\n")
                    i += 1
                i = stat.find("size=")
                stat2 = stat[i+5:]
                size, unit = stat2.split(",")[0].split(" ")
                size = float(size)
                unit = unit.lower()
                if unit[0] == "g":
                    size *= 1024*1024*1024
                elif unit[0] == "m":
                    size *= 1024*1024
                elif unit[0] == "k":
                    size *= 1024
                total += size
            unit = "B"
            if total >= 1024:
                total /= 1024
                unit = "KiB"
            if total >= 1024:
                total /= 1024
                unit = "MiB"
            if total >= 1024:
                total /= 1024
                unit = "GiB"
            f.write(f"{t} {title}: Total: {total} {unit}\n")
            f.write(f"{t} {title}: GC counts: {gc.get_count()}\n")

def get_random_start_time(step):
    y = np.random.randint(2010, 2021)
    m = np.random.randint(1, 13)
    _, num_days = monthrange(y, m)
    d = np.random.randint(1, num_days+1)
    h = np.random.randint(0, 24//int(step))
    h *= int(step)
    return datetime(y, m, d, h, 0, 0)

    
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
    data6 = data3**2
    data8 = data4**2
    return [np.sum(data), np.sum(data2), np.sum(data3), np.sum(data4),
             np.sum(abs_data), np.sum(abs_data3), np.sum(data6), np.sum(data8), np.product(data.shape)]

def collect_stats_2(data):
    n = data[:, :, 8]
    mean = data[:, :, 0]/n
    data2_mean = data[:, :, 1]/n
    data3_mean = data[:, :, 2]/n
    data4_mean = data[:, :, 3]/n
    abs_mean = data[:, :, 4]/n
    abs_data3_mean = data[:, :, 5]/n
    data6_mean = data[:, :, 6]/n
    data8_mean = data[:, :, 7]/n
    mean2 = mean**2
    var = data2_mean - mean2
    abs_var = data2_mean - abs_mean**2
    mean3 = mean2*mean
    abs_mean3 = abs_mean**3
    std = np.sqrt(var)
    std3 = var * std
    abs_std = np.sqrt(abs_var)
    abs_std3 = abs_var * abs_std
    skew = (data3_mean - 3*mean*var - mean3)/std3
    kurt = (data4_mean + mean*(6*mean*data2_mean - 4*data3_mean - 3*mean3))/(var**2) - 3
    abs_skew = (abs_data3_mean - 3*abs_mean*abs_var - abs_mean3)/abs_std3
    abs_kurt = (data4_mean + abs_mean*(6*abs_mean*data2_mean - 4*abs_data3_mean - 3*abs_mean3))/(abs_var**2) - 3
    
    rms = np.sqrt(data2_mean)
    mean4 = data2_mean**2
    var2 = data4_mean - mean4
    std2 = np.sqrt(var2)
    rms_std = np.sqrt(std2)
    mean6 = mean4*data2_mean
    std6 = var2*std2
    rms_skew = np.sqrt((data6_mean - 3*data2_mean*var2 - mean6)/std6)
    rms_kurt = np.sqrt((data8_mean + data2_mean*(6*data2_mean*data4_mean - 4*data6_mean - 3*mean6))/(var2**2) - 3)
    
    return [mean, std, skew, kurt, abs_mean, abs_std, abs_skew, abs_kurt, rms, rms_std, rms_skew, rms_kurt]


class stats:

    def __init__(self, patch_lons, patch_lats, patch_size, output_path="."):
        self.header = None
        self.num_frames = 0
        self.patch_lons = patch_lons
        self.patch_lats = patch_lats
        self.patch_size = patch_size
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
            self.tracked_times.add(entry.header["START_T"])
        self.data = np.zeros((self.num_patches, self.num_patches, 9))

    def get_data_for_header(self):
        return self.patch_size, self.num_patches, self.patch_lons[0], self.patch_lons[-1], self.patch_lats[0], self.patch_lats[-1]

    def get_date(self):
        return self.date
    
    def get_num_frames(self):
        return self.num_frames
    
    def set_header(self, header):
        assert(self.header is None)
        self.header = header
        
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
                if DEBUG2:
                    color = "rb"[(i % 2 + j % 2) % 2]
                    test_plot.plot(x_pix2, -y_pix2 + int(np.sqrt(len(lons))), params=f"{color}.")
        if DEBUG2:
            test_plot.save(f"patches_{plot_file}.png")
            test_plot.close()
        self.num_frames += 1

    def save(self):
        if self.header is not None:
            self.header.add_card(fits.Card(keyword="N_FRAMES", value=self.num_frames, comment="Number of frames processed"))
            self.header.add_card(fits.Card(keyword="PATCH_SZ", value=self.patch_size, comment="Patch size in degrees"))
            self.header.add_card(fits.Card(keyword="DATADIM", value=f"8,{self.num_patches},{self.num_patches}", comment="Dimensionality of the data"))
            self.header.add_card(fits.Card(keyword="DATADIM1", value=f"stats_index", comment="First dimension is index over statistics"))
            self.header.add_card(fits.Card(keyword="DATADIM2", value=f"lon_index", comment="Second dimension is index over longitudes"))
            self.header.add_card(fits.Card(keyword="DATADIM3", value=f"lat_index", comment="Third dimension is index over longitudes"))
            self.header.add_card(fits.Card(keyword="STATS1", value="<B>", comment="Mean"))
            self.header.add_card(fits.Card(keyword="STATS2", value="std(B)", comment="Standard deviation"))
            self.header.add_card(fits.Card(keyword="STATS3", value="skew(B)", comment="Skewness"))
            self.header.add_card(fits.Card(keyword="STATS4", value="kurt(B)", comment="Kurtosis"))
            self.header.add_card(fits.Card(keyword="STATS5", value="<|B|>", comment="Mean of the absolute value"))
            self.header.add_card(fits.Card(keyword="STATS6", value="std(|B|)", comment="Standard deviation of the absolute value"))
            self.header.add_card(fits.Card(keyword="STATS7", value="skew(|B|)", comment="Skewness of the absolute value"))
            self.header.add_card(fits.Card(keyword="STATS8", value="kurt(|B|)", comment="Kurtosis of the absolute value"))
            self.header.add_card(fits.Card(keyword="STATS9", value="sqrt(<B**2>)", comment="Root mean square"))
            self.header.add_card(fits.Card(keyword="STATS10", value="sqrt(std(B**2))", comment="Root mean square of standard deviation"))
            self.header.add_card(fits.Card(keyword="STATS11", value="sqrt(skew(B**2))", comment="Root mean square of skewness"))
            self.header.add_card(fits.Card(keyword="STATS12", value="sqrt(kurt(B**2))", comment="Root mean square of kurtosis"))
            self.header.add_card(fits.Card(keyword="MIN_LON", value=self.patch_lons[0], comment="Longitude of the first data entry"))
            self.header.add_card(fits.Card(keyword="MAX_LON", value=self.patch_lons[-1], comment="Longitude of the last data entry"))
            self.header.add_card(fits.Card(keyword="MIN_LAT", value=self.patch_lats[0], comment="Latitude of the first data entry"))
            self.header.add_card(fits.Card(keyword="MAX_LAT", value=self.patch_lats[-1], comment="Latitude of the last data entry"))
            
            hdu = fits.ImageHDU(data=np.array(collect_stats_2(self.data)), header=fits.Header(self.header.get_cards()), name='Statistics')
            if hdu.header["START_T"] not in self.tracked_times:
                self.tracked_times.add(hdu.header["START_T"])
                self.storage.append(hdu)
                self.storage.flush()
        self.data = np.zeros((self.num_patches, self.num_patches, 9))
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
    
    def __init__(self, num_hrs, step, num_bursts, path, files, start_times, commit_sha, random_start_time):
        self.step = step
        self.num_hrs = num_hrs
        self.num_bursts = num_bursts
        
        self.path = path
        self.files = files
        self.commit_sha = commit_sha
        self.random_start_time=random_start_time
        
        hdul = fits.open(self.path + "/" + self.files[0], ignore_missing_end=True)
        if len(start_times) == 0:
            t_rec = hdul[1].header['T_REC']
            year, month, day, hrs, mins, secs = parse_t_rec(t_rec)
            self.start_times = []
            self.start_time = datetime(int(year), int(month), int(day), int(hrs), int(mins), int(secs))
        else:
            self.start_time = start_times[0]
            self.start_times = start_times[1:]
            year, month, day, hrs, mins, secs = parse_t_rec(str(self.start_time))
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
        print("next_frame frame_index", self.frame_index)
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
        print("next")
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
            self.hdul = fits.open(self.file, ignore_missing_end=True)
        
        self.num_frames_per_day = len(self.hdul) - 1

        print("next frame_index", self.frame_index)
        if not self.next_frame():
            return False
        print("next frame_index", self.frame_index)
        self.metadata = self.hdul[self.frame_index + 1].header

        t_rec = self.metadata['T_REC']
        
        year, month, day, hrs, mins, secs = parse_t_rec(t_rec)
        date = year + "-" + month + "-" + day
        assert(file_date == date)
        print("next file_date", file_date)
        
        self.hrs = hrs
        self.mins = mins
        self.secs = secs

        # This field is only for debug purposes
        last_obs_time = self.obs_time
        self.obs_time = datetime(int(year), int(month), int(day), int(hrs), int(mins), int(secs))
        if last_obs_time is None or last_obs_time < self.obs_time:
            self.last_obs_time = last_obs_time
        else:
            self.last_obs_time = None
        
        if self.get_obs_time() < self.get_start_time():
            print("next False 1")
            return False
        
        self.obs_time_str = f"{date} {self.hrs}:{self.mins}:{self.secs}"
        self.obs_time_str2 = f"{date}_{self.hrs}:{self.mins}:{self.secs}"
        
        self.sdo_lon = self.metadata['CRLN_OBS']
        self.sdo_lat = self.metadata['CRLT_OBS']
        self.sdo_dist = self.metadata['DSUN_OBS']


        end_track = False
        if self.get_obs_time() >= self.get_end_time():
            print("Potentially files missing")
            end_track = True
        elif self.is_tracking() and self.obs_time >= self.end_time:
            end_track = True

        if end_track:
            self.stats.save()
            self.end_tracking()
            self.num_bursts -= 1
            print("next False 2")
            return False

        if not self.is_tracking():
            self.start_tracking()
        
        print("next True")
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

    def get_start_time_str2(self):
        start_time_str = self.get_start_time_str()
        return start_time_str[:10] + "_" + start_time_str[11:]

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
        print("start_tracking", self.get_start_time(), self.get_obs_time(), self.get_last_obs_time())
        assert(self.get_start_time() <= self.get_obs_time())
        assert(self.get_last_obs_time() is None or self.get_start_time() > self.get_last_obs_time())
        assert(self.get_obs_time() < self.get_end_time())
        self.observer = frames.HeliographicStonyhurst(0.*u.deg, self.sdo_lat*u.deg, radius=self.sdo_dist*u.m, obstime=self.get_obs_time_str())

        metadata = self.get_metadata()

        header = stats_header()
        header.add_card(fits.Card(keyword="START_T", value=self.get_start_time_str(), comment="Tracking start time"))
        header.add_card(fits.Card(keyword="END_T", value=self.get_end_time_str(), comment="Tracking end time"))
        header.add_card(fits.Card(keyword="CARR_LON", value=self.get_sdo_lon(), comment="Carrington longitude of start frame"))
        header.add_card(fits.Card(keyword="GIT_SHA", value=self.commit_sha, comment="Git commit SHA"))
        header.add_card(fits.Card(keyword="UNIT", value=metadata['BUNIT'], comment="Unit of the mean and std"))        

        self.stats.set_header(header)
        
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
        
        self.xs_arcsec = grid[:, 0]
        self.ys_arcsec = grid[:, 1]
        
        grid = np.transpose([np.tile(self.xs, ny), np.repeat(self.ys, nx)])
        grid = grid[fltr]
        self.xys = grid
        
        self.dbg_stack = []
        self.num_added = 0
        
        
    def get_xs_ys_arcsec(self):
        xs_arcsec, ys_arcsec = np.array(self.xs_arcsec), np.array(self.ys_arcsec)
        del self.xs_arcsec
        del self.ys_arcsec
        return xs_arcsec, ys_arcsec

    def get_xys(self):
        return self.xys

    def transform_index(self, i):
        for old_indices, new_indices in self.dbg_stack:
            for k in range(len(old_indices)):
                if old_indices[k] == i:
                    i = new_indices[k]
                    break
            #num = 0
            #for ind in indices_deleted:
            #    if ind < i:
            #        num += 1
            #i -= num
        return i

    def get_nx_ny(self):
        return self.nx, self.ny

    def frame_processed(self, xs_arcsec, ys_arcsec, observer, dbg_info):
        self.xs_arcsec = xs_arcsec
        self.ys_arcsec = ys_arcsec
        self.observer = observer
        if DEBUG:
            self.dbg_stack.append(dbg_info)

    def end_tracking(self):
        self.observer = None
        if self.random_start_time:
            self.start_time = get_random_start_time(self.step)
        elif len(self.start_times) > 0:
            self.start_time = self.start_times[0]
            self.start_times = self.start_times[1:]
        else:
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

def fix_sampling(x_pix, y_pix, xs_arcsec, ys_arcsec, lons, lats, xys, sdo_lon, observer, pix_dict, start_index, length, image_params):
        print("fix_sampling 1")
        min_y = int(np.nanmin(y_pix))
        max_y = int(np.nanmax(y_pix))
        xys = xys[(xys[:, 1] >= min_y) * (xys[:, 1] <= max_y)]
        print("fix_sampling 1", min_y, max_y)
        old_indices = []
        new_indices = []
        num_removed = 0
        east_limb_pixs = np.ones(max_y - min_y + 1, dtype=int)*int(np.nanmax(x_pix))
        west_limb_pixs = np.zeros(max_y - min_y + 1, dtype=int)
        #new_entries = []
        removed_entries = []
        j = 0
        for i in range(len(x_pix)):#range(len(x_pix) -1, -1, -1):
            assert(i == j + num_removed)
            if not np.isnan(x_pix[i]) and not np.isnan(y_pix[i]):
                x, y = int(x_pix[i]), int(y_pix[i])
                if east_limb_pixs[y - min_y] > x:
                    east_limb_pixs[y - min_y] = x
                if west_limb_pixs[y - min_y] < x:
                    west_limb_pixs[y - min_y] = x

                ind = pix_dict[x][y]
                if DEBUG:
                    old_indices.append(i + start_index)
                if ind < 0:
                    pix_dict[x][y] = j + length
                    #new_entries.append((x, y))
                    if DEBUG:
                        new_indices.append(j + length)
                    j += 1                        
                else:
                    x_pix[i] = -1
                    #del x_pix[j]
                    #del y_pix[j]
                    #del xs_arcsec[j]
                    #del ys_arcsec[j]
                    #del lons[j]
                    #del lats[j]
                    #for (x, y) in new_entries:
                    #    pix_dict[x][y] -= 1
                    if DEBUG:
                        new_indices.append(ind)
                        #for j in range(len(new_indices)):
                        #    if new_indices[j] > i + length:
                        #        new_indices[j] -= 1
                    num_removed += 1
            else:
                j += 1
        l = len(x_pix) - num_removed
        #new_entries.clear()
        print("fix_sampling 2")
        #x_pix = np.delete(x_pix, indices_to_delete)
        #y_pix = np.delete(y_pix, indices_to_delete)
        #xs_arcsec = np.delete(xs_arcsec, indices_to_delete)
        #ys_arcsec = np.delete(ys_arcsec, indices_to_delete)
        #lons = np.delete(lons, indices_to_delete)
        #lats = np.delete(lats, indices_to_delete)
        print("Number of pixels removed", num_removed)
        
        added_x_pix = [-1.]*added_pix_list_size
        added_y_pix = [-1]*added_pix_list_size
        print("fix_sampling 3")

        i = 0
        for x, y in xys:
            x, y = int(x), int(y)
            if x >= east_limb_pixs[y - min_y] and x <= west_limb_pixs[y - min_y]:
                if pix_dict[x][y] < 0:
                    if i >= len(added_x_pix):
                        added_x_pix.extend([-1]*added_pix_list_size)
                        added_y_pix.extend([-1]*added_pix_list_size)
                    added_x_pix[i] = x
                    added_y_pix[i] = y
                    pix_dict[x][y] = length + l + i
                    i += 1
        print("fix_sampling 4")
                    
        added_x_pix = np.asarray(added_x_pix)
        added_y_pix = np.asarray(added_y_pix)
        added_x_pix = added_x_pix[added_x_pix >= 0]
        added_y_pix = added_y_pix[added_y_pix >= 0]
        print("Number of pixels added", len(added_x_pix))
        x_pix.extend(added_x_pix)
        y_pix.extend(added_y_pix)
        #x_pix = np.append(x_pix, added_x_pix)
        #y_pix = np.append(y_pix, added_y_pix)
        dx, dy, xc, yc, cos_a, sin_a, arcsecs_per_pix_x, arcsecs_per_pix_y = image_params
        added_xs_arcsec, added_ys_arcsec = pix_to_image(added_x_pix, added_y_pix, dx, dy, xc, yc, cos_a, sin_a, arcsecs_per_pix_x, arcsecs_per_pix_y)
        
        xs_arcsec.extend(added_xs_arcsec.tolist())
        ys_arcsec.extend(added_ys_arcsec.tolist())
        #xs_arcsec = np.append(xs_arcsec, added_xs_arcsec)
        #ys_arcsec = np.append(ys_arcsec, added_ys_arcsec)

        added_xs_arcsec, added_ys_arcsec = added_xs_arcsec*u.arcsec, added_ys_arcsec*u.arcsec
                
        c1 = SkyCoord(added_xs_arcsec, added_ys_arcsec, frame=frames.Helioprojective, observer=observer)
        c2 = c1.transform_to(frames.HeliographicCarrington)
        added_lons = c2.lon.value - sdo_lon
        added_lats = c2.lat.value
            
        print("fix_sampling 5")
        
        lons.extend(added_lons.tolist())
        lats.extend(added_lats.tolist())
        #lons = np.append(lons, added_lons)
        #lats = np.append(lats, added_lats)
                
        assert(len(lons) == len(lats))
        assert(len(lons) == len(x_pix))
        assert(len(x_pix) == len(y_pix))
        assert(len(x_pix) == len(xs_arcsec))
        assert(len(xs_arcsec) == len(ys_arcsec))
        print("fix_sampling 6")
        take_snapshot("fix_sampling")
        return (old_indices, new_indices)


class track:

    def __init__(self, input_path, output_path, files, num_hrs=8, step=1, num_bursts=-1, num_patches=100, patch_size=15, 
                 stats_dbg = None, stats_file_mode="burst", start_times=[], commit_sha="", random_start_time=True):
        assert(stats_file_mode == "burst" or stats_file_mode == "day" or stats_file_mode == "month" or stats_file_mode == "year")
        self.num_patches = num_patches
        self.patch_size = patch_size
        
        # Here we already assume that tracked sequences are 8hrs
        assert(self.patch_size < 160)
        self.patch_lons = np.linspace(-80, 80 - patch_size, num_patches)
        self.patch_lats = self.patch_lons
        
        print(f"Input path: {input_path}")
        print(f"Output path: {output_path}")
        
        self.state = state(num_hrs, step, num_bursts, input_path, files, start_times, commit_sha, random_start_time)
        if stats_dbg is None:
            sts = stats(self.patch_lons, self.patch_lats, self.patch_size, output_path)
        else:
            sts = stats_dbg
        self.state.set_stats(sts)
        sts.init(self.state.get_start_time_str2())

        #metadata = self.state.get_metadata()
        #self.nx = metadata['NAXIS1']
        #self.ny = metadata['NAXIS2']
        
        #self.xs = (np.arange(1, self.nx + 1)).astype(float)
        #self.ys = (np.arange(1, self.ny + 1)).astype(float)

        self.stats_file_mode = stats_file_mode

    def process_frame(self):

        print("process_frame 1")
        metadata = self.state.get_metadata()
        
        obs_time = self.state.get_obs_time_str()
        print("obs_time", obs_time)

        data = self.state.get_data()
        print("process_frame 2")
        if DEBUG:
            ctype1 = metadata['CTYPE1']
            ctype2 = metadata['CTYPE2']
            assert(ctype1 == "HPLN-TAN" and ctype2 == "HPLT-TAN")
    
            cunit1 = metadata['CUNIT1']
            cunit2 = metadata['CUNIT2']
            assert(cunit1 == "arcsec" and cunit2 == "arcsec")

        xs_arcsec_all_last, ys_arcsec_all_last = self.state.get_xs_ys_arcsec()
        print("process_frame 3")
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
        print("process_frame 4")

        observer = self.state.get_observer()
        observer_i = frames.HeliographicStonyhurst(0.*u.deg, self.state.get_sdo_lat()*u.deg, radius=self.state.get_sdo_dist()*u.m, obstime=obs_time)
        
        pix_dict = (np.ones((nx, ny), dtype=int)*-1).tolist()
        chunk_size = int(len(xs_arcsec_all_last)/num_chunks)
        
        xys = self.state.get_xys()
        #xs_arcsec_head, xs_arcsec_tail = np.array([]), np.array([])
        #ys_arcsec_head, ys_arcsec_tail = np.array([]), np.array([])
        
        xs_arcsec_all = np.array([])
        ys_arcsec_all = np.array([])
       
        
        if DEBUG:
            #x_pix_head, x_pix_tail = np.array([]), np.array([])
            #y_pix_head, y_pix_tail = np.array([]), np.array([])
            #lons_head, lons_tail = np.array([]), np.array([])
            #lats_head, lats_tail = np.array([]), np.array([])
            
            x_pix_all = np.array([])
            y_pix_all = np.array([])
            lons_all = np.array([])
            lats_all = np.array([])
        
        dbg_info_all = ([], [])
        
        start_index = 0
        for chunk_index in range(num_chunks):
            if chunk_index < num_chunks - 1:
                end_index = chunk_size
            else:
                end_index = len(xs_arcsec_all_last)
            xs_arcsec, ys_arcsec = xs_arcsec_all_last[:end_index], ys_arcsec_all_last[:end_index]
            xs_arcsec_all_last = xs_arcsec_all_last[end_index:]
            ys_arcsec_all_last = ys_arcsec_all_last[end_index:]
            
            c1 = SkyCoord(xs_arcsec*u.arcsec, ys_arcsec*u.arcsec, frame=frames.Helioprojective, observer=observer)
            c2 = c1.transform_to(frames.HeliographicCarrington)
            print("process_frame 5", chunk_index)
            lons = c2.lon.value - self.state.get_sdo_lon()
            lats = c2.lat.value
                        
            #c3 = [SkyCoord(c2.lon[i], c2.lat[i], frame=frames.Helioprojective, observer=observer_i) for i in range(len(c2.lon))]
            c3 = SkyCoord(c2.lon, c2.lat, frame=frames.HeliographicCarrington, observer=observer_i, obstime=obs_time)
            c4 = c3.transform_to(frames.Helioprojective)
            
            xs_arcsec = c4.Tx.value
            ys_arcsec = c4.Ty.value
                        
            print("process_frame 6", chunk_index)
            
            x_pix, y_pix = image_to_pix(xs_arcsec, ys_arcsec, dx, dy, xc, yc, cos_a, sin_a, coef_x, coef_y)
            x_pix = np.round(x_pix)
            y_pix = np.round(y_pix)
            
            print("process_frame 7", chunk_index)
            x_pix = x_pix.tolist()
            y_pix = y_pix.tolist()
            xs_arcsec = xs_arcsec.tolist()
            ys_arcsec = ys_arcsec.tolist()
            lons = lons.tolist()
            lats = lats.tolist()                                                            
                                                                        
            dbg_info = fix_sampling(x_pix, y_pix, xs_arcsec, ys_arcsec, lons, lats, 
                  xys, self.state.get_sdo_lon(), observer_i, pix_dict, start_index, len(xs_arcsec_all),
                  (dx, dy, xc, yc, cos_a, sin_a, arcsecs_per_pix_x, arcsecs_per_pix_y))
            
            lons = np.asarray(lons)
            lats = np.asarray(lats)
            x_pix = np.asarray(x_pix)
            y_pix = np.asarray(y_pix)
            xs_arcsec = np.asarray(xs_arcsec)
            ys_arcsec = np.asarray(ys_arcsec)
            
            fltr = (x_pix >= 0)
            x_pix = x_pix[fltr]
            y_pix = y_pix[fltr]
            lons = lons[fltr]
            lats = lats[fltr]
            xs_arcsec = xs_arcsec[fltr]
            ys_arcsec = ys_arcsec[fltr]            
                        
            self.state.get_stats().process_frame(lons, lats, x_pix, y_pix, data, obs_time=self.state.get_obs_time(), plot_file=self.state.get_obs_time_str2() + "_" + str(chunk_index))

            xs_arcsec_all = np.append(xs_arcsec_all, xs_arcsec)
            ys_arcsec_all = np.append(ys_arcsec_all, ys_arcsec)

            #xs_arcsec_head = np.append(xs_arcsec_head, xs_arcsec[:split_point])
            #xs_arcsec_tail = np.append(xs_arcsec_tail, xs_arcsec[split_point:])
            #xs_arcsec.clear()
            #ys_arcsec_head = np.append(ys_arcsec_head, ys_arcsec[:split_point])
            #ys_arcsec_tail = np.append(ys_arcsec_tail, ys_arcsec[split_point:])
            #ys_arcsec.clear()
            
            if DEBUG:
                dbg_info_all[0].extend(dbg_info[0])
                dbg_info_all[1].extend(dbg_info[1])
                
                x_pix_all = np.append(x_pix_all, x_pix)
                y_pix_all = np.append(y_pix_all, y_pix)
                
                lons_all = np.append(lons_all, lons)
                lats_all = np.append(lats_all, lats)
            
                #x_pix_head = np.append(x_pix_head, x_pix[:split_point])
                #x_pix_tail = np.append(x_pix_tail, x_pix[split_point:])
                #y_pix_head = np.append(y_pix_head, y_pix[:split_point])
                #y_pix_tail = np.append(y_pix_tail, y_pix[split_point:])
                #lons_head = np.append(lons_head, lons[:split_point])
                #lons_tail = np.append(lons_tail, lons[split_point:])
                #lats_head = np.append(lats_head, lats[:split_point])
                #lats_tail = np.append(lats_tail, lats[split_point:])
    
            print("process_frame 8", chunk_index)

            start_index += chunk_size
        
            sys.stdout.flush()
        take_snapshot("process_frame 1")
        
        #xs_arcsec = np.concatenate([xs_arcsec_head, xs_arcsec_tail])
        #del xs_arcsec_head
        #del xs_arcsec_tail
        #ys_arcsec = np.concatenate([ys_arcsec_head, ys_arcsec_tail])
        #del ys_arcsec_head
        #del ys_arcsec_tail
        self.state.frame_processed(xs_arcsec_all, ys_arcsec_all, observer_i, dbg_info_all)

        #if DEBUG:
        #    x_pix = np.concatenate([x_pix_head, x_pix_tail])
        #    del x_pix_head
        #    del x_pix_tail
        #    y_pix = np.concatenate([y_pix_head, y_pix_tail])
        #    del y_pix_head
        #    del y_pix_tail
        #    lons = np.concatenate([lons_head, lons_tail])
        #    del lons_head
        #    del lons_tail
        #    lats = np.concatenate([lats_head, lats_tail])
        #    del lats_head
        #    del lats_tail

        #######################
        if DEBUG:
            data_for_plot = np.empty((ny, nx), dtype=np.float32)
            data_for_plot[:, :] = np.nan
        
            xys = self.state.get_xys()
            for l in range(len(xys)):
                l1 = self.state.transform_index(l)
                if l1 >= 0:
                    j = int(xys[l, 1])
                    k = int(xys[l, 0])
                    if np.isnan(y_pix_all[l1]) or np.isnan(x_pix_all[l1]):
                        data_for_plot[j, k] = np.nan
                    else:
                        data_for_plot[j, k] = data[int(y_pix_all[l1]), int(x_pix_all[l1])]

        print("process_frame 9")
        if DEBUG:
            test_plot = plot.plot(nrows=1, ncols=1, size=plot.default_size(data_for_plot.data.shape[1]//8, data_for_plot.data.shape[0]//8))
            test_plot.colormap(data_for_plot, cmap_name="bwr", show_colorbar=True)
            suffix = self.state.get_obs_time_str2()
            test_plot.save(f"frame_{suffix}.png")
            test_plot.close()

        take_snapshot("process_frame 2")
        sys.stdout.flush()
        if DEBUG:
            return lons_all, lats_all, x_pix_all, y_pix_all, data

    '''
    def process_frame(self):
        print("process_frame 1")
        lons, lats, x_pix, y_pix, data = self.transform()

        self.state.get_stats().process_frame(lons, lats, x_pix, y_pix, data, obs_time=self.state.get_obs_time(), plot_file=self.state.get_obs_time_str2())
        
        print("process_frame 2")
        #self.state.frame_processed()

        #print(f"time 7: {time.perf_counter()}")        
        sys.stdout.flush()
    '''    
        
    def track(self):
        while not self.state.is_done():
            take_snapshot()
            if self.state.next():
                self.process_frame()
            if self.state.is_done():
                break
            if not self.state.is_tracking():
                create_new_stats = False
                date = self.state.get_start_time_str2()
                stats_date = self.state.get_stats().get_date()
                print("track: date, stats_date", date, stats_date)
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
                    self.state.get_stats().init(self.state.get_start_time_str2())
        self.state.close()

def my_print(*args):
    std_print(f"{time.perf_counter()} {pid}:", *args)

if (__name__ == '__main__'):
    
    if PROFILE:
        tracemalloc.start()
    
    pid = os.getpid()
    std_print = print
    print = my_print
    
    input_path = '.'
    output_path = '.'
    num_hrs = 8 # Duration of tracking
    step = 1 # Step in hours between tracked sequences of num_hrs length
    num_bursts = -1 # For how many days to run the script
    num_patches = 100
    patch_size = 15
    
    commit_sha = ""
    random_start_time = True
    
    if os.path.isfile("params.txt"):
        with open("params.txt", "r") as file:
            line = file.readlines()[-1]
            argv = line.strip().split(" ")
            i = 0
    else:
        argv = sys.argv
        i = 1

    if len(argv) > i:
        input_path = argv[i]
    i += 1
    if len(argv) > i:
        output_path = argv[i]
    i += 1
    if len(argv) > i:
        start_time = argv[i]
    i += 1
    if len(argv) > i:
        num_hrs = float(argv[i])
    i += 1
    if len(argv) > i:
        step = float(argv[i])
    i += 1
    if len(argv) > i:
        num_bursts = int(argv[i])
    i += 1
    if len(argv) > i:
        num_patches = int(argv[i])
    i += 1
    if len(argv) > i:
        patch_size = float(argv[i])
    i += 1
    if len(argv) > i:
        commit_sha = argv[i]
    i += 1
    if len(argv) > i:
        random_start_time = bool(int(argv[i]))
    print("Commit SHA", commit_sha)
    assert(step <= 24)
    
    if random_start_time:
        start_time = str(get_random_start_time(step))
        print("Overriding start_time with", start_time)
    start_times = []
    if len(start_time) < 4:        
        all_start_times = []
        start_time = None
        for root, dirs, files in os.walk(output_path):
            for file in files:
                if file[-5:] == ".fits":
                    try:
                        hdul = fits.open(output_path + "/" + file, ignore_missing_end=True)
                        for i in range(len(hdul)):
                            all_start_times.append(hdul[i].header["START_T"])
                        hdul.close()
                    except:
                        # Corrupted fits file
                        pass
        all_start_times.sort()
        if len(all_start_times) > 0:
            last_start_time = all_start_times[0]
            year, month, day, hrs, mins, secs = parse_t_rec(last_start_time)
            last_start_time = datetime(int(year), int(month), int(day), int(hrs), int(mins), int(secs))
            for t in all_start_times:
                year, month, day, hrs, mins, secs = parse_t_rec(t)
                t = datetime(int(year), int(month), int(day), int(hrs), int(mins), int(secs))
                next_time = last_start_time + timedelta(hours=step)
                while next_time < t:
                    start_times.append(next_time)
                    next_time = next_time + timedelta(hours=step)
                last_start_time = t
            year, month, day, hrs, mins, secs = parse_t_rec(all_start_times[-1])
            last_start_time = datetime(int(year), int(month), int(day), int(hrs), int(mins), int(secs))
            start_times.append(last_start_time + timedelta(hours=step))
            
    else:
        if len(start_time) <= 16:
            if len(start_time) <= 13:
                if len(start_time) <= 10:
                    if len(start_time) <= 7:
                        if len(start_time) == 4:
                            start_time += "-01"
                        start_time += "-01"
                    start_time += " 00"
                start_time += ":00"
            start_time += ":00"

        year, month, day, hrs, mins, secs = parse_t_rec(start_time)
        start_time = datetime(int(year), int(month), int(day), int(hrs), int(mins), int(secs))
        #start_time = start_time + timedelta(hours=step)
        start_times.append(start_time)

    if len(start_times) > 0:        
        start_date = str(start_times[0])[:10]
    else:
        start_date = ""
    
    print("Start time", str(start_time))
        
    all_files = list()
    
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file >= start_date:
                all_files.append(file)
    all_files.sort()
    
    take_snapshot()
    
    tr = track(input_path=input_path, output_path=output_path, files=all_files, num_bursts=num_bursts, num_hrs=num_hrs, step=step, 
               num_patches=num_patches, patch_size=patch_size, start_times=start_times, commit_sha=commit_sha, random_start_time=random_start_time)
    tr.track()

