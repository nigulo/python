import sys
sys.path.append('..')
sys.path.append('../..')
import numpy as np
import unittest
import track

import plot
import os
import sys

from astropy.io import fits
from datetime import datetime, timedelta

class stats(track.stats):
    def __init__(self, patch_lons, patch_lats, patch_size):
        super(stats, self).__init__(patch_lons, patch_lats, patch_size)
        self.obs_times = list()
 
    def process_frame(self, lons, lats, x_pix, y_pix, data, obs_time, plot_file=None):
        super(stats, self).process_frame(lons, lats, x_pix, y_pix, data, obs_time, plot_file)
        self.obs_times.append(obs_time)
        print("obs_time", obs_time)

    def save(self):
        super(stats, self).save()
        print(self.obs_times, self.obs_times_expected)
        np.testing.assert_equal(len(self.obs_times), len(self.obs_times_expected[0]))
        for i in range(len(self.obs_times)):
            np.testing.assert_equal(self.obs_times[i], self.obs_times_expected[0][i])
        self.obs_times_expected = self.obs_times_expected[1:]
        self.obs_times = list()
        
    def set_obs_times_expected(self, obs_times_expected):
        self.obs_times_expected = obs_times_expected
        
'''
class test_track(unittest.TestCase):
    
    def test(self):
        
        try:
            os.remove("2013-02-03 00:00:00.fits")
        except:
            pass
        
        tr = track.track(".", ".", ["2013-02-03.fits"], num_hrs=24, step=1, num_bursts=1, num_patches=100, patch_size=15)
        
        hdul = fits.open("2013-02-03.fits")
        data_0 = hdul[1].data
 
        test_plot0 = plot.plot(nrows=1, ncols=1, size=plot.default_size(1000, 1000))
        test_plot0.colormap(data_0, show_colorbar=True)
        test_plot0.save(f"track{0}.png")
        #test_plot0.close()
       
        for i in range(12):
            tr.state.next()
            lons, lats, x_pix, y_pix, data = tr.transform()
            #tr.state.frame_processed()
            
            #test_plot = plot.plot(nrows=1, ncols=1, size=plot.default_size(1000, 1000))
            #test_plot.colormap(data, show_colorbar=True)
            #test_plot.save(f"track{i+1}.png")
            #test_plot.close()
            
            data_tracked = np.zeros_like(data)
            
            xys = tr.state.get_xys()
            for l in range(len(x_pix)):
                j = xys[l, 1]
                k = xys[l, 0]
                if (not np.isnan(y_pix[l])) and (not np.isnan(x_pix[l])):
                    data_tracked[j, k] = data[int(y_pix[l]), int(x_pix[l])]
            
            test_plot = plot.plot(nrows=1, ncols=1, size=plot.default_size(1000, 1000))
            test_plot.colormap(data_tracked, show_colorbar=True)
            test_plot.save(f"track{i+1}.png")
            test_plot.close()
'''

class test_stats(unittest.TestCase):


    def test1(self):
        num_patches = 5
        patch_size = 20
        try:
            os.remove("2013-02-03 00:00:00.fits")
        except:
            pass
        try:
            os.remove("2013-02-03 04:00:00.fits")
        except:
            pass
        try:
            os.remove("2013-02-03 08:00:00.fits")
        except:
            pass
        try:
            os.remove("2013-02-03 12:00:00.fits")
        except:
            pass

        patch_lons = np.linspace(-80, 80 - patch_size, num_patches)
        patch_lats = patch_lons
        obs_times_expected = list()
        for start_hr in [0, 4, 8, 12]:
            obs_times_expected_i = list()
            for hr in np.arange(start_hr, start_hr+12, 2):
                obs_times_expected_i.append(datetime(2013, 2, 3, hr, 0, 0))
            obs_times_expected.append(obs_times_expected_i)

        sts = stats(patch_lons, patch_lats, patch_size)
        sts.set_obs_times_expected(obs_times_expected)
        tr = track.track(".", ".", ["2013-02-03.fits"], num_hrs=12, step=4, num_bursts=4, num_patches=num_patches, patch_size=patch_size, stats_dbg = sts)

        hdul = fits.open("2013-02-03.fits")
        print(len(hdul))
        assert(len(hdul) == 13)
        print("First time", hdul[1].header["T_REC"])

        tr.track()
        
        np.testing.assert_equal(len(tr.state.get_stats().obs_times_expected), 0)

        i = 1
        storages = list()
        for f in ["2013-02-03 00:00:00.fits", "2013-02-03 04:00:00.fits", "2013-02-03 08:00:00.fits", "2013-02-03 12:00:00.fits"]:
            np.testing.assert_equal(os.path.isfile(f), True)
            storage = fits.open(f)
            storages.append(storage)
    
            #storage = tr.state.get_stats().storage
            np.testing.assert_equal(len(storage), 1)
            t_rec = hdul[i].header["T_REC"]
            print(i, t_rec)
            year, month, day, hrs, mins, secs = track.parse_t_rec(t_rec)
            start_time = datetime(int(year), int(month), int(day), int(hrs), int(mins), int(secs))
            end_time = start_time + timedelta(hours=12)
            np.testing.assert_equal(storage[0].header["START_TIME"], str(start_time)[:19])
            np.testing.assert_equal(storage[0].header["END_TIME"], str(end_time)[:19])
            np.testing.assert_equal(storage[0].header["CLON"], hdul[i].header["CRLN_OBS"])
            
            np.testing.assert_equal(storage[0].data.shape, (10, num_patches, num_patches))
            for lon_i in np.arange(num_patches):
                for lat_i in np.arange(num_patches):
                    mean = storage[0].data[0, lon_i, lat_i]
                    std = storage[0].data[1, lon_i, lat_i]
                    if lon_i == num_patches-1 and lat_i == num_patches-1:
                        #We omit the last patch
                        continue
                    if (lon_i == 0 or lon_i == 4) and lat_i == num_patches-1:
                        #We omit the last patch
                        continue
                    if lon_i == num_patches-1:
                        np.testing.assert_almost_equal(mean, lon_i * num_patches + lat_i, 0)
                        #np.testing.assert_almost_equal(std, 0, 0)
                        continue
                    np.testing.assert_almost_equal(mean, lon_i * num_patches + lat_i, 1)
                    #np.testing.assert_almost_equal(std, 0, 1)
            i += 2

        # Tracking second time should have no effect
        tr = track.track(".", ".", ["2013-02-03.fits"], num_hrs=12, step=4, num_bursts=4, num_patches=num_patches, patch_size=patch_size)
        tr.track()
        
        i = 0
        for f in ["2013-02-03 00:00:00.fits", "2013-02-03 04:00:00.fits", "2013-02-03 08:00:00.fits", "2013-02-03 12:00:00.fits"]:
            np.testing.assert_equal(os.path.isfile(f), True)
            storage2 = fits.open(f)
            storage = storages[i]
        
            np.testing.assert_equal(len(storage2), 1)
            np.testing.assert_equal(storage2[0].header["START_TIME"], storage[0].header["START_TIME"])
            np.testing.assert_equal(storage2[0].header["END_TIME"], storage[0].header["END_TIME"])
            np.testing.assert_equal(storage2[0].header["CLON"], storage[0].header["CLON"])
            np.testing.assert_array_equal(storage2[0].data, storage[0].data)
            i += 1
            storage.close()
            storage2.close()

        hdul.close()


    def test2(self):
        num_patches = 5
        patch_size = 20
        try:
            os.remove("2013-02-03 00:00:00.fits")
        except:
            pass
        try:
            os.remove("2013-02-04 03:00:00.fits")
        except:
            pass
        
        patch_lons = np.linspace(-80, 80 - patch_size, num_patches)
        patch_lats = patch_lons
        obs_times_expected = list()
        for day in [3, 4]:
            if day == 3:
                start_hrs = [0, 10, 18]
            else:
                start_hrs = [4, 12, 22]
            for start_hr in start_hrs:
                obs_times_expected_i = list()
                end_hr = start_hr+8
                if day == 4 and start_hr == 22:
                    end_hr = 24
                for hr in np.arange(start_hr, end_hr, 2):
                    day1 = day
                    if hr >= 24:
                        hr %= 24
                        day1 = day + 1
                    obs_times_expected_i.append(datetime(2013, 2, day1, hr, 0, 0))
                obs_times_expected.append(obs_times_expected_i)

        sts = stats(patch_lons, patch_lats, patch_size)
        sts.set_obs_times_expected(obs_times_expected)
        
        tr = track.track(".", ".", ["2013-02-03.fits", "2013-02-04.fits"], num_hrs=8, step=9, num_bursts=6, num_patches=num_patches, 
                         patch_size=patch_size, stats_dbg=sts, stats_file_mode="day")

        hdul1 = fits.open("2013-02-03.fits")
        hdul2 = fits.open("2013-02-04.fits")

        tr.track()
        #while not tr.state.is_done():
        #    if tr.state.next():
        #        lons, lats, x_pix, y_pix, data = tr.transform()
        #        tr.state.get_stats().process_frame(lons, lats, x_pix, y_pix, data, obs_time=tr.state.get_obs_time(), plot_file=tr.state.get_obs_time_str2())
        #    else:
        #        if tr.state.is_done():
        #            break
        #    #tr.state.frame_processed()
        #tr.state.close()
        np.testing.assert_equal(len(tr.state.get_stats().obs_times_expected), 0)

        np.testing.assert_equal(os.path.isfile("2013-02-03 00:00:00.fits"), True)
        storage1 = fits.open("2013-02-03 00:00:00.fits")

        np.testing.assert_equal(os.path.isfile("2013-02-04 03:00:00.fits"), True)
        storage2 = fits.open("2013-02-04 03:00:00.fits")

        #storage = tr.state.get_stats().storage
        np.testing.assert_equal(len(storage1), 3)
        np.testing.assert_equal(len(storage2), 3)
        
        for j in range(3):
            hdul = hdul1
            storage = storage1
            if j == 0:
                i = 1
                start_time = "2013-02-03 00:00:00"
                end_time = "2013-02-03 08:00:00"
            elif j == 1:
                i = 6
                start_time = "2013-02-03 09:00:00"
                end_time = "2013-02-03 17:00:00"
            else:
                i = 10
                start_time = "2013-02-03 18:00:00"
                end_time = "2013-02-04 02:00:00"

            np.testing.assert_equal(storage[j].header["START_TIME"], start_time)
            np.testing.assert_equal(storage[j].header["END_TIME"], end_time)
            np.testing.assert_equal(storage[j].header["CLON"], hdul[i].header["CRLN_OBS"])
        
        for j in range(3):
            hdul = hdul2
            storage = storage2
            if j == 0:
                i = 3
                start_time = "2013-02-04 03:00:00"
                end_time = "2013-02-04 11:00:00"
            elif j == 1:
                i = 7
                start_time = "2013-02-04 12:00:00"
                end_time = "2013-02-04 20:00:00"
            else:
                i = 12
                start_time = "2013-02-04 21:00:00"
                end_time = "2013-02-05 05:00:00"

            np.testing.assert_equal(storage[j].header["START_TIME"], start_time)
            np.testing.assert_equal(storage[j].header["END_TIME"], end_time)
            np.testing.assert_equal(storage[j].header["CLON"], hdul[i].header["CRLN_OBS"])
        
        storage1.close()
        storage2.close()
        
        hdul1.close()
        hdul2.close()


if __name__ == '__main__':
    unittest.main()