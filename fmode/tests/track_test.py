import sys
sys.path.append('..')
sys.path.append('../..')
import numpy as np
import unittest
import track

import plot

from astropy.io import fits


class test_track(unittest.TestCase):
    
    def test(self):
        tr = track.track(".", ["2013-02-03.fits"], num_days=1, num_frames=12, step=10, num_patches=100, patch_size=15)
        
        hdul = fits.open("2013-02-03.fits")
        data_0 = hdul[1].data
 
        test_plot0 = plot.plot(nrows=1, ncols=1, size=plot.default_size(1000, 1000))
        test_plot0.colormap(data_0, show_colorbar=True)
        test_plot0.save(f"track{0}.png")
        #test_plot0.close()
       
        for i in range(12):
            tr.state.next()
            lons, lats, x_pix, y_pix, data = tr.transform()
            tr.state.frame_processed()
            
            test_plot = plot.plot(nrows=1, ncols=1, size=plot.default_size(1000, 1000))
            test_plot.colormap(data, show_colorbar=True)
            test_plot.save(f"track{i+1}.png")
            #test_plot.close()
            
            data_tracked = np.zeros_like(data)
            
            l = 0
            for j in range(data_0.shape[0]):
                for k in range(data_0.shape[1]):
                    if (not np.isnan(y_pix[l])) and (not np.isnan(x_pix[l])):
                        data_tracked[j, k] = data[int(y_pix[l]), int(x_pix[l])]
                        #if data[int(y_pix[l]), int(x_pix[l])] != data_0[j, k]:
                        #    test_plot.plot([int(x_pix[l])], [data_0.shape[0] - 1 - int(y_pix[l])], params="r+")
                        #    test_plot.plot([k], [data_0.shape[0] - j - 1], params="go")
                        #    test_plot.save(f"track{i+1}.png")
                        #
                        #    test_plot0.plot([int(x_pix[l])], [data_0.shape[0] - 1 - int(y_pix[l])], params="r+")
                        #    test_plot0.plot([k], [data_0.shape[0] - j - 1], params="go")
                        #    test_plot0.save(f"track{0}.png")
                        #    print("j, k", j, k, data[int(y_pix[l]), int(x_pix[l])], data_0[j, k])
                        #if i == 0:
                        #    if int(y_pix[l]) != j or int(x_pix[l]) != k:
                        #        test_plot.plot([int(x_pix[l])], [int(y_pix[l])], params="r+")
                        #        test_plot.plot([k], [j], params="go")
                        #        test_plot.save(f"track{i+1}.png")
                        #        print("j, k", j, k)
                        #        
                        #    assert(int(y_pix[l]) == j and int(x_pix[l]) == k)
                        pass
                        #np.testing.assert_equal(data[int(y_pix[l]), int(x_pix[l])], data_0[j, k])
                    l += 1
            
            test_plot = plot.plot(nrows=1, ncols=1, size=plot.default_size(1000, 1000))
            test_plot.colormap(data_tracked, show_colorbar=True)
            test_plot.save(f"track{i+1}.png")
            test_plot.close()
            
        
        

'''
class test_stats_get_indices(unittest.TestCase):
    
    def get_indices(self, lon, lat):
        indices = []
        for i in range(len(self.patch_lons)):
            for j in range(len(self.patch_lats)):
                if lon >= self.patch_lons[i] and lon < self.patch_lons[i] + self.patch_size and lat >= self.patch_lats[j] and lat < self.patch_lats[j] + self.patch_size:
                    indices.append(np.array([i, j]))
        return np.asarray(indices)
        
    
    def test(self):
        self.patch_lons = np.linspace(-80, 65, 100)
        self.patch_lats = self.patch_lons
        self.patch_size = 15
        s = track.stats("2021-02-01", self.patch_lons, self.patch_lats, self.patch_size)
        
        lons = np.random.uniform(-80, 65, size=100)
        lats = np.random.uniform(-80, 65, size=100)
        for lon in lons:
            for lat in lats:
                lon_start, lon_end, lat_start, lat_end = s.get_indices(lon, lat)                
                indices_expected = self.get_indices(lon, lat)
                
                #print("indices_expected", indices_expected)
                
                lon_indices = np.arange(lon_start, lon_end+1)
                lat_indices = np.arange(lat_start, lat_end+1)
                
                #print("lon_indices", lon_indices)
                #print("lat_indices", lat_indices)
                indices = np.transpose([np.repeat(lon_indices, len(lat_indices)), np.tile(lat_indices, len(lon_indices))])
                #print("indices", indices)

                np.testing.assert_array_equal(indices, indices_expected)
'''

if __name__ == '__main__':
    unittest.main()