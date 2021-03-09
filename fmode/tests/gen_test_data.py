import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../utils"))
sys.path.append('../..')
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
import track
import misc

downsample_coef = .1
num_chunks = 10

for file_date in ["2013-02-03", "2013-02-04"]:

    hdul = fits.open(f"{file_date}_hmi.M_720s.fits")
    
    try:
        os.remove(f"{file_date}.fits")
    except:
        pass
    
    output = fits.open(f"{file_date}.fits", mode="append")
    hdu = fits.ImageHDU(data=None, header=None)
    output.append(hdu)
    
    metadata = hdul[1].header
    
    t_rec = metadata['T_REC']
    
    date = t_rec[:10]
    date = date[:4] + "-" + date[5:7] + "-" +date[8:10]
    
    hrs = t_rec[11:13]
    mins = t_rec[14:16]
    secs = t_rec[17:19]
    
    obs_time_0 = f"{date} {hrs}:{mins}:{secs}"    
    
    sdo_lon_0 = metadata['CRLN_OBS']
    sdo_lat_0 = metadata['CRLT_OBS']
    sdo_dist_0 = metadata['DSUN_OBS']
    
    observer_0 = frames.HeliographicStonyhurst(0.*u.deg, sdo_lat_0*u.deg, radius=sdo_dist_0*u.m, obstime=obs_time_0)
    
    nx = int(round(metadata['NAXIS1']*downsample_coef))
    ny = int(round(metadata['NAXIS2']*downsample_coef))
    
    xs1 = (np.arange(1, nx + 1)).astype(float)
    ys1 = (np.arange(1, ny + 1)).astype(float)
    xys = np.transpose([np.tile(xs1, ny), np.repeat(ys1, nx)])
    
    data = hdul[1].data
    max_val = np.nanmax(data)
    fltr = np.isnan(data)
    data[fltr] = max_val

    data = misc.sample_image(data, downsample_coef)
    fltr = data > .9*max_val
    data[fltr] = np.nan
    test_plot = plot.plot(nrows=1, ncols=1, size=plot.default_size(1000, 1000))
    test_plot.colormap(data, show_colorbar=True)
    test_plot.save(f"image.png")
    test_plot.close()
    
    fltr0 = fltr
    xys = xys[np.logical_not(fltr).flatten()]
    
    for i in np.arange(1, len(hdul), 10):
        metadata = hdul[i].header
    
        a = metadata['CROTA2']*np.pi/180
            
        dx = metadata['CRVAL1']
        dy = metadata['CRVAL2']
        arcsecs_per_pix_x = metadata['CDELT1']/downsample_coef
        arcsecs_per_pix_y = metadata['CDELT2']/downsample_coef
        coef_x = 1./arcsecs_per_pix_x
        coef_y = 1./arcsecs_per_pix_y
        xc = metadata['CRPIX1']*downsample_coef
        yc = metadata['CRPIX2']*downsample_coef
        
        t_rec = metadata['T_REC']
        
        date = t_rec[:10]
        date = date[:4] + "-" + date[5:7] + "-" +date[8:10]
        
        hrs = t_rec[11:13]
        mins = t_rec[14:16]
        secs = t_rec[17:19]
        
        obs_time = f"{date} {hrs}:{mins}:{secs}"    
               
        sdo_lon = metadata['CRLN_OBS']
        sdo_lat = metadata['CRLT_OBS']
        sdo_dist = metadata['DSUN_OBS']
        
        
        if i == 1:
            sin_a = np.sin(a)
            cos_a = np.cos(a)
                
            print(obs_time)
            xs_arcsec, ys_arcsec = track.pix_to_image(xs1, ys1, dx, dy, xc, yc, cos_a, sin_a, arcsecs_per_pix_x, arcsecs_per_pix_y)
            grid = np.transpose([np.tile(xs_arcsec, ny), np.repeat(ys_arcsec, nx)])
            grid = grid[np.logical_not(fltr0).flatten()]
            
            xs_all_last = grid[:, 0]*u.arcsec
            ys_all_last = grid[:, 1]*u.arcsec
            
            
            
        
        observer_i = frames.HeliographicStonyhurst(0.*u.deg, sdo_lat*u.deg, radius=sdo_dist*u.m, obstime=obs_time)
        
        pix_dict = dict()
        chunk_size = int(len(xs_all_last)/num_chunks)
        start_index = 0
        
        x_pix_head, x_pix_tail = [], []
        y_pix_head, y_pix_tail = [], []
        xs_head, xs_tail = [], []
        ys_head, ys_tail = [], []
        lons_head, lons_tail = [], []
        lats_head, lats_tail = [], []

        for chunk_index in range(num_chunks):
            if chunk_index < num_chunks - 1:
                end_index = start_index + chunk_size
            else:
                end_index = len(xs_all_last)
            xs, ys = xs_all_last[start_index:end_index], ys_all_last[start_index:end_index]
        
            #c1 = SkyCoord(grid[:, 0]*u.arcsec, grid[:, 1]*u.arcsec, frame=frames.Helioprojective, observer=observer_0)
            c1 = SkyCoord(xs, ys, frame=frames.Helioprojective, observer=observer_0)
            c2 = c1.transform_to(frames.HeliographicCarrington)
            lons = c2.lon.value - sdo_lon_0
            lats = c2.lat.value
            c3 = SkyCoord(c2.lon, c2.lat, frame=frames.HeliographicCarrington, observer=observer_i, obstime=obs_time)
            c4 = c3.transform_to(frames.Helioprojective)
        
            x_pix, y_pix = track.image_to_pix(c4.Tx.value, c4.Ty.value, dx, dy, xc, yc, cos_a, sin_a, coef_x, coef_y)
            x_pix = np.round(x_pix)
            y_pix = np.round(y_pix)
    
            observer_0 = observer_i
            #sdo_lon_0 = sdo_lon
            xs = c4.Tx
            ys = c4.Ty
            
            x_pix = x_pix.tolist()
            y_pix = y_pix.tolist()
            xs = xs.value.tolist()
            ys = ys.value.tolist()
            lons = lons.tolist()
            lats = lats.tolist()

            split_point, _ = track.fix_sampling(x_pix, y_pix, xs, ys, lons, lats, xys, sdo_lon_0, observer_i, pix_dict, start_index,
                                                                  (dx, dy, xc, yc, cos_a, sin_a, arcsecs_per_pix_x, arcsecs_per_pix_y))

            start_index += chunk_size
            
            x_pix_head.extend(x_pix[:split_point])
            x_pix_tail.extend(x_pix[split_point:])
            x_pix.clear()
            y_pix_head.extend(y_pix[:split_point])
            y_pix_tail.extend(y_pix[split_point:])
            y_pix.clear()
            xs_head.extend(xs[:split_point])
            xs_tail.extend(xs[split_point:])
            xs.clear()
            ys_head.extend(ys[:split_point])
            ys_tail.extend(ys[split_point:])
            ys.clear()
            lons_head.extend(lons[:split_point])
            lons_tail.extend(lons[split_point:])
            lons.clear()
            lats_head.extend(lats[:split_point])
            lats_tail.extend(lats[split_point:])
            lats.clear()
        
        
        x_pix = x_pix_head + x_pix_tail
        x_pix_head.clear()
        x_pix_tail.clear()

        y_pix = y_pix_head + y_pix_tail
        y_pix_head.clear()
        y_pix_tail.clear()

        lons = lons_head + lons_tail
        lons_head.clear()
        lons_tail.clear()
        
        lats = lats_head + lats_tail
        lats_head.clear()
        lats_tail.clear()
        
        data = np.empty((ny, nx))
        data[:, :] = np.nan
        
        ###########################################################################
        if (i - 1) % 10 == 0:
            j = 0
            num_patches = 5
            patch_size = 20
            for lon in np.linspace(-80, 65, num_patches):
                k = 0
                lon_filter = (lons >= lon) * (lons < lon + patch_size)
                for lat in np.linspace(-80, 65, num_patches):
                    lat_filter = (lats >= lat) * (lats < lat + patch_size)
                    fltr = lon_filter * lat_filter
                    value = j * num_patches + k
                    #value = [1., -1.][(j % 2 + k % 2) % 2]
                    data[y_pix[fltr].astype(int), x_pix[fltr].astype(int)] = value
                    k += 1
                j += 1
                
            test_plot = plot.plot(nrows=1, ncols=1, size=plot.default_size(1000, 1000))
            test_plot.colormap(data, show_colorbar=True)
            test_plot.save(f"output{date}_{i}.png")
            test_plot.close()
            
            
            metadata['NAXIS1'] = nx
            metadata['NAXIS2'] = ny
        
            metadata['CDELT1'] = arcsecs_per_pix_x
            metadata['CDELT2'] = arcsecs_per_pix_y
            metadata['CRPIX1'] = xc
            metadata['CRPIX2'] = yc
            
            hdu = fits.ImageHDU(data=data, header=metadata)
        
            output.append(hdu)
            output.flush()
    
    output.close()
    hdul.close()    