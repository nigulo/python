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
import misc

import astropy.units as u
from astropy.coordinates import SkyCoord
import sunpy.coordinates
from sunpy.coordinates import frames
import track

hdul = fits.open("2013-02-03_hmi.M_720s.fits")

output = fits.open(f"test_data.fits", mode="append")

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

for i in np.arange(1, len(hdul), 10):
    metadata = hdul[i].header
    data = hdul[i].data

    downsample_coef = .1

    nx = int(round(metadata['NAXIS1']*downsample_coef))
    ny = int(round(metadata['NAXIS2']*downsample_coef))
    
    xs = (np.arange(1, nx + 1)).astype(float)
    ys = (np.arange(1, ny + 1)).astype(float)
    
    a = metadata['CROTA2']*np.pi/180
        
    dx = metadata['CRVAL1']
    dy = metadata['CRVAL2']
    arcsecs_per_pix_x = metadata['CDELT1']/downsample_coef
    arcsecs_per_pix_y = metadata['CDELT2']/downsample_coef
    coef_x = 1./arcsecs_per_pix_x
    coef_y = 1./arcsecs_per_pix_y
    xc = int(round(metadata['CRPIX1']*downsample_coef))
    yc = int(round(metadata['CRPIX2']*downsample_coef))
    
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
    
    
    sin_a = np.sin(a)
    cos_a = np.cos(a)
        
    print(obs_time)
    xs_arcsec, ys_arcsec = track.pix_to_image(xs, ys, dx, dy, xc, yc, cos_a, sin_a, arcsecs_per_pix_x, arcsecs_per_pix_y)
    grid = np.transpose([np.tile(xs_arcsec, ny), np.repeat(ys_arcsec, nx)])

    
    observer_i = frames.HeliographicStonyhurst(0.*u.deg, sdo_lat*u.deg, radius=sdo_dist*u.m, obstime=obs_time)
    
    c1 = SkyCoord(grid[:, 0]*u.arcsec, grid[:, 1]*u.arcsec, frame=frames.Helioprojective, observer=observer_0)
    c2 = c1.transform_to(frames.HeliographicCarrington)
    lons = c2.lon.value - sdo_lon_0
    lats = c2.lat.value
    c3 = SkyCoord(c2.lon, c2.lat, frame=frames.HeliographicCarrington, observer=observer_i, obstime=obs_time)
    c4 = c3.transform_to(frames.Helioprojective)

    x_pix, y_pix = track.image_to_pix(c4.Tx.value, c4.Ty.value, dx, dy, xc, yc, cos_a, sin_a, coef_x, coef_y)
    x_pix = np.round(x_pix)
    y_pix = np.round(y_pix)

    data = np.zeros((ny, nx))

    ###########################################################################

    lon_filter = (lons > -70) * (lons < -50)
    lat_filter = (lats > -70) * (lats < -50)
    fltr = lon_filter * lat_filter
    data[y_pix[fltr].astype(int), x_pix[fltr].astype(int)] = 1.
    
    ###########################################################################

    lon_filter = (lons < 70) * (lons > 50)
    lat_filter = (lats > -70) * (lats < -50)
    fltr = lon_filter * lat_filter
    data[y_pix[fltr].astype(int), x_pix[fltr].astype(int)] = 1.

    ###########################################################################

    lon_filter = (lons > -70) * (lons < -50)
    lat_filter = (lats < 70) * (lats > 50)
    fltr = lon_filter * lat_filter
    data[y_pix[fltr].astype(int), x_pix[fltr].astype(int)] = 1.

    ###########################################################################

    lon_filter = (lons < 70) * (lons > 50)
    lat_filter = (lats < 70) * (lats > 50)
    fltr = lon_filter * lat_filter
    data[y_pix[fltr].astype(int), x_pix[fltr].astype(int)] = 1.

    ###########################################################################

    lon_filter = (lons > -30) * (lons < -10)
    lat_filter = (lats > -30) * (lats < -10)
    fltr = lon_filter * lat_filter
    data[y_pix[fltr].astype(int), x_pix[fltr].astype(int)] = 1.
    
    ###########################################################################

    lon_filter = (lons < 30) * (lons > 10)
    lat_filter = (lats > -30) * (lats < -10)
    fltr = lon_filter * lat_filter
    data[y_pix[fltr].astype(int), x_pix[fltr].astype(int)] = 1.

    ###########################################################################

    lon_filter = (lons > -30) * (lons < -10)
    lat_filter = (lats < 30) * (lats > 10)
    fltr = lon_filter * lat_filter
    data[y_pix[fltr].astype(int), x_pix[fltr].astype(int)] = 1.

    ###########################################################################

    lon_filter = (lons < 30) * (lons > 10)
    lat_filter = (lats < 30) * (lats > 10)
    fltr = lon_filter * lat_filter
    data[y_pix[fltr].astype(int), x_pix[fltr].astype(int)] = 1.

    ###########################################################################

    lon_filter = (lons > -70) * (lons < -50)
    lat_filter = (lats < 5) * (lats > -5)
    fltr = lon_filter * lat_filter
    data[y_pix[fltr].astype(int), x_pix[fltr].astype(int)] = 1.

    ###########################################################################

    lon_filter = (lons < 70) * (lons > 50)
    lat_filter = (lats < 5) * (lats > -5)
    fltr = lon_filter * lat_filter
    data[y_pix[fltr].astype(int), x_pix[fltr].astype(int)] = 1.

    ###########################################################################

    lon_filter = (lons < 5) * (lons > -5)
    lat_filter = (lats < 5) * (lats > -5)
    fltr = lon_filter * lat_filter
    data[y_pix[fltr].astype(int), x_pix[fltr].astype(int)] = 1.

    ###########################################################################
        
    test_plot = plot.plot(nrows=1, ncols=1, size=plot.default_size(1000, 1000))
    test_plot.colormap(data, show_colorbar=True)
    test_plot.save(f"output{i}.png")
    test_plot.close()
    
    hdu = fits.ImageHDU(data=data, header=metadata)

    output.append(hdu)
    output.flush()

output.close()
hdul.close()    