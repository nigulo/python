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
from track import parse_t_rec, pix_to_image, image_to_pix


if (__name__ == '__main__'):
       
    input_file1 = sys.argv[1]
    input_file2 = sys.argv[2]
    hr = sys.argv[3]
    lon = float(sys.argv[4])
    lat = float(sys.argv[5])
    
    hdul = fits.open(input_file1)
    data = fits.getdata(input_file1, 1)
    for i in np.arange(1, len(hdul)):
        t_rec = hdul[i].header['T_REC']
        year, month, day, hrs, mins, secs = parse_t_rec(t_rec)
        if hrs == hr:
            break
    assert(i < len(hdul))   
    date = year + "-" + month + "-" + day
    metadata = hdul[i].header

    nx = metadata['NAXIS1']
    ny = metadata['NAXIS2']
    
    xs = np.arange(1, nx + 1)
    ys = np.arange(1, ny + 1)

    date = year + "-" + month + "-" + day
    
    obs_time_str = f"{date} {hrs}:{mins}:{secs}"
    
    sdo_lon = metadata['CRLN_OBS']
    sdo_lat = metadata['CRLT_OBS']
    sdo_dist = metadata['DSUN_OBS']
    
    observer = frames.HeliographicStonyhurst(0.*u.deg, sdo_lat*u.deg, radius=sdo_dist*u.m, obstime=obs_time_str)
    
    a = metadata['CROTA2']*np.pi/180
    
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
    

    xs_arcsec, ys_arcsec = pix_to_image(xs, ys, dx, dy, xc, yc, cos_a, sin_a, arcsecs_per_pix_x, arcsecs_per_pix_y)
    grid = np.transpose([np.tile(xs_arcsec, ny), np.repeat(ys_arcsec, nx)])
    
    fltr = np.logical_not(np.isnan(data)).flatten()
    grid = grid[fltr]
    
    xs_arcsec = grid[:, 0]*u.arcsec
    ys_arcsec = grid[:, 1]*u.arcsec
        
    observer_i = frames.HeliographicStonyhurst(0.*u.deg, sdo_lat*u.deg, radius=sdo_dist*u.m, obstime=obs_time_str)
    
    c1 = SkyCoord(xs_arcsec, ys_arcsec, frame=frames.Helioprojective, observer=observer)
    c2 = c1.transform_to(frames.HeliographicCarrington)
    lons = c2.lon.value - sdo_lon
    lats = c2.lat.value

    c3 = SkyCoord(c2.lon, c2.lat, frame=frames.HeliographicCarrington, observer=observer, obstime=obs_time_str)
    c4 = c3.transform_to(frames.Helioprojective)
    
    xs_arcsec = c4.Tx
    ys_arcsec = c4.Ty

    x_pix, y_pix = image_to_pix(xs_arcsec.value, ys_arcsec.value, dx, dy, xc, yc, cos_a, sin_a, coef_x, coef_y)
    x_pix = np.round(x_pix)
    y_pix = np.round(y_pix)

    data1 = []

    for l in range(len(x_pix)):
        if lons[l] >= lon and lons[l] < lon+15 and lats[l] >= lat and lats[l] < lat+15:
            if not np.isnan(y_pix[l]) and not np.isnan(x_pix[l]):
                data1.append([lons[l], lats[l], data[int(y_pix[l]), int(x_pix[l])]])
    #data1.sort()
    data1 = np.asarray(data1)
    min_lon = np.min(data1[:, 0])
    max_lon = np.max(data1[:, 0])
    min_lat = np.min(data1[:, 1])
    max_lat = np.max(data1[:, 1])

    resolution = 300
    lons = np.linspace(min_lon, max_lon, resolution)
    lats = np.linspace(min_lat, max_lat, resolution)
    
    data_for_plot = np.zeros((resolution, resolution))
    for i in range(len(lons) - 1):
        fltr = (data1[:, 0] >= lons[i]) * (data1[:, 0] < lons[i+1])
        data2 = data1[fltr]
        for j in range(len(lats) - 1):
            fltr2 = (data2[:, 1] >= lats[j]) * (data2[:, 1] < lats[j+1])
            data3 = data2[fltr2]
            if len(data3) > 0:
                print(len(data3))
                data_for_plot[i, j] = data3[0, 2]
    test_plot = plot.plot(nrows=1, ncols=1, size=plot.default_size(data_for_plot.data.shape[1], data_for_plot.data.shape[0]))
    vmax = max(abs(np.min(data_for_plot)), abs(np.max(data_for_plot)))
    test_plot.colormap(data_for_plot, cmap="bwr", show_colorbar=True, vmin=-vmax, vmax=vmax)
    test_plot.save("patch1.png")
    test_plot.close()
    data_for_plot = fits.getdata(input_file2, 1)[0]
    test_plot = plot.plot(nrows=1, ncols=1, size=plot.default_size(data_for_plot.data.shape[1], data_for_plot.data.shape[0]))
    vmax = max(abs(np.min(data_for_plot)), abs(np.max(data_for_plot)))
    #test_plot.colormap(data_for_plot.T, 1, cmap="bwr", show_colorbar=True, vmin=-vmax, vmax=vmax)
    test_plot.colormap(data_for_plot.T, cmap="gnuplot", show_colorbar=True)
    test_plot.save("patch2.png")
    test_plot.close()
    
