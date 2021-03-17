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
num_chunks = 1

for file_date in ["2013-02-03", "2013-02-04"]:

    hdul = fits.open(f"{file_date}_hmi.M_720s.fits")
    
    try:
        os.remove(f"{file_date}_2.fits")
    except:
        pass
    
    output = fits.open(f"{file_date}_2.fits", mode="append")
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

        ###########################################################################
        if (i - 1) % 10 == 0:
            data = hdul[i].data
            max_val = np.nanmax(data)
            fltr = np.isnan(data)
            data[fltr] = max_val
        
            data = misc.sample_image(data, downsample_coef)
            fltr = data > .9*max_val
            data[fltr] = np.nan
            test_plot = plot.plot(nrows=1, ncols=1, size=plot.default_size(1000, 1000))
            test_plot.colormap(data, show_colorbar=True)
            test_plot.save(f"image{date}_{i}.png")
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