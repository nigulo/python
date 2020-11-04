import numpy as np
from pyana import pyana as pa
#import pyana as pa
import glob
import scipy.io as io
from tqdm import tqdm
import os
import sys
sys.path.append('../utils')
import misc
import utils
from numcodecs import Blosc
compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
out_dir = "data_out"
#nx = 96
num_modes = 44
use_zarr = True

if use_zarr:
    import zarr

dir_name = "mihi/data"
if len(sys.argv) > 1:
    dir_name = sys.argv[1]

n_frames = 100
if len(sys.argv) > 2:
    n_frames = int(sys.argv[2])

n_objects = 200

compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

train = True
shuffle = True
if len(sys.argv) > 3:
    if sys.argv[3].upper() == "TEST":
        print("Generating test data")
        train = False
        shuffle = False
        n_objects = None

if len(sys.argv) > 4:
    n_objects = int(sys.argv[4])

def generate_set(path, files, num_objects=None, num_frames=100, shuffle=True):
    
    # Count the number of available images for which we have all information
    n_full_images = 0
    present_images = []
    images = []
    images_defocus = []
    alphas = []
    objs_momfbd = []

    # Go through all input momfbd.sav files and check which files are available
    for f in files:
        tmp = f.split('.')
        lower = int(tmp[-6])
        upper = int(tmp[-4])
        tmp = io.readsav(f)

        objs_momfbd.append(tmp['img']) # MOMBD Restored objects
        loop = 0

        for i in tqdm(range(lower, upper+1)):
            si = str(i)
            si = "0"*(9-len(si))+si
            if (os.path.isfile(f'{path}image.{si}.f0.ch1.cor.f0')):
                present_images.append(i)

                f = pa.fzread(f'{path}image.{si}.f0.ch1.cor.f0')
                images.append(f['data'])
                if loop >= tmp['alpha'].shape[2]:
                    # This is a HACK for the case when we have full restoration only for single image
                    alphas.append(tmp['alpha'][:,:,0,:])
                else:
                    alphas.append(tmp['alpha'][:,:,loop,:])

                f = pa.fzread(f'{path}image.{si}.f0.ch2.cor.f0')
                images_defocus.append(f['data'])
                n_full_images += 1

            loop += 1    

    num_frames = min(num_frames, n_full_images)

    print(f"Number of full images: {n_full_images}")
    print(f"Number of frames: {num_frames}")
    nx_full = images[0].shape[1]
    ny_full = images[0].shape[0]
    print(f"Full image dimensions: {nx_full}, {ny_full}")

    nx = objs_momfbd[0][0, 0].shape[0]
    assert(nx == objs_momfbd[0][0, 0].shape[1])
    print(f"Batch size: {nx}")
    
    # Get x and y positions of each one of the MOMFBD patches
    xl = tmp['yl'][:,0] - 1
    yl = tmp['xl'][0,:] - 1

    # Num patches
    npx = len(xl)
    npy = len(yl)

    print(f"Number of patches: {npx}, {npy}")

    #overlap_x = ((npx*nx)-nx_full)/npx/2
    #overlap_y = ((npy*nx)-ny_full)/npy/2
    #print(xl[1:]-xl[:-1])
    #print(yl[1:]-yl[:-1])
    #print((npx*nx), (npy*nx), overlap_x, overlap_y)
    

    #assert(overlap_x == overlap_y)
    #assert(overlap_x == int(overlap_x))
    #overlap = int(overlap_x)

    num_modes = len(alphas[0][0, 0])
    print(f"Number of modes: {num_modes}")

    if num_objects is None:
        num_objects = npx*npy
    if not shuffle:
        assert(num_objects <= npx*npy)
        npx *= num_objects/npx/npy
        npy *= num_objects/npx/npy
        npx = int(npx)
        npy = int(npy)
        num_objects = npx*npy


    if use_zarr:
        handler = zarr.open(out_dir + '/Ds.zarr', 'w')
        Ds = handler.create_dataset('Ds', shape=(num_objects, num_frames, 2, nx, nx), chunks=(max(num_objects//100, 1), num_frames, 2, nx, nx), compressor=None, dtype=np.int16)
        if train:
            objs = None
        else:
            objs = handler.create_dataset('objs', shape=(num_objects, nx, nx), chunks=(max(num_objects//100, 1), nx, nx), compressor=None, dtype=np.int16)
        momfbd_coefs = handler.create_dataset('alphas', shape=(num_objects, num_frames, num_modes), chunks=(max(num_objects//100, 1), num_frames, num_modes), compressor=None, dtype=np.float32)
        positions = handler.create_dataset('positions', shape=(num_objects, 2), dtype=np.int16, compressor=None)
        coords = handler.create_dataset('coords', shape=(num_objects, 2), dtype=np.int16, compressor=None)
    else:
        Ds = np.zeros((num_objects, num_frames, 2, nx, nx), dtype=np.int16)
        if train:
            objs = None
        else:
            objs = np.zeros((num_objects, nx, nx), dtype=np.int16)
        momfbd_coefs = np.zeros((num_objects, num_frames, num_modes), dtype=np.float32)
        positions = np.zeros((num_objects, 2), dtype=np.int16)
        coords = np.zeros((num_objects, 2), dtype=np.int16)
        handler = None
        

    # Randomly extract patches and times for selecting the bursts
    # This way of extracting the patches is slightly limited because I only consider
    # the MOMFBD patches. One can always extract patches of size 96x96 randomly
    # on the field of view. I did this when I was comparing with the wavefront 
    # obtained with MOMFBD but since now we are training self-supervisedly, this is not a limitation anymore
    # Anyway, one should recompute the diversity in this case
    if shuffle:
        indt = np.random.randint(low=0, high=n_full_images-num_frames, size=num_objects)
        indx = np.random.randint(low=0, high=npx, size=num_objects)
        indy = np.random.randint(low=0, high=npy, size=num_objects)
    else:
        #assert(num_objects <= npx*npy)
        num_objects = npx*npy
        indt = np.zeros(num_objects, dtype="int")
        indx = np.repeat(np.arange(0, npx, dtype="int"), npy)
        indy = np.tile(np.arange(0, npy, dtype="int"), npx)
        
    images = np.asarray(images)
    images_defocus = np.asarray(images_defocus)
    alphas = np.asarray(alphas)
    for loop in tqdm(range(num_objects)):

        x0 = xl[indx[loop]]
        y0 = yl[indy[loop]]
        
        dx = tmp['dy'][indx[loop], indy[loop], 1]
        dy = tmp['dx'][indx[loop], indy[loop], 1]
        
        #print("dx, dy", dx, dy, indx[loop], indy[loop])
        #if (indx[loop] == 0 or indx[loop] == npx-1 or indy[loop] == 0 or indy[loop] == npy-1):
        #    dx = 0
        #    dy = 0
        
        #    print("Setting zero")
        if objs is not None:
            if len(objs_momfbd) > indt[loop]:
                objs[loop] = objs_momfbd[indt[loop]][indx[loop], indy[loop]]
        #for i in range(num_frames):
        #    Ds[loop, i, 0] = images[indt[loop]+i][x0:x0+nx,y0:y0+nx]
        #    Ds[loop, i, 1] = images_defocus[indt[loop]+i][x0+dx:x0+nx+dx,y0+dy:y0+nx+dy]
        #    momfbd_coefs[loop, i] = alphas[indt[loop]+i][indx[loop],indy[loop],:]
        start_index = indt[loop]
        end_index = start_index + num_frames
        Ds[loop, :num_frames, 0] = images[start_index:end_index,x0:x0+nx,y0:y0+nx]
        defocus_image = np.array(images_defocus[start_index:end_index,:,:])
        x_left = x0+dx
        x_right = x0+nx+dx
        y_bottom = y0+dy
        y_top = y0+nx+dy
        pad_left = 0
        pad_right = 0
        pad_bottom = 0
        pad_top = 0
        if x_left < 0:
            pad_left = abs(x_left)
            x_left = 0
            x_right += pad_left
        if x_right > defocus_image.shape[1]:
            pad_right = x_right - defocus_image.shape[1]
        if y_bottom < 0:
            pad_bottom = abs(y_bottom)
            y_bottom = 0
            y_top += pad_bottom
        if y_top > defocus_image.shape[2]:
            pad_top = y_top - defocus_image.shape[2]
        print(x_left, x_right, y_bottom, y_top)
        print(defocus_image.shape)
        defocus_image = np.pad(defocus_image, ((0, 0), (pad_left, pad_right), (pad_bottom, pad_top)), mode='constant')
        print(defocus_image.shape)
        Ds[loop, :num_frames, 1] = defocus_image[:,x_left:x_right,y_bottom:y_top]
        momfbd_coefs[loop, :num_frames] = alphas[start_index:end_index,indx[loop],indy[loop],:]
        positions[loop, 0] = indx[loop]
        positions[loop, 1] = indy[loop]
        
        coords[loop, 0] = x0
        coords[loop, 1] = y0
        
    return Ds, objs, momfbd_coefs, positions, coords, handler


if __name__ == '__main__':

    if dir_name[-1] != "/":
        dir_name += "/"
    
    files = glob.glob(f'{dir_name}*.sav')
    files.sort()

    # Training set
    Ds, objs, momfbd_coefs, positions, coords, handler = generate_set(dir_name, files, num_objects=n_objects, num_frames=n_frames, shuffle=shuffle)

    nx = Ds.shape[3]
    tmp = io.readsav(files[0])
    diversity = np.array(tmp['div'])
    modes = np.array(tmp['mode'])
    pupil = np.array(tmp['pupil'])

    if diversity.shape[-1] != nx:
        assert(diversity.shape[-1] == nx // 2)
        pad_nx = nx // 4
        #diversity = misc.sample_image(diversity, 2)
        diversity2 = np.zeros((diversity.shape[0], diversity.shape[1], diversity.shape[2], nx, nx))
        for i in np.arange(diversity.shape[2]):
            diversity2[:, :, i] = np.pad(diversity[:, :, i, :, :], ((0, 0), (0, 0), (pad_nx, pad_nx), (pad_nx, pad_nx)), mode='constant', constant_values=0.0)
        diversity = diversity2
    if pupil.shape[-1] != nx:
        assert(pupil.shape[-1] == nx // 2)
        #pupil = misc.sample_image(pupil, 2)
        pupil = np.pad(pupil, ((pad_nx, pad_nx), (pad_nx, pad_nx)), mode='constant', constant_values=0.0)
    if modes.shape[-1] != nx:
        assert(modes.shape[-1] == nx // 2)
        #modes = misc.sample_image(modes, 2)
        modes = np.pad(modes, ((0, 0), (pad_nx, pad_nx), (pad_nx, pad_nx)), mode='constant', constant_values=0.0)

    modes *= utils.mode_scale[:, None, None]

    if not use_zarr:
        np.savez_compressed(out_dir + '/Ds', Ds=Ds, objs=objs, pupil=pupil, modes=modes, diversity=diversity, 
                        alphas=momfbd_coefs, positions=positions, coords=coords)
    else:
        pupil_zarr = handler.create_dataset('pupil', shape=pupil.shape, compressor=None)
        modes_zarr = handler.create_dataset('modes', shape=modes.shape, compressor=None)
        diversity_zarr = handler.create_dataset('diversity', shape=diversity.shape, compressor=None)
        pupil_zarr[:] = pupil
        modes_zarr[:] = modes
        diversity_zarr[:] = diversity
