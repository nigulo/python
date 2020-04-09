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
out_dir = "data_out"
#nx = 96
num_modes = 44
use_zarr = False

if use_zarr:
    import zarr

dir_name = "mihi/data"
if len(sys.argv) > 1:
    dir_name = sys.argv[1]

n_frames = 100
if len(sys.argv) > 2:
    n_frames = int(sys.argv[2])

n_objects = 200

shuffle = True
if len(sys.argv) > 3:
    if sys.argv[3].upper() == "TEST":
        print("Generating test data")
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

        for i in tqdm(range(lower, upper)):
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
        Ds = handler.create_dataset('Ds', shape=(num_objects, num_frames, 2, nx, nx), compressor=None)
        objs = handler.create_dataset('objs', shape=(num_objects, nx, nx), compressor=None)
        momfbd_coefs = handler.create_dataset('alphas', shape=(num_objects, num_frames, num_modes), compressor=None)
        positions = handler.create_dataset('positions', shape=(num_objects, 2), dtype='int', compressor=None)
        coords = handler.create_dataset('coords', shape=(num_objects, 2), dtype='int', compressor=None)
    else:
        Ds = np.zeros((num_objects, num_frames, 2, nx, nx)) # in real space
        objs = np.zeros((num_objects, nx, nx)) # in real space
        momfbd_coefs = np.zeros((num_objects, num_frames, num_modes))
        positions = np.zeros((num_objects, 2), dtype='int')
        coords = np.zeros((num_objects, 2), dtype='int')
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
        
    
    for loop in tqdm(range(num_objects)):

        x0 = xl[indx[loop]]
        y0 = yl[indy[loop]]
        
        dx = tmp['dy'][indx[loop], indy[loop], 1]
        dy = tmp['dx'][indx[loop], indy[loop], 1]

        if (indx[loop] == 0 or indx[loop] == npx-1 or indy[loop] == 0 or indy[loop] == npy-1):
            dx = 0
            dy = 0

        if len(objs_momfbd) > indt[loop]:
            objs[loop] = objs_momfbd[indt[loop]][indx[loop], indy[loop]]
        for i in range(num_frames):
            Ds[loop, i, 0] = images[indt[loop]+i][x0:x0+nx,y0:y0+nx]
            Ds[loop, i, 1] = images_defocus[indt[loop]+i][x0+dx:x0+nx+dx,y0+dy:y0+nx+dy]
            momfbd_coefs[loop, i] = alphas[indt[loop]+i][indx[loop],indy[loop],:]
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
        diversity = misc.sample_image(diversity, 2)
        #diversity2 = np.zeros((diversity.shape[0], diversity.shape[1], diversity.shape[2], nx, nx))
        #for i in np.arange(diversity.shape[2]):
        #    diversity2[:, :, i] = np.pad(diversity[:, :, i, :, :], ((0, 0), (0, 0), (pad_nx, pad_nx), (pad_nx, pad_nx)), mode='constant', constant_values=0.0)
        #diversity = diversity2
    if pupil.shape[-1] != nx:
        assert(pupil.shape[-1] == nx // 2)
        pupil = misc.sample_image(pupil, 2)
        #pupil = np.pad(pupil, ((pad_nx, pad_nx), (pad_nx, pad_nx)), mode='constant', constant_values=0.0)
    if modes.shape[-1] != nx:
        assert(modes.shape[-1] == nx // 2)
        modes = misc.sample_image(modes, 2)
        #modes = np.pad(modes, ((0, 0), (pad_nx, pad_nx), (pad_nx, pad_nx)), mode='constant', constant_values=0.0)

    mode_scale = np.array([3.4211644e-07, 2.9869247e-07, 1.2330536e-07, 1.2948983e-07,\
        1.3634113e-07, 6.2635998e-08, 6.0679490e-08, 6.1960371e-08,\
        6.2712253e-08, 4.1066169e-08, 4.8136709e-08, 4.3251813e-08,\
        4.8604090e-08, 4.6081762e-08, 3.6688341e-08, 4.7021366e-08,\
        4.4507608e-08, 4.7089408e-08, 3.8737561e-08, 3.7861817e-08,\
        5.0583559e-08, 5.0688101e-08, 4.7258556e-08, 4.9367131e-08,\
        4.6206999e-08, 3.9753179e-08, 3.9710063e-08, 4.7511332e-08,\
        3.4647051e-08, 4.4375597e-08, 3.8252473e-08, 3.7187508e-08,\
        3.6801211e-08, 3.1744438e-08, 3.1704403e-08, 4.5903555e-08,\
        2.5063319e-08, 2.7119935e-08, 2.6932595e-08, 2.9540985e-08,\
        2.2285006e-08, 2.0293584e-08, 2.1038879e-08, 1.8963931e-08])

    modes *= mode_scale[:, None, None]

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
