import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
#import numpy.random as random
sys.setrecursionlimit(10000)
import torch
import torch.utils.data

import zarr
#import pickle
import os.path

import time
from NN import NN


i = 1
dir_name = None
if len(sys.argv) > i:
    dir_name = sys.argv[i]
i += 1

if len(sys.argv) > i:
    nn_mode = int(sys.argv[i])
i += 1

train = True
if len(sys.argv) > i:
    if sys.argv[i].upper() == "TEST":
        train = False
i += 1

if train:
    data_files = "Ds"
else:
    data_files = "Ds_test"

if len(sys.argv) > i:
    data_files = sys.argv[i]

data_files = data_files.split(',')
assert(len(data_files) > 0)


i += 1

n_test_frames = None
if len(sys.argv) > i:
    n_test_frames = int(sys.argv[i])
i += 1

n_test_objects = None
if len(sys.argv) > i:
    n_test_objects = int(sys.argv[i])

benchmarking_level = 0
i += 1
if len(sys.argv) > i:
    benchmarking_level = int(sys.argv[i])

start_index=0
i += 1
if len(sys.argv) > i:
    start_index = int(sys.argv[i])

state_file = "state.tar"
i += 1
if len(sys.argv) > i:
    state_file = sys.argv[i]

num_reps = 1000
train_perc = 0.99

if dir_name is None:
    dir_name = "results" + time.strftime("%Y%m%d-%H%M%S")
    os.mkdir(dir_name)
    
cuda = torch.cuda.is_available()
n_gpus_available = torch.cuda.device_count()

device = torch.device("cuda:0" if cuda else "cpu")

if train:
    sys.stdout = open(dir_name + '/log.txt', 'a')
else:
    device = torch.device("cuda:1" if cuda else "cpu")

print(f"Device : {device}")
    
n_gpus = 1#len(gpus)

if n_gpus >= 1:
    from numba import cuda

def load_data(data_file):
    is_zarr = False
    f = dir_name + '/' + data_file + ".zarr"
    if os.path.exists(f):
        is_zarr = True
        loaded = zarr.open(f, 'r')
    else:
        f = dir_name + '/' + data_file + ".npz"
        if os.path.exists(f):
            loaded = np.load(f, mmap_mode='r')
        else:
            raise Exception("No data found")
    Ds = loaded['Ds'][:]
    try:
        objs = loaded['objs'][:]
    except:
        objs = None
    pupil = loaded['pupil'][:]
    modes = loaded['modes'][:]
    diversity = loaded['diversity'][:]
    try:
        coefs = loaded['alphas'][:]
    except:
        coefs = None
    try:
        positions = loaded['positions'][:]
    except:
        positions = None
    try:
        coords = loaded['coords'][:]
    except:
        coords = None
    try:
        neighbours = loaded['neighbours'][:]
    except:
        neighbours = None
    return Ds, objs, pupil, modes, diversity, coefs, positions, coords, neighbours




        

if train:

    datasets = []
    
    Ds, objs, pupil, modes, diversity, true_coefs, positions, coords, neighbours = load_data(data_files[0])
    
    nx = Ds.shape[3]
    model = NN(nx, pupil, modes, device, dir_name, state_file)
    model.init()
    
    datasets.append((Ds, objs, diversity, positions, coords, neighbours))
    
    for data_file in data_files[1:]:
        Ds3, objs3, pupil3, modes3, diversity3, true_coefs3, positions3, coords3, neighbours3 = load_data(data_file)
        Ds3 = Ds3[:,:Ds.shape[1]]
        #Ds = np.concatenate((Ds, Ds3))
        #objs = np.concatenate((objs, objs3))
        #positions = np.concatenate((positions, positions3))
        datasets.append((Ds3, objs3, diversity3, positions3, coords3, neighbours3))

    num_modes = len(modes)
    num_modes_to_use = 4

    num_data = 0
    for d in datasets:
        num_data += len(d[0])
    
    try:
        Ds_test, objs_test, _, _, _, _, positions_test, _, neighbours_test = load_data(data_files[0]+"_valid")
        n_test = min(Ds_test.shape[0], 10)
        Ds_test = Ds_test[:n_test, :min(Ds_test.shape[1], model.num_frames)]
        objs_test = objs_test[:n_test]
        positions_test = positions_test[:n_test]
        neighbours_test = neighbours_test[:n_test]

        objs_train = objs
        positions_train = positions
        neighbours_train = neighbours

        n_train = num_data
        print("validation set: ", data_files[0]+"_valid")
        print("n_train, n_test", len(Ds), len(Ds_test))
    except:
        
        n_train = int(num_data*train_perc)
        n_test = num_data - n_train
        n_test = min(len(datasets[-1][0]), n_test)
        print("n_train, n_test", n_train, n_test)

        if n_test == len(datasets[-1][0]):
            Ds_test, objs_test, _, positions_test, _, neighbours_test = datasets.pop()
        else:
            Ds_last, objs_last, diversity_last, positions_last, coords_last, neighbours_last = datasets[-1]
            Ds_train = Ds_last[:-n_test]
            Ds_test = Ds_last[-n_test:]
            if objs_last is not None:
                objs_train = objs_last[:-n_test]
                objs_test = objs_last[-n_test:]
            else:
                objs_train = None
                objs_test = None
            if positions_last is not None:
                positions_train = positions_last[:-n_test]
                positions_test = positions_last[-n_test:]
            else:
                positions_train = None
                positions_test = None
            if coords_last is not None:
                coords_train = coords_last[:-n_test]
                coords_test = coords_last[-n_test:]
            else:
                coords_train = None
                coords_test = None
            if neighbours_last is not None:
                neighbours_train = neighbours_last[:-n_test]
                neighbours_test = neighbours_last[-n_test:]
            else:
                neighbours_train = None
                neighbours_test = None
            datasets[-1] = (Ds_train, objs_train, diversity_last, positions_train, coords_train, neighbours_train)
            
    print("num_frames", Ds.shape[1])

    coords_test = None
    
    #my_test_plot = plot.plot()
    #if len(diversity.shape) == 5:
    #    my_test_plot.colormap(diversity[0, 0, 1], show_colorbar=True)
    #elif len(diversity.shape) == 3:
    #    my_test_plot.colormap(diversity[1], show_colorbar=True)
    #else:
    #    my_test_plot.colormap(diversity, show_colorbar=True)
    #my_test_plot.save(dir_name + "/diversity.png")
    #my_test_plot.close()

    #my_test_plot = plot.plot()
    #my_test_plot.colormap(pupil, show_colorbar=True)
    #my_test_plot.save(dir_name + "/pupil_train.png")
    #my_test_plot.close()
    
    n_test_frames = Ds_test.shape[1] # Necessary to be set (TODO: refactor)

    model.set_data([(Ds_test, objs_test, diversity, positions_test, coords, neighbours_test)], train_data=False)
    model.set_data(datasets, train_data=True)
    for rep in np.arange(0, num_reps):
        
        print("Rep no: " + str(rep))
    
        #model.psf.set_jmax_used(num_modes_to_use)
        model.do_train()
        
        if num_modes_to_use <= num_modes//2:
            num_modes_to_use *= 2
        else:
            num_modes_to_use = num_modes
            

else:

    Ds, objs, pupil, modes, diversity, true_coefs, positions, coords, neighbours = load_data(data_files[0])

    nx = Ds.shape[3]
    num_modes = len(modes)

    if n_test_objects is None:
        n_test_objects = Ds.shape[0]
    if n_test_frames is None:
        n_test_frames = Ds.shape[1]
    n_test_objects = min(Ds.shape[0], n_test_objects)
    
    assert(n_test_frames <= Ds.shape[1])
    
    print("n_test_objects, n_test_frames", n_test_objects, n_test_frames)
    
    max_pos = np.max(positions, axis = 0)
    min_pos = np.array([start_index, start_index])
    print(max_pos)
    max_pos = np.floor(max_pos*np.sqrt(n_test_objects/len(Ds))).astype(int) + min_pos
    print(max_pos)
    filtr = np.all((positions <= max_pos) & (positions >= min_pos), axis=1)
    print("filtr", filtr)
    print("positions", positions.shape)

    if True:
        stride = 1
    else:
        #random_indices = random.choice(Ds.shape[1], size=Ds.shape[1], replace=False)
        #Ds = Ds[:, random_indices]
        stride = Ds.shape[1] // n_test_frames
    #n_test_frames //= stride
    Ds = Ds[filtr, :stride*n_test_frames:stride]
    objs = objs[filtr]
    positions = positions[filtr]
    coords = coords[filtr]
    if neighbours is not None:
        neighbours = neighbours[filtr]
    true_coefs = true_coefs[filtr, :stride*n_test_frames:stride]

    ###########################################################################
    ## Some plots for test purposes
    #my_test_plot = plot.plot()
    #my_test_plot.colormap(Ds[0, 0, 0], show_colorbar=True)
    #my_test_plot.save(dir_name + "/D0_test.png")
    #my_test_plot.close()
    
    #my_test_plot = plot.plot()
    #my_test_plot.colormap(Ds[0, 0, 1])
    #my_test_plot.save(dir_name + "/D0_d_test.png")
    #my_test_plot.close()

    #my_test_plot = plot.plot()
    #my_test_plot.colormap(pupil, show_colorbar=True)
    #my_test_plot.save(dir_name + "/pupil_test.png")
    #my_test_plot.close()
    ###########################################################################    

    model = NN(nx, pupil, modes, device, dir_name, state_file)
    model.init()

    model.do_test((Ds, objs, diversity, positions, coords, neighbours), file_prefix="test", num_test_frames=n_test_frames, 
                  true_coefs=true_coefs, benchmarking_level=benchmarking_level)

#logfile.close()
