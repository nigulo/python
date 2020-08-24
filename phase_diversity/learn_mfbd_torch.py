import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import numpy.random as random
sys.setrecursionlimit(10000)
import torch
import torch.nn as nn
import torch.utils.data

#tf.compat.v1.disable_eager_execution()
#from multiprocessing import Process

import math

import zarr
#import pickle
import os.path
import numpy.fft as fft
import matplotlib.pyplot as plt
import tqdm

import time
#import scipy.signal as signal

gamma = 1.0

MODE_1 = 1 # aberrated images --> wavefront coefs --> MFBD loss
MODE_2 = 2 # aberrated images --> wavefront coefs --> object (using MFBD formula) --> aberrated images
MODE_3 = 3 # aberrated images --> wavefront coefs --> object (using MFBD formula) --> aberrated images
nn_mode = MODE_2

#logfile = open(dir_name + '/log.txt', 'w')
#def print(*xs):
#    for x in xs:
#        logfile.write('%s' % x)
#    logfile.write("\n")
#    logfile.flush()
    
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

train_perc = 0.8
activation_fn = nn.ReLU
tt_weight = 0.0#0.001

learning_rate = 1.
weight_decay = 0.0
scheduler_decay = 0.5
scheduler_iterations = 20


if nn_mode == MODE_1:
    
    shuffle0 = False
    shuffle1 = True
    shuffle2 = False
    
    num_reps = 1000

    n_epochs_2 = 5
    n_epochs_1 = 1
    
    # How many frames to use in training
    num_frames = 128
    # How many objects to use in training
    num_objs = 100#0#None
    
    # How many frames of the same object are sent to NN input
    # Must be power of 2
    num_frames_input = 1
    
    batch_size = 128
    n_channels = 32
    
    sum_over_batch = True
    
    zero_avg_tiptilt = True
    
elif nn_mode == MODE_2:

    shuffle0 = False
    shuffle1 = False
    shuffle2 = False
    
    num_reps = 1000
    
    n_epochs_2 = 20
    n_epochs_1 = 1
    
    n_epochs_mode_2 = 10
    mode_2_index = 1
    
    # How many frames to use in training
    num_frames = 256
    # How many objects to use in training
    num_objs = 80#None
    
    # How many frames of the same object are sent to NN input
    # Must be power of 2
    num_frames_input = 1
    
    batch_size = 32
    n_channels = 64
    
    sum_over_batch = True
    
    zero_avg_tiptilt = True

    num_alphas_input = 10
    
    if zero_avg_tiptilt:
        n_test_frames = num_frames


no_shuffle = not shuffle0 and not shuffle2

assert(num_frames % num_frames_input == 0)
if sum_over_batch:
    if not train:
        assert((n_test_frames // num_frames_input) % batch_size == 0)

if dir_name is None:
    dir_name = "results" + time.strftime("%Y%m%d-%H%M%S")
    os.mkdir(dir_name)
    
images_dir_train = "images_in"
images_dir_test = "images_in"#images_in_test"

sys.path.append('../utils')
sys.path.append('..')

cuda = torch.cuda.is_available()        
n_gpus_available = torch.cuda.device_count()

device = torch.device(f"cuda:0" if cuda else "cpu")      

if train:
    sys.stdout = open(dir_name + '/log.txt', 'a')
else:
    device = torch.device(f"cuda:1" if cuda else "cpu")      

print(f"Device : {device}")      
    
#else:
#    dir_name = "."
#    images_dir = "../images_in_old"

#    sys.path.append('../../utils')
#    sys.path.append('../..')


import config
import misc
import plot
import psf
import psf_torch
import utils
#import gen_images
#import gen_data


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
    return Ds, objs, pupil, modes, diversity, coefs, positions, coords


'''
    Ds_in: [num_objs, num_frames, 2, nx, nx]
    
    Ds_out:  [num_objs*(num_frames-num_frames_input+1), 2*num_frames_input, nx, nx]
'''
def convert_data(Ds_in, objs_in, diversity_in=None, positions=None, coords=None):
    assert(Ds_in.shape[2] == 2)
    assert(Ds_in.shape[0] == objs_in.shape[0])
    num_objects = Ds_in.shape[0]
    num_frames = Ds_in.shape[1]
    Ds_out = np.zeros(((num_frames-num_frames_input+1)*num_objects, Ds_in.shape[2]*num_frames_input, Ds_in.shape[3], Ds_in.shape[4]))
    if objs_in is not None:
        objs_out = np.zeros(((num_frames-num_frames_input+1)*num_objects, objs_in.shape[1], objs_in.shape[2]))
    else:
        objs_out  = None
    if diversity_in is not None:
        diversity_out = np.zeros(((num_frames-num_frames_input+1)*num_objects, Ds_in.shape[2], Ds_in.shape[3], Ds_in.shape[4]))
    else:
        diversity_out = None
    ids = np.zeros((num_frames-num_frames_input+1)*num_objects, dtype='int')
    positions_out = np.zeros(((num_frames-num_frames_input+1)*num_objects, 2), dtype='int')
    coords_out = np.zeros(((num_frames-num_frames_input+1)*num_objects, 2), dtype='int')
        
    k = 0
    for i in np.arange(num_objects):
        l = 0
        Ds_k = np.zeros((Ds_in.shape[2]*num_frames_input, Ds.shape[3], Ds_in.shape[4]))
        diversity_k = np.zeros((Ds_in.shape[2], Ds_in.shape[3], Ds_in.shape[4]))
        for j in np.arange(num_frames):
            Ds_k[2*l, :, :] = Ds_in[i, j, 0, :, :]
            Ds_k[2*l+1, :, :] = Ds_in[i, j, 1, :, :]
            if diversity_out is not None and l == 0:
                if positions is None:
                    if len(diversity_in.shape) == 3:
                        # Diversities both for focus and defocus image
                        #diversity_out[k, :, :, 0] = diversity_in[0]
                        for div_i in np.arange(diversity_in.shape[0]):
                            diversity_k[1, :, :] += diversity_in[div_i]
                    else:
                        assert(len(diversity_in.shape) == 2)
                        # Just a defocus
                        diversity_k[1, :, :] = diversity_in
                else:
                    assert(len(diversity_in.shape) == 5)
                    #for div_i in np.arange(diversity_in.shape[2]):
                    #    #diversity_out[k, :, :, 0] = diversity_in[positions[i, 0], positions[i, 1], 0]
                    #    diversity_out[k, :, :, 1] += diversity_in[positions[i, 0], positions[i, 1], div_i]
                    #    #diversity_out[k, :, :, 1] = diversity_in[positions[i, 0], positions[i, 1], 1]
                    diversity_k[1, :, :] += diversity_in[positions[i, 0], positions[i, 1], 1]
            l += 1
            if l >= num_frames_input:
                Ds_out[k] = Ds_k
                if diversity_out is not None:
                    diversity_out[k] = diversity_k
                if objs_out is not None:
                    objs_out[k] = objs_in[i]
                ids[k] = i
                if positions is not None:
                    positions_out[k] = positions[i]
                if coords is not None:
                    coords_out[k] = coords[i]
                l = 0
                k += 1
                Ds_k = np.zeros((Ds_in.shape[2]*num_frames_input, Ds.shape[3], Ds_in.shape[4]))
                diversity_k = np.zeros((Ds_in.shape[2], Ds_in.shape[3], Ds_in.shape[4]))
    Ds_out = Ds_out[:k]
    if objs_out is not None:
        objs_out = objs_out[:k]
    if diversity_out is not None:
        diversity_out = diversity_out[:k]
    ids = ids[:k]
    positions_out = positions_out[:k]
    coords_out = coords_out[:k]
    #assert(k == (num_frames-num_frames_input+1)*num_objects)
    return Ds_out, objs_out, diversity_out, num_frames - num_frames_input + 1, ids, positions_out, coords_out


class Dataset(torch.utils.data.Dataset):
    def __init__(self, Ds, objs, diversities, positions, obj_ids):
        super(Dataset, self).__init__()
        
        self.Ds = Ds
        self.objs = objs
        self.diversities = diversities
        self.positions = positions
        self.obj_ids = obj_ids
               
    def __getitem__(self, index):
        return self.Ds[index].astype('float32'), self.diversities[index].astype('float32')
        
    def __len__(self):
        return len(self.Ds)
    
    def length(self):
        return len(self.Ds)


def weights_init_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv2d') != -1:
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, max_pooling=True, batch_normalization=True, num_convs=3, activation=activation_fn):
        super(ConvLayer, self).__init__()

        self.batch_normalization = batch_normalization
        self.out_channels = out_channels
        
        self.layers = []
        
        n_channels = in_channels
        for i in np.arange(num_convs):
            conv1 = nn.Conv2d(n_channels, out_channels, kernel_size=1, stride=1)
            conv2 = nn.Conv2d(n_channels, out_channels, kernel_size=kernel, stride=1, padding=kernel//2, padding_mode='reflect')
            n_channels = out_channels
            act = activation(inplace=True)
            if batch_normalization:
                bn = nn.BatchNorm2d(n_channels)
            else:
                bn = None
            self.layers.append((conv1, conv2, act, bn))
        if max_pooling:
            self.pool = nn.MaxPool2d(2)
        else:
            self.pool = None

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        for i in np.arange(len(self.layers)):
            conv1, conv2, act, bn = self.layers[i]
            conv1 = conv1.to(*args, **kwargs)
            conv2 = conv2.to(*args, **kwargs)
            act = act.to(*args, **kwargs)
            bn = bn.to(*args, **kwargs)
            self.layers[i] = (conv1, conv2, act, bn)
        return self

    def forward(self, x):
        for layer in self.layers:
            conv1, conv2, act, bn = layer
            x1 = conv1(x)
            x2 = conv2(x)
            x = act(x1 + x2)
            if bn is not None:
                x = bn(x)
        if self.pool is not None:
            x = self.pool(x)

        return x


class NN(nn.Module):
    def __init__(self, jmax, nx, num_frames, num_objs, pupil, modes):
        super(NN, self).__init__()

        self.jmax = jmax
        self.nx = nx
        self.num_frames = num_frames
        
        
        self.num_frames = num_frames
        assert(num_frames_input <= self.num_frames)
        self.num_objs = num_objs
        
        self.i1 = None # See set_data method for meaning
        self.i2 = None # See set_data method for meaning
        
        self.hanning = utils.hanning(nx, 10)
        self.filter = utils.create_filter(nx, freq_limit = 0.4)
        #self.pupil = pupil[nx//4:nx*3//4,nx//4:nx*3//4]
        #self.modes = modes[:, nx//4:nx*3//4,nx//4:nx*3//4]

        self.pupil = pupil
        self.modes = modes
        
        self.modes_orig = self.modes/utils.mode_scale[:, None, None]
        #normalization = np.max(np.abs(modes), axis=(1, 2))
        #self.modes = self.modes / normalization[:, None, None]
        

        self.modes_orig = modes
        self.pupil_orig = pupil

         
        pa_check = psf.phase_aberration(len(modes), start_index=1)
        pa_check.set_terms(modes)
        ctf_check = psf.coh_trans_func()
        ctf_check.set_phase_aberr(pa_check)
        ctf_check.set_pupil(pupil)
        #ctf_check.set_diversity(diversity[i, j])
        self.psf_check = psf.psf(ctf_check, corr_or_fft=False)
        
        pa = psf_torch.phase_aberration_torch(len(modes), start_index=1, device=device)
        pa.set_terms(modes)
        ctf = psf_torch.coh_trans_func_torch(device=device)
        ctf.set_phase_aberr(pa)
        ctf.set_pupil(pupil)
        #ctf.set_diversity(diversity[i, j])
        batch_size_per_gpu = max(1, batch_size//max(1, n_gpus))
        self.psf = psf_torch.psf_torch(ctf, num_frames=num_frames_input, batch_size=batch_size_per_gpu, set_diversity=True, 
                                 mode=nn_mode, sum_over_batch=sum_over_batch, fltr=self.filter, tt_weight=tt_weight, device=device)
        print("batch_size_per_gpu, num_frames_input", batch_size_per_gpu, num_frames_input)
        
        self.psf_test = psf_torch.psf_torch(ctf, num_frames=n_test_frames, batch_size=1, set_diversity=True, 
                                      mode=nn_mode, sum_over_batch=sum_over_batch, fltr=self.filter, device=device)
        
        num_defocus_channels = 2

        self.layers1 = []

        l = ConvLayer(in_channels=num_defocus_channels*num_frames_input, out_channels=n_channels, kernel=9, num_convs=1)
        self.layers1.append(l)
        l = ConvLayer(in_channels=l.out_channels, out_channels=2*n_channels, kernel=7)
        self.layers1.append(l)
        l = ConvLayer(in_channels=l.out_channels, out_channels=4*n_channels, kernel=5)
        self.layers1.append(l)
        l = ConvLayer(in_channels=l.out_channels, out_channels=4*n_channels)
        self.layers1.append(l)

        self.layers2 = []
        
        self.layers2.append(nn.Linear(l.out_channels*(nx//(2**len(self.layers1)))**2, 36*n_channels))
        self.layers2.append(activation_fn())
        self.layers2.append(nn.Linear(36*n_channels, 1024))
        self.layers2.append(activation_fn())
        size = 256
        self.layers2.append(nn.Linear(1024, size))
        self.layers2.append(activation_fn())

        self.lstm = nn.LSTM(size, size//2, batch_first=True, bidirectional=True, dropout=0.0)
        
        self.layers3 = []
        self.layers3.append(nn.Linear(size, size))
        self.layers3.append(activation_fn())
        self.layers3.append(nn.Linear(size, jmax*num_frames_input))
        
        self.optimizer = torch.optim.Adadelta(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        #self.loss_fn = nn.MSELoss().to(device)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_iterations, gamma=scheduler_decay)
        
        self.apply(weights_init_uniform)
        self.load_state()
        
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        for i in np.arange(len(self.layers1)):
            self.layers1[i] = self.layers1[i].to(*args, **kwargs)
        for i in np.arange(len(self.layers2)):
            self.layers2[i] = self.layers2[i].to(*args, **kwargs)
        for i in np.arange(len(self.layers3)):
            self.layers3[i] = self.layers3[i].to(*args, **kwargs)
        self.lstm = self.lstm.to(*args, **kwargs)
        return self
    
    def forward(self, data):

        if nn_mode == 1:
            image_input, diversity_input = data
        elif nn_mode == 2:
            image_input, diversity_input, DD_DP_PP_input, tt_sums_input = data
        

        x = image_input

        # Convolutional blocks
        for layer in self.layers1:
            x = layer(x)
            
        # Fully connected layers
        x = x.view(-1, x.size()[-1]*x.size()[-2]*x.size()[-3])
        for layer in self.layers2:
            x = layer(x)

        # We want to use LSQM over the whole batch,
        # so we make first dimension of size 1
        x = x.unsqueeze(dim=0)
        x, _ = self.lstm(x)
        x = x.squeeze()

        # Fully connected layers
        for layer in self.layers3:
            x = layer(x)
        
        alphas = x

        alphas = alphas.view(-1, num_frames_input, self.jmax)

        if zero_avg_tiptilt:
            tip_tilt_sums = torch.sum(alphas[:, :, :2], dim=(0, 1), keepdims=True).repeat(alphas.size()[0], alphas.size()[1], 1)
            
            if nn_mode == 1:
                tip_tilt_means = tip_tilt_sums / (alphas.size()[0] * alphas.size()[1])    
            else:
                #TODO double check
                tt_sums = tt_sums_input.view(-1, 1, 2).repeat(1, num_frames_input, 1)
                tip_tilt_sums = tip_tilt_sums + tt_sums
                tip_tilt_means = tip_tilt_sums / self.num_frames
            tip_tilt_means = torch.cat([tip_tilt_means, torch.zeros(alphas.size()[0], alphas.size()[1], alphas.size()[2]-2).to(device, dtype=torch.float32)], axis=2)
            alphas = alphas - tip_tilt_means
        
        # image_input is [batch_size, num_objects*num_frames*2, nx, nx]
        # mfbd_loss takes [batch_size, num_objects*num_frames, 2, nx, nx]
        #image_input = torch.transpose(image_input, 0, 1)
        #image_input = image_input.view(image_input.size()[0]//2, 2, image_input.size()[1], image_input.size()[2], image_input.size()[3])
        #image_input = torch.transpose(image_input, 1, 2)
        #image_input = torch.transpose(image_input, 0, 1)
        if nn_mode == 1:
            loss, num, den, num_conj, psf, wf = self.psf.mfbd_loss(image_input, alphas, diversity_input)
        else:
            loss, num, den, num_conj, DD, DP_real, DP_imag, PP, psf, wf = self.psf.mfbd_loss(image_input, alphas, diversity_input, DD_DP_PP=None)

        loss = torch.mean(loss)/nx/nx

        return loss, alphas, num, den, num_conj, psf, wf
        


    # Inputs should be grouped per object (first axis)
    def deconvolve(self, Ds, alphas, diversity, do_fft=True):
        num_objs = Ds.shape[0]
        #assert(len(alphas) == len(Ds))
        #assert(Ds.shape[3] == 2) # Ds = [num_objs, num_frames, nx, nx, 2]
        if len(Ds.shape) == 4:
            # No batch dimension
            num_frames = Ds.shape[0]
        else:
            num_frames = Ds.shape[1]
        if not train:
            assert(num_frames == n_test_frames)
        self.psf_test.set_num_frames(num_frames)
        self.psf_test.set_batch_size(num_objs)

        alphas = torch.tensor(alphas).to(device, dtype=torch.float32)
        diversity = torch.tensor(diversity).to(device, dtype=torch.float32)
        Ds = torch.tensor(Ds).to(device, dtype=torch.float32)
        #Ds = tf.reshape(tf.transpose(Ds, [0, 2, 3, 1, 4]), [num_objs, nx, nx, 2*num_frames])
        print("Ds", Ds.size())
        image_deconv, Ps, wf, loss = self.psf_test.deconvolve(Ds, alphas, diversity, do_fft=do_fft)
        return image_deconv, Ps, wf, loss
        
    # Inputs should be grouped per object (first axis)
    def Ds_reconstr(self, Ds, alphas, diversity):
        image_deconv, _, _, _ = self.deconvolve(Ds, alphas, diversity, do_fft=False)

        image_deconv = image_deconv.view(alphas.shape[0], 1, self.nx, self.nx)
        image_deconv = image_deconv.repeat(1, 2*alphas.shape[1], 1, 1)
        alphas = torch.tensor(alphas).to(device, dtype=torch.float32)
        diversity = torch.tensor(diversity).to(device, dtype=torch.float32)
        
        num_frames = alphas.shape[1]
        num_objs = alphas.shape[0]
        
        self.psf_test.set_batch_size(num_objs)
        self.psf_test.set_num_frames(num_frames)
        ret_val = self.psf_test.Ds_reconstr2(image_deconv, alphas, diversity)#tf.reshape(a3, [num_objs*(num_frames*jmax+2*nx*nx)]))
        #print("image_deconv", image_deconv.numpy().shape)
        return ret_val
        

    def set_data(self, Ds, objs, diversity, positions, train_perc=train_perc, train_data=True):
        if train_data:
            assert(Ds.shape[1] >= self.num_frames)
            #assert(self.num_frames <= Ds.shape[1])
            if self.num_objs is None or self.num_objs <= 0:
                self.num_objs = Ds.shape[0]
            self.num_objs = min(self.num_objs, Ds.shape[0])
            #assert(self.num_objs <= Ds.shape[0])
            assert(Ds.shape[2] == 2)
            if objs is None:
                # Just generate dummy array in case we don't have true object data
                objs = np.zeros((Ds.shape[0], Ds.shape[3], Ds.shape[4]))
            if shuffle1:
                i1 = random.randint(0, Ds.shape[0] + 1 - self.num_objs)
                i2 = random.randint(0, Ds.shape[1] + 1 - self.num_frames)
            else:
                if self.i1 is None:
                    self.i1 = self.num_objs * random.randint(0, Ds.shape[0]//self.num_objs)
                if self.i2 is None:
                    self.i2 = self.num_frames * random.randint(0, Ds.shape[1]//self.num_frames)
                print("i1, i2", self.i1, self.i2)
                # No shuffleing, but shift the used data window
                # along frames and objects
                i1 = self.i1
                i2 = self.i2
                self.i2 += self.num_frames
                if self.i2 > Ds.shape[1] - self.num_frames:
                    self.i2 = 0
                    self.i1 += self.num_objs
                    if self.i1 > Ds.shape[0] - self.num_objs:
                        self.i1 = 0
                
            Ds = Ds[i1:i1+self.num_objs, i2:i2+self.num_frames]
            objs = objs[i1:i1+self.num_objs]
            if positions is not None:
                positions = positions[i1:i1+self.num_objs]
        
            Ds, objs, diversities, num_frames, obj_ids, positions, _s = convert_data(Ds, objs, diversity, positions)
            
            med = np.median(Ds, axis=(2, 3), keepdims=True)
            #std = np.std(Ds, axis=(1, 2), keepdims=True)
            Ds -= med
            Ds = self.hanning.multiply(Ds, axis=2)
            Ds += med
            ##Ds /= std
            Ds /= med
            
                        
            if shuffle2 and (not sum_over_batch or batch_size == 1):
                # Shuffle the data
                random_indices = random.choice(len(Ds), size=len(Ds), replace=False)
                Ds = Ds[random_indices]
                if objs is not None:
                    objs = objs[random_indices]
                if diversities is not None:
                    diversities = diversities[random_indices]
                obj_ids = obj_ids[random_indices]
                positions = positions[random_indices]

            '''
            # Validation data has to be set first
            self.Ds = np.concatenate([Ds, self.Ds_validation])
            # Create new views
            self.Ds_train = self.Ds[:len(Ds)]
            self.Ds_validation = self.Ds[len(Ds):]
            
            self.objs = np.concatenate([objs, self.objs_validation])
            self.objs_train = self.objs[:len(objs)]
            self.objs_validation = self.objs[len(objs):]

            self.diversities = np.concatenate([diversities, self.diversities_validation])
            self.diversities_train = self.diversities[:len(diversities)]
            self.diversities_validation = self.diversities[len(diversities):]

            if np.max(obj_ids) >= np.min(self.obj_ids_validation):
                self.obj_ids_validation += np.max(obj_ids) + 1
            assert(np.max(obj_ids) + 1 == np.min(self.obj_ids_validation))
            self.obj_ids = np.concatenate([obj_ids, self.obj_ids_validation])
            self.obj_ids_train = self.obj_ids[:len(obj_ids)]
            self.obj_ids_validation = self.obj_ids[len(obj_ids):]
            '''
            
            self.Ds_train = Dataset(Ds, objs, diversities, positions, obj_ids)

        else:
            Ds = Ds[:, :self.num_frames]
            Ds_validation, objs_validation, diversities_validation, _, obj_ids_validation, positions_validation, _s = convert_data(Ds, objs, diversity, positions)
            med = np.median(Ds_validation, axis=(2, 3), keepdims=True)
            #std = np.std(Ds, axis=(1, 2), keepdims=True)
            Ds_validation -= med
            Ds_validation = self.hanning.multiply(Ds_validation, axis=2)
            Ds_validation += med
            ##Ds /= std
            Ds_validation /= med
            
            self.Ds_validation = Dataset(Ds_validation, objs_validation, diversities_validation, positions_validation, obj_ids_validation)
        

    def group_per_obj(self, Ds, alphas, diversities, obj_ids, DD_DP_PP=None, tt=None):
        unique_obj_ids = np.unique(obj_ids)
        used_obj_ids = dict()
        
        num_frames = Ds.shape[0]//len(unique_obj_ids)

        Ds_per_obj = np.empty((len(unique_obj_ids), num_frames, Ds.shape[1], Ds.shape[2], Ds.shape[3]))
        if alphas is not None:
            alphas_per_obj = np.empty((len(unique_obj_ids), num_frames, alphas.shape[1]))
        else:
            alphas_per_obj = None
        diversities_per_obj = np.empty((len(unique_obj_ids), diversities.shape[1], diversities.shape[2], diversities.shape[3]))
        
        if DD_DP_PP is not None:
            DD_DP_PP_sums_per_obj = np.zeros((len(unique_obj_ids), 4, self.nx, self.nx))
        else:
            DD_DP_PP_sums_per_obj = None

        if tt is not None:
            tt_sums_per_obj = np.zeros((len(unique_obj_ids), 2))
        else:
            tt_sums_per_obj = None

        for i in np.arange(len(Ds)):
            #if sum_over_batch:
            #    obj_id = obj_ids[i*batch_size]
            #else:
            obj_id = obj_ids[i]
            if not obj_id in used_obj_ids:
                used_obj_ids[obj_id] = (len(used_obj_ids), 0)
            obj_index, frame_index = used_obj_ids[obj_id]
            Ds_per_obj[obj_index, frame_index] = Ds[i]
            if alphas is not None:
                alphas_per_obj[obj_index, frame_index] = alphas[i]
            used_obj_ids[obj_id] = (obj_index, frame_index + 1)
            diversities_per_obj[obj_index] = diversities[i]
            if DD_DP_PP is not None:
                DD_DP_PP_sums_per_obj[obj_id] += DD_DP_PP[i]
            if tt is not None:
                tt_sums_per_obj[obj_id] += np.sum(tt[i], axis=0)

        return Ds_per_obj, alphas_per_obj, diversities_per_obj, DD_DP_PP_sums_per_obj, tt_sums_per_obj


    def predict_mode2(self, Ds, diversities, DD_DP_PP, obj_ids, tt_sums, alphas_in, Ds_diff=None):
        output_layer_model = Model(inputs=self.model.input, outputs=self.model.get_layer("output_layer").output)
        
        if nn_mode == MODE_3:
            output = output_layer_model.predict([Ds, diversities, DD_DP_PP, tt_sums, alphas_in, Ds_diff], batch_size=batch_size)
            alphas_layer_model = Model(inputs=self.model.input, outputs=self.model.get_layer("alphas_layer").output)
            alphas = alphas_layer_model.predict([Ds, diversities, DD_DP_PP, tt_sums, alphas_in, Ds_diff], batch_size=batch_size)
        else:
            output = output_layer_model.predict([Ds, diversities, DD_DP_PP, tt_sums, alphas_in], batch_size=batch_size)
            alphas_layer_model = Model(inputs=self.model.input, outputs=self.model.get_layer("alphas_layer").output)
            alphas = alphas_layer_model.predict([Ds, diversities, DD_DP_PP, tt_sums, alphas_in], batch_size=batch_size)
            #alphas = None

        DD_DP_PP_out = output[:, 1:, :, :]
        #DD_DP_PP_sums = dict()
        #DD_DP_PP_counts = dict()
        assert(len(DD_DP_PP_out) == len(Ds))
        tt = np.empty((len(alphas), num_frames_input, 2))
        for frame in np.arange(num_frames_input):
            tt[:, frame] = alphas[:, frame*jmax:frame*jmax+2]
        Ds_per_obj, alphas_per_obj, diversities_per_obj, DD_DP_PP_sums_per_obj, tt_sums_per_obj = self.group_per_obj(Ds, alphas, diversities, obj_ids, DD_DP_PP_out, tt)

        num_frames = alphas_per_obj.shape[1]
        assert(num_frames == self.num_frames)

            
        for i in np.arange(len(Ds)):
            if sum_over_batch:
                if i % batch_size == 0:
                    for j in np.arange(i, i+batch_size):
                        assert(obj_ids[j] == obj_ids[i])
                    DD_DP_PP_out_batch_sum = np.sum(DD_DP_PP_out[i:i+batch_size], axis=0)
                    DD_DP_PP[i:i+batch_size] = (DD_DP_PP_sums_per_obj[obj_ids[i]] - DD_DP_PP_out_batch_sum)/batch_size
            else:
                DD_DP_PP[i] = DD_DP_PP_sums_per_obj[obj_ids[i]] - DD_DP_PP_out[i]
            if i % batch_size*num_frames_input == 0:
                for j in np.arange(i, i+batch_size*num_frames_input):
                    assert(obj_ids[j] == obj_ids[i])
                tt_batch_sum = np.sum(tt[i:i+batch_size], axis=(0, 1))
                tt_sums[i:i+batch_size] = tt_sums_per_obj[obj_ids[i]] - tt_batch_sum
            if no_shuffle:#not shuffle0 and not shuffle1 and not shuffle2:
                # Use alphas of previous frame
                for j in np.arange(0, num_alphas_input):
                    if (i + j) % (num_frames/num_frames_input) < num_alphas_input:
                        alphas_in[i, jmax*num_frames_input*j:jmax*num_frames_input*(j+1)] = np.zeros_like(alphas[i])
                    else:
                        alphas_in[i, jmax*num_frames_input*j:jmax*num_frames_input*(j+1)] = alphas[i-num_alphas_input+j]
                        
    
    def do_epoch(self, data_loader, train=True):
        if train:
            self.train()
        else:
            self.eval()
        progress_bar = tqdm.tqdm(data_loader, file=sys.stdout)
        
        loss_sum = 0.
        count = 0.
        
        #num_data = data_loader.dataset.length()
        all_alphas = []#np.empty((num_data, jmax))
        all_num = []#np.empty((num_data, nx, nx))
        all_DP_conj = []#np.empty((num_data, nx, nx), dtype="complex64")
        all_den = []#np.empty((num_data, nx, nx))
        all_psf = []#np.empty((num_data, nx, nx))
        all_wf = []#np.empty((num_data, nx, nx))
        for batch_idx, (Ds, diversity) in enumerate(progress_bar):
            Ds = Ds.to(device)
            diversity = diversity.to(device)
            if train:
                self.optimizer.zero_grad()
                result = self((Ds, diversity))
            else:
                with torch.no_grad():
                    result = self((Ds, diversity))
            if nn_mode == 1:
                loss, alphas, num, den, DP_conj, psf, wf = result
                #print("num, den, DP_conj, psf, wf", num.size(), den.size(), DP_conj.size(), psf.size(), wf.size())
            else:
                loss, alphas, num, den, DP_conj, DD, DP_real, DP_imag, PP, psf, wf = result
            if train:
                loss.backward()
                self.optimizer.step()
            loss_sum += loss.item()
            count += 1
            
            all_alphas.append(alphas.detach().cpu().numpy())
            all_num.append(num.detach().cpu().numpy())
            all_den.append(den.detach().cpu().numpy())
            all_DP_conj.append(DP_conj.detach().cpu().numpy())
            all_psf.append(psf.detach().cpu().numpy())
            all_wf.append(wf.detach().cpu().numpy())

            if train:        
                progress_bar.set_postfix({"lr": self.optimizer.param_groups[0]['lr'], "Training error": loss_sum/count})
            else:
                progress_bar.set_postfix({"Validation error": loss_sum/count})
            
        #print("all_num", len(all_num), all_num[0].shape)
        all_alphas = np.reshape(np.asarray(all_alphas), [-1, jmax])
        all_num = np.reshape(np.asarray(all_num), [-1, nx, nx])
        all_den = np.reshape(np.asarray(all_den), [-1, nx, nx])
        all_DP_conj = np.reshape(np.asarray(all_DP_conj), [-1, nx, nx, 2]) # Complex array
        all_psf = np.reshape(np.asarray(all_psf), [-1, 2, nx, nx, 2]) # Complex array
        all_wf = np.reshape(np.asarray(all_wf), [-1, nx, nx])
        
        return loss_sum/count, all_alphas, all_num, all_den, all_DP_conj, all_psf, all_wf

    def save_state(self, state):
        torch.save(state, dir_name + "/state.tar")

    def load_state(self):
        try:
            state = torch.load(dir_name + "/state.tar", map_location=lambda storage, loc: storage)
            self.load_state_dict(state['state_dict'])       
            self.n_epochs_1 = state('n_epochs_1')
            self.n_epochs_2 = state('n_epochs_2')
            self.epoch = state('epoch')
            self.val_loss = state('val_loss')
        except:
            self.epoch = 0
            self.n_epochs_1 = n_epochs_1
            self.n_epochs_2 = n_epochs_2
            self.val_loss = float("inf")
            print("No state found")
        
 
    def do_train(self):
        jmax = self.jmax


        shuffle_epoch = True
        if sum_over_batch and batch_size > 1:
            shuffle_epoch = False

        Ds_train_loader = torch.utils.data.DataLoader(self.Ds_train, batch_size=batch_size, shuffle=shuffle_epoch, drop_last=True)
        Ds_validation_loader = torch.utils.data.DataLoader(self.Ds_validation, batch_size=batch_size, shuffle=shuffle_epoch, drop_last=False)

        # TODO: find out why it doesnt work with datasets
        #ds = tf.data.Dataset.from_tensors((self.Ds_train, output_data_train)).batch(batch_size)
        #ds_val = tf.data.Dataset.from_tensors((self.Ds_validation, output_data_validation)).batch(batch_size)
        
        if nn_mode == MODE_1:
            for epoch in np.arange(self.epoch, self.n_epochs_2):
                self.do_epoch(Ds_train_loader)
                self.scheduler.step()
                val_loss, _, _, _, _, _, _ = self.do_epoch(Ds_validation_loader, train=False)

                if True:#self.val_loss > history.history['val_loss'][-1]:
                    self.save_state({
                        'n_epochs_1': self.n_epochs_1,
                        'n_epochs_2': self.n_epochs_2,
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'state_dict': self.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                    })
                else:
                    self.load_state()
                    #print("Validation loss increased", self.val_loss, history.history['val_loss'][-1])
                    ##self.val_loss = float("inf")
                    #load_weights(model)
                    break
                
            self.epoch = 0
        elif nn_mode >= MODE_2:
            n_train = len(self.Ds_train)
            n_validation = len(self.Ds_validation)
            DD_DP_PP = np.zeros((n_train + n_validation, 4, nx, nx))
            DD_DP_PP_train = DD_DP_PP[:n_train]
            DD_DP_PP_validation = DD_DP_PP[n_train:n_train+n_validation]
            tt_sums = np.zeros((n_train + n_validation, 2))
            tt_sums_train = tt_sums[:n_train]
            tt_sums_validation = tt_sums[n_train:n_train+n_validation]
            alphas = np.zeros((n_train + n_validation, num_alphas_input*num_frames_input*jmax))
            alphas_train = alphas[:n_train]
            alphas_validation = alphas[n_train:n_train+n_validation]
            Ds_diff = None
                                        
            print(self.Ds.shape, DD_DP_PP.shape)
            self.predict_mode2(self.Ds, self.diversities, DD_DP_PP, self.obj_ids, tt_sums, alphas, Ds_diff)
            print("Index, num epochs, epoch_mode_2:", self.mode_2_index, self.n_epochs_2, self.epoch_mode_2)
            for epoch_mode_2 in np.arange(self.epoch_mode_2, self.n_epochs_mode_2):
                start_epoch = 0
                end_epoch = 10#1
                    
                for epoch in np.arange(start_epoch, end_epoch):
                    print("Not implemented")
                    #TODO:
                    
                self.predict_mode2(self.Ds, self.diversities, DD_DP_PP, self.obj_ids, tt_sums, alphas, Ds_diff)
                    
                self.epoch = 0
                print("epoch_mode_2, n_epochs_mode_2", epoch_mode_2, self.n_epochs_mode_2)
            self.epoch_mode_2 = 0
            self.mode_2_index += 1

        
        #######################################################################
        # Plot some of the training data results
        n_test = min(num_objs, 1)

        if nn_mode == MODE_1:
            _, pred_alphas, _, den, num_conj, psf, wf = self.do_epoch(Ds_train_loader, train=False)
        elif nn_mode >= MODE_2:
            print("Not implemented")
            #input_data = [self.Ds, self.diversities, DD_DP_PP, tt_sums, alphas]
            #pred_alphas = alphas_layer_model.predict(input_data, batch_size=batch_size)
            
        pred_Ds = None
        #pred_Ds = model.predict([self.Ds, self.objs], batch_size=1)
        #predicted_coefs = model.predict(Ds_train[0:n_test])
    
        #self.Ds_reconstr = np.array(self.Ds_train.shape[0], 1, self.Ds_train.shape[2], self.Ds_train.shape[3])
    
        #self.Ds_reconstr = np.array(self.Ds_train.shape[0], 1, self.Ds_train.shape[2], self.Ds_train.shape[3])
        obj_ids_used = []
        objs_reconstr = []
        i = 0
        while len(obj_ids_used) < n_test and i < len(self.Ds_train.objs):
            
            obj = self.Ds_train.objs[i]#np.reshape(self.objs[i], (self.nx, self.nx))
            found = False

            ###################################################################            
            # Just to plot results only for different objects
            for obj_id in obj_ids_used:
                if obj_id == self.Ds_train.obj_ids[i]:
                    found = True
                    break
            if found:
                i += 1
                continue
            ###################################################################            
            obj_ids_used.append(self.Ds_train.obj_ids[i])

            #DF = np.zeros((num_frames_input, 2, self.nx, self.nx), dtype="complex")
            ##DF = np.zeros((num_frames_input, 2, 2*self.pupil.shape[0]-1, 2*self.pupil.shape[0]-1), dtype="complex")
            #for l in np.arange(num_frames_input):
            #    D = self.Ds[i, :, :, 2*l]
            #    D_d = self.Ds[i, :, :, 2*l+1]
            #    #D = misc.sample_image(self.Ds[i, :, :, 2*l], (2.*self.pupil.shape[0] - 1)/nx)
            #    #D_d = misc.sample_image(self.Ds[i, :, :, 2*l+1], (2.*self.pupil.shape[0] - 1)/nx)
            #    DF[l, 0] = fft.fft2(D)
            #    DF[l, 1] = fft.fft2(D_d)
                
            DFs = []
            Ds_ = []
            alphas = []
            if pred_alphas is not None:
                for j in np.arange(i, len(self.Ds_train.objs)):
                    if self.Ds_train.obj_ids[j] == self.Ds_train.obj_ids[i]:
                        for l in np.arange(num_frames_input):
                            D = self.Ds_train.Ds[j, 2*l, :, :]
                            D_d = self.Ds_train.Ds[j, 2*l+1, :, :]
                            #D = misc.sample_image(Ds[j, :, :, 2*l], (2.*self.pupil.shape[0] - 1)/nx)
                            #D_d = misc.sample_image(Ds[j, :, :, 2*l+1], (2.*self.pupil.shape[0] - 1)/nx)
                            DF = fft.fft2(D)
                            DF_d = fft.fft2(D_d)
                            Ds_.append(self.Ds_train.Ds[j, 2*l:2*l+2, :, :])
                            DFs.append(np.array([DF, DF_d]))
                            alphas.append(pred_alphas[j, l*jmax:(l+1)*jmax])
                            if len(alphas) > 32:
                                break
                            #print("alphas", j, l, alphas[-1][0])
                            #if n_test_frames is not None and len(alphas) >= n_test_frames:
                            #    break
                    #if len(alphas) >= n_test_frames:
                    #    break
            Ds_ = np.asarray(Ds_)
            DFs = np.asarray(DFs, dtype="complex")
            alphas = np.asarray(alphas)
                
            print("tip-tilt mean", np.mean(alphas[:, :2], axis=0))
            
            if pred_alphas is not None:
                diversity = self.Ds_train.diversities[i]#np.concatenate((self.Ds_train.diversities[i, :, :, 0], self.Ds_train.diversities[i, :, :, 1]))
                #diversity = np.concatenate((self.diversities[i, :, :, 0][nx//4:nx*3//4,nx//4:nx*3//4], self.diversities[i, :, :, 1][nx//4:nx*3//4,nx//4:nx*3//4]))
                self.psf_check.coh_trans_func.set_diversity(diversity)
                #obj_reconstr = self.psf_check.deconvolve(DF, alphas=np.reshape(pred_alphas[i], (num_frames_input, jmax)), gamma=gamma, do_fft = True, fft_shift_before = False, ret_all=False, a_est=None, normalize = False, fltr=self.filter)
                obj_reconstr = self.psf_check.deconvolve(DFs, alphas=alphas, gamma=gamma, do_fft = True, fft_shift_before = False, ret_all=False, a_est=None, normalize = False, fltr=self.filter)
                obj_reconstr = fft.ifftshift(obj_reconstr)
                
                #obj_reconstr = self.deconvolve(num_frames_input, pred_alphas[i], self.diversities[i], self.Ds[i])
                
                objs_reconstr.append(obj_reconstr)
                #pred_Ds = self.psf_check.convolve(obj, alphas=np.reshape(pred_alphas[i], (num_frames_input, jmax)))
                pred_Ds = self.psf_check.convolve(obj, alphas=alphas)
            #print("pred_alphas", i, pred_alphas[i])

            num_rows = 0
            if pred_alphas is not None:
                num_rows += 1
            if pred_Ds is not None:
                num_rows += 2
            my_test_plot = plot.plot(nrows=num_rows, ncols=3)
            #my_test_plot.colormap(np.reshape(self.objs[i], (self.nx+1, self.nx+1)), [0])
            #my_test_plot.colormap(np.reshape(pred_objs[i], (self.nx+1, self.nx+1)), [1])
            row = 0
            if pred_alphas is not None:
                my_test_plot.colormap(obj, [row, 0], show_colorbar=True)
                my_test_plot.colormap(obj_reconstr, [row, 1])
                my_test_plot.colormap(obj - obj_reconstr, [row, 2])
                #my_test_plot.colormap(misc.sample_image(obj, (2.*self.pupil.shape[0] - 1)/nx) - obj_reconstr, [row, 2])
                row += 1
            if pred_Ds is not None:
                my_test_plot.colormap(self.Ds_train.Ds[i, 0, :, :], [row, 0])
                my_test_plot.colormap(pred_Ds[0, 0, :, :], [row, 1])
                my_test_plot.colormap(np.abs(self.Ds_train.Ds[i, 0, :, :] - pred_Ds[0, 0, :, :]), [row, 2])
                #my_test_plot.colormap(np.abs(misc.sample_image(self.Ds[i, :, :, 0], (2.*self.pupil.shape[0] - 1)/nx) - pred_Ds[0, :, :, 0]), [row, 2])
                row += 1
                my_test_plot.colormap(self.Ds_train.Ds[i, 1, :, :], [row, 0])
                my_test_plot.colormap(pred_Ds[0, 1, :, :], [row, 1])
                my_test_plot.colormap(np.abs(self.Ds_train.Ds[i, 1, :, :] - pred_Ds[0, 1, :, :]), [row, 2])
                #my_test_plot.colormap(np.abs(misc.sample_image(self.Ds[i, :, :, 1], (2.*self.pupil.shape[0] - 1)/nx) - pred_Ds[0, :, :, 1]), [row, 2])

            my_test_plot.save(f"{dir_name}/train{i}.png")
            my_test_plot.close()
            
            i += 1

 
        #######################################################################
    
    def coords_of_pos(self, coords, positions, pos):
        #print("pos", pos)
        max_pos = np.max(positions, axis = 0)
        if pos[0] < 0 or pos[1] < 0:
            # extrapolate left coord
            coord0 = self.coords_of_pos(coords, positions, [0, 0])
            if max_pos[0] == 0:
                if max_pos[1] == 0: # We have only single patch
                    coord1 = coord0 - [nx, nx] + [nx//10, nx//10]
                else:
                    coord1 = np.array([coord0[0] - nx + nx//10, 2*coord0[1] - self.coords_of_pos(coords, positions, [0, 1])[1]])
            elif max_pos[1] == 0:
                coord1 = np.array([2*coord0[0] - self.coords_of_pos(coords, positions, [1, 0])[0], coord0[1] - nx + nx//10])
            else:                
                coord1 = 2*coord0 - self.coords_of_pos(coords, positions, [1, 1])
            if pos[0] < 0:
                if pos[1] < 0:
                    return coord1
                else:
                    coord0 = self.coords_of_pos(coords, positions, [0, pos[1]])
                    return np.array([coord1[0], coord0[1]])
            else:
                coord0 = self.coords_of_pos(coords, positions, [pos[0], 0])
                return np.array([coord0[0], coord1[1]])
        #print("max_pos", max_pos, positions)
        if pos[0] > max_pos[0] or pos[1] > max_pos[1]:
            # extrapolate left coord
            coord0 = self.coords_of_pos(coords, positions, max_pos)
            if max_pos[0] == 0:
                if max_pos[1] == 0: # We have only single patch
                    coord1 = coord0 - [nx, nx] + [nx//10, nx//10]
                else:
                    coord1 = np.array([coord0[0] - nx + nx//10, 2*coord0[1] - self.coords_of_pos(coords, positions, max_pos - [0, 1])[1]])
            elif max_pos[1] == 0:
                coord1 = np.array([2*coord0[0] - self.coords_of_pos(coords, positions, max_pos - [1, 0])[0], coord0[1] - nx + nx//10])
            else:
                coord1 = 2*coord0 - self.coords_of_pos(coords, positions, max_pos - [1, 1])
            if pos[0] > max_pos[0]:
                if pos[1] > max_pos[1]:
                    return coord1
                else:
                    coord0 = self.coords_of_pos(coords, positions, [max_pos[0], pos[1]])
                    return np.array([coord1[0], coord0[1]])
            else:
                coord0 = self.coords_of_pos(coords, positions, [pos[0], max_pos[1]])
                return np.array([coord0[0], coord1[1]])
        filtr = np.all(positions == pos, axis=1)
        #print("pos, filtr", pos, filtr)
        return coords[filtr][0]
    
    def crop(self, i, coords, positions):
        nx = self.nx
        coord = coords[i]
        pos = positions[i]
        top_left_coord = self.coords_of_pos(coords, positions, pos - [1, 1]) + [nx, nx]
        bottom_right_coord = self.coords_of_pos(coords, positions, pos + [1, 1])
        print("top_left_coord, bottom_right_coord", top_left_coord, bottom_right_coord)
        
        top_left_coord  = (top_left_coord + coord)//2
        bottom_right_coord = (bottom_right_coord + coord + [nx, nx])//2
        top_left_delta = top_left_coord - coord 
        bottom_right_delta = bottom_right_coord - coord - [nx, nx]
    
        print("pos, coord, i", pos, coord, i)
        return top_left_coord, bottom_right_coord, top_left_delta, bottom_right_delta
    
    def do_test(self, Ds_, objs, diversity, positions, coords, file_prefix, true_coefs=None):
        estimate_full_image = True
        if coords is None:
            estimate_full_image = False
        #print("positions, coords", positions, coords)
        if objs is None:
            # Just generate dummy array in case we don't have true object data
            objs = np.zeros((Ds_.shape[0], Ds_.shape[3], Ds_.shape[4]))
        
        jmax = self.jmax
        
        num_frames = Ds_.shape[1]
        #num_objects = Ds_.shape[0]

        Ds, objs, diversities, num_frames, obj_ids, positions, coords = convert_data(Ds_, objs, diversity, positions, coords)
        #print("positions1, coords1", positions, coords)
        med = np.median(Ds, axis=(2, 3), keepdims=True)
        #std = np.std(Ds, axis=(1, 2), keepdims=True)
        Ds -= med
        Ds = self.hanning.multiply(Ds, axis=2)
        Ds += med
        ##Ds /= std
        Ds /= med
        
        Ds_test = Dataset(Ds, objs, diversities, positions, obj_ids)
        Ds_test_loader = torch.utils.data.DataLoader(Ds_test, batch_size=batch_size, shuffle=False, drop_last=False)
        

        start = time.time()    

        if nn_mode == MODE_1:
            losses, pred_alphas, _, dens, nums_conj, psf, wf = self.do_epoch(Ds_test_loader, train=False)
        elif nn_mode >= MODE_2:
            print("Not implemented")
            #DD_DP_PP = np.zeros((len(Ds), 4, nx, nx))
            #tt_sums = np.zeros((len(Ds), 2))
            #alphas = np.zeros((len(Ds), num_alphas_input*num_frames_input*jmax))
            #input_data = [Ds, diversities, DD_DP_PP, tt_sums, alphas]
            #Ds_diff = None
            #for epoch in np.arange(n_epochs_mode_2):
            #    self.predict_mode2(Ds, diversities, DD_DP_PP, obj_ids, tt_sums, alphas, Ds_diff)
            #pred_alphas = alphas_layer_model.predict(input_data, batch_size=batch_size)

        psf = fft.ifft2(psf[..., 0] + psf[..., 1]*1.j).real
        psf = fft.ifftshift(psf, axes=(2, 3))
            
        #Ds *= std
        Ds *= med
        #Ds += med
        
        end = time.time()
        print("Prediction time: " + str(end - start))

        #obj_reconstr_mean = np.zeros((self.nx-1, self.nx-1))
        #DFs = np.zeros((len(objs), 2, 2*self.nx-1, 2*self.nx-1), dtype='complex') # in Fourier space
        
        obj_ids_test = []
        
        cropped_Ds = []
        cropped_objs = []
        cropped_reconstrs = []
        cropped_reconstrs_true = []
        cropped_coords = []
        
        loss_diffs = []
        
        full_shape = np.zeros(2, dtype="int")
        
        #print("coords, pos", coords, positions)
    
        min_loss_diff = float("inf")
        max_loss_diff = -min_loss_diff
        
        min_loss_plot = None
        max_loss_plot = None

    
        for i in np.arange(len(objs)):
            #if len(obj_ids_test) >= n_test_objects:
            #    break
            obj = objs[i]#np.reshape(self.objs[i], (self.nx, self.nx))
            found = False
            ###################################################################            
            # Just to plot results only for different objects
            # TODO: use set for obj_ids_test, instead of list
            for obj_id in obj_ids_test:
                if obj_id == obj_ids[i]:
                    found = True
                    break
            if found:
                continue
            ###################################################################            
            obj_ids_test.append(obj_ids[i])
            
            if estimate_full_image:
                top_left_coord, bottom_right_coord, top_left_delta, bottom_right_delta = self.crop(i, coords, positions)
                print("Crop:", top_left_coord, bottom_right_coord, top_left_delta, bottom_right_delta)
                cropped_obj = obj[top_left_delta[0]:bottom_right_delta[0], top_left_delta[1]:bottom_right_delta[1]]
                cropped_objs.append(cropped_obj)
                
                cropped_coords.append(top_left_coord)
                cropped_Ds.append(Ds[i, 0, :, :][top_left_delta[0]:bottom_right_delta[0], top_left_delta[1]:bottom_right_delta[1]])
                full_shape += cropped_obj.shape
                print("cropped_obj.shape", cropped_obj.shape, top_left_coord)

            # Find all other realizations of the same object
            #DFs = []
            Ds_ = []
            alphas = []
            DP = 0.
            PP = 0.
            psfs = []
            wfs = []
            print("nums, dens, psf, wf", nums_conj.shape, dens.shape, psf.shape, wf.shape)
            if pred_alphas is not None:
                for j in np.arange(i, len(objs)):
                    if obj_ids[j] == obj_ids[i]:
                        for l in np.arange(num_frames_input):
                            #D = Ds[j, :, :, 2*l]
                            #D_d = Ds[j, :, :, 2*l+1]
                            #DF = fft.fft2(D)
                            #DF_d = fft.fft2(D_d)
                            Ds_.append(Ds[j, 2*l:2*l+2, :, :])
                            #DFs.append(np.array([DF, DF_d]))
                            alphas.append(pred_alphas[j, l*jmax:(l+1)*jmax])
                        DP += nums_conj[j//batch_size]
                        PP += dens[j//batch_size]
                        psfs.append(psf[j])
                        wfs.append(wf[j])
            Ds_ = np.asarray(Ds_)
            #DFs = np.asarray(DFs, dtype="complex")
            alphas = np.asarray(alphas)
            psfs = np.asarray(psfs)
            wfs = np.asarray(wfs)
                            
            #print("alphas", alphas.shape, Ds_.shape)
            print("tip-tilt mean", np.mean(alphas[:, :2], axis=0))
            
            #obj_reconstr = psf_check.deconvolve(np.array([[DF, DF_d]]), alphas=np.array([pred_alphas[i]]), gamma=gamma, do_fft = True, fft_shift_before = False, ret_all=False, a_est=None, normalize = False)
            #obj_reconstr = fft.ifftshift(obj_reconstr[0])
            #obj_reconstr_mean += obj_reconstr

            diversity = diversities[i]
            #print("diversity", diversities.shape)
            #diversity = np.concatenate((diversities[i, :, :, 0], diversities[i, :, :, 1]))
            #print("diversity", diversity.shape)
            obj_reconstr, loss = self.psf_test.reconstr_(torch.tensor(DP), psf_torch.to_complex(torch.tensor(PP)))#self.deconvolve(Ds_, alphas, diversity)
            obj_reconstr = obj_reconstr.cpu().numpy()
            #psf = psf.numpy()
            wfs = wfs*self.pupil
            psfs = fft.ifftshift(fft.ifft2(psfs).real, axes=(2, 3))
            

            if estimate_full_image:
                cropped_reconstrs.append(obj_reconstr[top_left_delta[0]:bottom_right_delta[0], top_left_delta[1]:bottom_right_delta[1]])

            #my_test_plot = plot.plot(nrows=2, ncols=2, width=8, height=6)
            #row = 0
            #my_test_plot.colormap(obj, [row, 0], show_colorbar=True)
            #my_test_plot.colormap(obj_reconstr, [row, 1])
            #row += 1
            #my_test_plot.colormap(Ds[i, :, :, 0], [row, 0])
            #my_test_plot.colormap(Ds[i, :, :, 1], [row, 1])
            #my_test_plot.save(f"{dir_name}/{file_prefix}{i}.png")
            #my_test_plot.close()

            if true_coefs is not None:
                true_alphas = true_coefs[obj_ids[i]]
                nf = min(alphas.shape[0], true_alphas.shape[0])

                if benchmarking_level >= 2:
                    obj_reconstr_true, psf_true, wf_true, loss_true = self.deconvolve(Ds_[:nf], true_alphas[:nf]/utils.mode_scale, diversity)
                    obj_reconstr_true = obj_reconstr_true.cpu().numpy()
                    
                    loss_diff = (loss.cpu().numpy() - loss_true.cpu().numpy())/nx/nx
                    loss_diffs.append(loss_diff)
                    
                    psf_true = psf_torch.real(psf_torch.ifft(psf_true))
                    psf_true = psf_true.cpu().numpy()
                    print("psf_true, obj_reconstr_true", psf_true.shape, obj_reconstr_true.shape)
                    wf_true = wf_true.cpu().numpy()*self.pupil
                    #psf_true = fft.ifftshift(fft.ifft2(fft.ifftshift(psf_true, axes=(1, 2))), axes=(1, 2)).real
                    psf_true = fft.ifftshift(psf_true, axes=(2, 3))
                    for j in np.arange(nf):
                        if j % 100 == 0:
                            print("psf_true[j]", np.max(psf_true[j]), np.min(psf_true[j]))
                            print("psf[j]", np.max(psfs[j]), np.min(psfs[j]))
                            print("psf MSE", np.sum((psf_true[j] - psfs[j])**2))
                            my_test_plot = plot.plot(nrows=2, ncols=5)
                            my_test_plot.colormap(utils.trunc(psf_true[j, 0], 1e-3), [0, 0], show_colorbar=True)
                            my_test_plot.colormap(utils.trunc(psfs[j, 0], 1e-3), [0, 1], show_colorbar=True)
                            my_test_plot.colormap(np.abs(psf_true[j, 0]-psfs[j, 0]), [0, 2], show_colorbar=True)
                            my_test_plot.colormap(obj_reconstr_true, [0, 3], show_colorbar=True)
                            my_test_plot.colormap(obj_reconstr, [0, 4], show_colorbar=True)
                            my_test_plot.colormap(wf_true[j], [1, 0], show_colorbar=True)
                            my_test_plot.colormap(wfs[j], [1, 1], show_colorbar=True)
                            my_test_plot.colormap(np.abs(wf_true[j]-wf[j]), [1, 2], show_colorbar=True)
                            my_test_plot.save(f"{dir_name}/psf{i // n_test_frames}_{j}.png")
                            my_test_plot.close()

                    if estimate_full_image:
                        cropped_reconstrs_true.append(obj_reconstr_true[top_left_delta[0]:bottom_right_delta[0], top_left_delta[1]:bottom_right_delta[1]])
                
                if benchmarking_level >= 1:
                    nrows = int(np.sqrt(jmax))
                    ncols = int(math.ceil(jmax/nrows))
                    my_test_plot = plot.plot(nrows=nrows, ncols=ncols, smart_axis=False)
                    row = 0
                    col = 0
                    #xs = np.arange(modes_nn.shape[0]*modes_nn.shape[1])
                    xs = np.arange(nf)
                    for coef_index in np.arange(alphas.shape[1]):
                        scale = 1.//utils.mode_scale[coef_index]
                        #scale = np.std(alphas[:, coef_index])/np.std(true_alphas[:, coef_index])
                        mean = np.mean(alphas[:, coef_index])
                        my_test_plot.plot(xs, np.reshape(alphas[:nf, coef_index]-mean, -1), [row, col], "r-")
                        my_test_plot.plot(xs, np.reshape(true_alphas[:nf, coef_index]*scale, -1), [row, col], "b--")
                        col += 1
                        if col >= ncols:
                            row += 1
                            col = 0
                    my_test_plot.save(f"{dir_name}/alphas{i // n_test_frames}.png")

                    if benchmarking_level >= 2:
                        if loss_diff > max_loss_diff or loss_diff < min_loss_diff:
                            if loss_diff > max_loss_diff:
                                if max_loss_plot is not None:
                                    max_loss_plot.close()
                                max_loss_diff = loss_diff
                                max_loss_plot = my_test_plot
            
                            if loss_diff < min_loss_diff:
                                if min_loss_plot is not None:
                                    min_loss_plot.close()
                                min_loss_diff = loss_diff
                                min_loss_plot = my_test_plot
                        else:
                            my_test_plot.close()
                    else:
                        my_test_plot.close()
                    
                
        if max_loss_plot is not None:
            max_loss_plot.save(f"{dir_name}/alphas_max.png")
            max_loss_plot.close()

        if min_loss_plot is not None:
            min_loss_plot.save(f"{dir_name}/alphas_min.png")
            min_loss_plot.close()

                

        if estimate_full_image:
            max_pos = np.max(positions, axis = 0)
            min_coord = np.min(cropped_coords, axis = 0)
            full_shape[0] = full_shape[0] // (max_pos[1] + 1)
            full_shape[1] = full_shape[1] // (max_pos[0] + 1)
            print("full_shape", full_shape)
            full_obj = np.zeros(full_shape)
            full_reconstr = np.zeros(full_shape)
            full_reconstr_true = np.zeros(full_shape)
            full_D = np.zeros(full_shape)
            
            for i in np.arange(len(cropped_objs)):
                x = cropped_coords[i][0]-min_coord[0]
                y = cropped_coords[i][1]-min_coord[1]
                s = cropped_objs[i].shape
                print(x, y, s)
                full_obj[x:x+s[0],y:y+s[1]] = cropped_objs[i]
                full_reconstr[x:x+s[0],y:y+s[1]] = cropped_reconstrs[i]
                if len(cropped_reconstrs_true) > 0:
                    full_reconstr_true[x:x+s[0],y:y+s[1]] = cropped_reconstrs_true[i]
                else:
                    full_reconstr_true[x:x+s[0],y:y+s[1]] = cropped_objs[i]
                full_D[x:x+s[0],y:y+s[1]] = cropped_Ds[i]

            plot_loss_diffs = len(loss_diffs) > 0
            num_cols = 3
            if plot_loss_diffs:
                num_cols += 1
                
            my_test_plot = plot.plot(nrows=1, ncols=num_cols, size=plot.default_size(len(full_obj), len(full_obj)))
            my_test_plot.set_default_cmap(cmap_name="Greys")
            #my_test_plot.colormap(utils.trunc(full_obj, 1e-3), [0], show_colorbar=True)
            min_val = min(np.min(full_reconstr_true), np.min(full_reconstr))
            max_val = max(np.max(full_reconstr_true), np.max(full_reconstr))
            #my_test_plot.colormap(utils.trunc(full_reconstr_true, 1e-3), [0])
            #my_test_plot.colormap(utils.trunc(full_reconstr, 1e-3), [1])
            my_test_plot.colormap(full_reconstr_true, [0], show_colorbar=True)#, vmin=min_val, vmax=max_val)
            my_test_plot.colormap(full_reconstr, [1])#, vmin=min_val, vmax=max_val)
            my_test_plot.colormap(full_D, [2])
            
            my_test_plot.set_axis_title([0], "MOMFBD")
            my_test_plot.set_axis_title([1], "Neural network")
            my_test_plot.set_axis_title([2], "Raw frame")

            if plot_loss_diffs:
                loss_diffs = np.reshape(loss_diffs, (max_pos[0] + 1, max_pos[1] + 1)).T
                loss_diffs = np.repeat(np.repeat(loss_diffs, 10, axis=1), 10, axis=0)
                max_val = max(abs(np.max(loss_diffs)), abs(np.min(loss_diffs)))
                my_test_plot.set_default_cmap(cmap_name="bwr")
                my_test_plot.colormap(dat=loss_diffs, ax_index=[3], vmin=-max_val, vmax=max_val, show_colorbar=True)
                my_test_plot.set_axis_title([3], "Losses")
            
            #my_test_plot.set_axis_title([0], "MOMFBD filtered")
            my_test_plot.save(f"{dir_name}/{file_prefix}.png")
            my_test_plot.close()
            
            max_obj = np.max(full_obj)
            min_obj = np.min(full_obj)
            c1_obj = (max_obj-min_obj)/(max_obj+min_obj)

            max_reconstr = np.max(full_reconstr)
            min_reconstr = np.min(full_reconstr)
            c1_reconstr = (max_reconstr-min_reconstr)/(max_reconstr+min_reconstr)

            print("Contrasts 1", c1_obj, c1_reconstr)
            print("Contrasts 2", np.std(full_obj), np.std(full_reconstr))
            

if train:

    Ds, objs, pupil, modes, diversity, true_coefs, positions, coords = load_data(data_files[0])
    
    for data_file in data_files[1:]:
        Ds3, objs3, pupil3, modes3, diversity3, true_coefs3, positions3, coords3 = load_data(data_file)
        Ds3 = Ds3[:,:Ds.shape[1]]
        Ds = np.concatenate((Ds, Ds3))
        objs = np.concatenate((objs, objs3))
        positions = np.concatenate((positions, positions3))

    nx = Ds.shape[3]
    jmax = len(modes)
    jmax_to_use = 4

    if shuffle0:
        random_indices = random.choice(Ds.shape[1], size=Ds.shape[1], replace=False)
        Ds = Ds[:, random_indices]

    #hanning = utils.hanning(nx, 10)
    #med = np.median(Ds, axis=(3, 4), keepdims=True)
    #std = np.std(Ds, axis=(3, 4), keepdims=True)
    #Ds -= med
    #Ds = hanning.multiply(Ds, axis=3)
    #Ds += med
    ##Ds /= std
    #Ds /= med
    
    try:
        Ds_test, objs_test, _, _, _, _, positions_test, _ = load_data(data_files[0]+"_valid")
        Ds_train = Ds
        n_test = min(Ds_test.shape[0], max(1, num_objs//10))
        Ds_test = Ds_test[:n_test, :min(Ds_test.shape[1], num_frames)]
        objs_test = objs_test[:n_test]
        positions_test = positions_test[:n_test]

        objs_train = objs
        positions_train = positions

        n_train = int(len(Ds))
        print("validation set: ", data_files[0]+"_valid")
        print("n_train, n_test", len(Ds), len(Ds_test))
    except:
        
        n_train = int(len(Ds)*train_perc)
        print("n_train, n_test", n_train, len(Ds) - n_train)
            
        Ds_train = Ds[:n_train]
        #num_frames_valid = num_frames_input*batch_size
        Ds_test = Ds[n_train:]#, :num_frames_valid]
        if objs is not None:
            objs_train = objs[:n_train]
            objs_test = objs[n_train:]
        else:
            objs_train = None
            objs_test = None
        
        if positions is not None:
            positions_train = positions[:n_train]
            positions_test = positions[n_train:]
        else:
            positions_train = None
            positions_test = None

    print("num_frames", Ds.shape[1])

    #if coords is not None:
    #    coords_test = coords[n_train:]
    #else:
    coords_test = None
    
    #Ds_mean = np.mean(Ds, axis=(2,3))
    #Ds_std = np.std(Ds, axis=(2,3))
    #Ds -= np.tile(np.reshape(Ds_mean, (Ds_mean.shape[0], Ds_mean.shape[0], 1, 1)), (1, 1, nx, nx))
    #Ds /= np.tile(np.reshape(Ds_std, (Ds_mean.shape[0], Ds_mean.shape[0], 1, 1)), (1, 1, nx, nx))
    
    my_test_plot = plot.plot()
    my_test_plot.colormap(Ds[0, 0, 0], show_colorbar=True)
    my_test_plot.save(dir_name + "/D0_train.png")
    my_test_plot.close()
    
    my_test_plot = plot.plot()
    my_test_plot.colormap(Ds[0, 0, 1])
    my_test_plot.save(dir_name + "/D0_d_train.png")
    my_test_plot.close()
    
    pupil_check = pupil[nx//4:nx*3//4,nx//4:nx*3//4]
    #pupil_check[np.where(pupil_check < 0.001)] = 0.
    #pupil_check[np.where(pupil_check > 0.1)] = 1.
    #pupil_check = np.ones_like(pupil_check)
    modes_check = modes[:, nx//4:nx*3//4,nx//4:nx*3//4]
    
    my_test_plot = plot.plot()
    if len(diversity.shape) == 5:
        my_test_plot.colormap(diversity[0, 0, 1], show_colorbar=True)
    elif len(diversity.shape) == 3:
        my_test_plot.colormap(diversity[1], show_colorbar=True)
    else:
        my_test_plot.colormap(diversity, show_colorbar=True)
    my_test_plot.save(dir_name + "/diversity.png")
    my_test_plot.close()

    my_test_plot = plot.plot()
    my_test_plot.colormap(pupil, show_colorbar=True)
    my_test_plot.save(dir_name + "/pupil_train.png")
    my_test_plot.close()
    
    #for i in np.arange(len(modes)):
    #    my_test_plot = plot.plot()
    #    my_test_plot.colormap(modes[i], show_colorbar=True)
    #    my_test_plot.save(dir_name + f"/mode{i}.png")
    #    my_test_plot.close()
        

    ###########################################################################
    # Null check of deconvolution
    pa_check = psf.phase_aberration([])#len(modes), start_index=1)
    pa_check.set_terms(np.array([]))#np.zeros((jmax, nx//2, nx//2)))#modes)
    ctf_check = psf.coh_trans_func()
    ctf_check.set_phase_aberr(pa_check)
    ctf_check.set_pupil(pupil_check)
    #ctf_check.set_diversity(diversity[i, j])
    psf_check = psf.psf(ctf_check)

    D = misc.sample_image(Ds[0, 0, 0], (nx - 1)/nx)
    #D = plt.imread("tests/psf_tf_test_input.png")[0:95, 0:95]
    #D = psf_check.critical_sampling(D, threshold=1e-3)

    #hanning = utils.hanning(D.shape[0], 20)
    #med = np.median(D)
    #D -= med
    #D = hanning.multiply(D)
    #D += med
    #D /= med
    
    D_d = D

    #D = misc.sample_image(Ds[0, 0, 0], (2.*nx - 1)/nx)
    #D_d = misc.sample_image(Ds[0, 0, 1], (2.*nx - 1)/nx)
    DF = fft.fft2(D)
    #DF[np.where(np.abs(DF) < np.std(D)/10)] = 0.
    DF[-90:, -90:] = 0.
    D1 = fft.ifft2(DF).real
    DF_d = DF#fft.fft2(D)#fft.fft2(D_d)
            
    #diversity = np.concatenate((diversity[0, :, :, 0], diversity[0, :, :, 1]))
    #self.psf_check.coh_trans_func.set_diversity(diversity)
    psf_check.coh_trans_func.set_diversity(np.zeros((2, nx//2, nx//2)))
    obj_reconstr = psf_check.deconvolve(np.array([[DF, DF_d]]), alphas=None, gamma=gamma, do_fft = True, fft_shift_before = False, 
                                        ret_all=False, a_est=None, normalize = False)
    obj_reconstr = fft.ifftshift(obj_reconstr)
    #D1 = psf_check.convolve(D)

    my_test_plot = plot.plot(nrows=1, ncols=3)
    my_test_plot.colormap(D, [0], show_colorbar=True)
    my_test_plot.colormap(D1, [1])
    my_test_plot.colormap(obj_reconstr, [2])

    my_test_plot.save(f"{dir_name}/null_deconv.png")
    my_test_plot.close()

    ###########################################################################
    n_test_frames = Ds_test.shape[1] # Necessary to be set (TODO: refactor)
    model = NN(jmax, nx, num_frames, num_objs, pupil, modes)
    model = model.to(device)

    sys.stdout.flush()

    model.set_data(Ds_test, objs_test, diversity, positions_test, train_data=False)
    for rep in np.arange(0, num_reps):
        model.set_data(Ds_train, objs_train, diversity, positions_train, train_data=True)
        print("Rep no: " + str(rep))
        #model.psf.set_jmax_used(jmax_to_use)
        model.do_train()
        
        if jmax_to_use <= jmax//2:
            jmax_to_use *= 2
        else:
            jmax_to_use = jmax
            

        #if rep % 5 == 0:
        #model.test(Ds_test, objs_test, diversity, positions_test, coords_test, "validation")
        
        #if np.mean(model.validation_losses[-10:]) > np.mean(model.validation_losses[-20:-10]):
        #    break
        #model.validation_losses = model.validation_losses[-20:]
else:

    #in_dir = "images"
    #image_file = None#"icont"
    #image_size = nx
    #tile=False
    #scale=1.0

    #num_angles = 1
    #num_subimages = n_test_objects

    #images = gen_images.gen_images(in_dir, None, image_file, image_size, tile, scale, num_subimages, num_angles, ret=True)
    #Ds, images, pupil, modes, diversity, true_coefs = gen_data.gen_data(images, n_test_frames, num_images=num_objs)

    
    Ds, objs, pupil, modes, diversity, true_coefs, positions, coords = load_data(data_files[0])

    nx = Ds.shape[3]
    jmax = len(modes)

    '''
    pa_check = psf.phase_aberration([])#len(modes), start_index=1)
    pa_check.set_terms(np.array([]))#np.zeros((jmax, nx//2, nx//2)))#modes)
    ctf_check = psf.coh_trans_func()
    ctf_check.set_phase_aberr(pa_check)
    ctf_check.set_pupil(pupil)
    #ctf_check.set_diversity(diversity[i, j])
    psf_check = psf.psf(ctf_check, corr_or_fft=False)
    for i in np.arange(100, 110):
        ###############################################################
        # DBG
        obj_tmp = psf_check.critical_sampling(objs[i])
        my_test_plot = plot.plot(nrows=1, ncols=3)
        my_test_plot.colormap(objs[i], [0])
        my_test_plot.colormap(obj_tmp, [1])
        my_test_plot.colormap(objs[i]-obj_tmp, [2])
        my_test_plot.save(f"{dir_name}/critical_sampling{i}.png")
        my_test_plot.close()
        ###############################################################
    print("Critical sampling tests")
    '''

    if n_test_objects is None:
        n_test_objects = Ds.shape[0]
    if n_test_frames is None:
        n_test_frames = Ds.shape[1]
    n_test_objects = min(Ds.shape[0], n_test_objects)
    
    assert(n_test_frames <= Ds.shape[1])
    
    print("n_test_objects, n_test_frames", n_test_objects, n_test_frames)
    #if nn_mode == MODE_2 and n_epochs_mode_2 > 0:
    #    assert(n_test_frames == num_frames_mode_2)
    
    max_pos = np.max(positions, axis = 0)

    print(max_pos)
    max_pos = np.floor(max_pos*np.sqrt(n_test_objects/len(Ds))).astype(int)
    print(max_pos)
    filtr = np.all(positions <= max_pos, axis=1)

    if no_shuffle:
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
    true_coefs = true_coefs[filtr, :stride*n_test_frames:stride]

    # TODO: Comment out #######################################################
    #np.savez_compressed(dir_name + '/Ds_tmp', Ds=Ds, objs=objs, pupil=pupil, modes=modes, diversity=diversity, 
    #                alphas=true_coefs, positions=positions, coords=coords)
    ###########################################################################
    
    #hanning = utils.hanning(nx, 10)
    #med = np.median(Ds, axis=(3, 4), keepdims=True)
    #std = np.std(Ds, axis=(3, 4), keepdims=True)
    #Ds -= med
    #Ds = hanning.multiply(Ds, axis=3)
    #Ds += med
    ##Ds /= std
    #Ds /= med
    
    print("true_coefs", true_coefs.shape)
    #print(positions)
    #print(coords)

    #mean = np.mean(Ds, axis=(3, 4), keepdims=True)
    #std = np.std(Ds, axis=(3, 4), keepdims=True)
    #Ds -= mean
    #Ds /= np.median(Ds, axis=(3, 4), keepdims=True)

    ###########################################################################
    # Some plots for test purposes
    my_test_plot = plot.plot()
    my_test_plot.colormap(Ds[0, 0, 0], show_colorbar=True)
    my_test_plot.save(dir_name + "/D0_test.png")
    my_test_plot.close()
    
    my_test_plot = plot.plot()
    my_test_plot.colormap(Ds[0, 0, 1])
    my_test_plot.save(dir_name + "/D0_d_test.png")
    my_test_plot.close()

    my_test_plot = plot.plot()
    my_test_plot.colormap(pupil, show_colorbar=True)
    my_test_plot.save(dir_name + "/pupil_test.png")
    my_test_plot.close()
    ###########################################################################

    model = NN(jmax, nx, n_test_frames, num_objs, pupil, modes)
    model = model.to(device)

    model.do_test(Ds, objs, diversity, positions, coords, "test", true_coefs=true_coefs)

#logfile.close()
