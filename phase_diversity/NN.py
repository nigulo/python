import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import datetime
import json

import sys

import os.path
import numpy.fft as fft
import tqdm

import time

import plot
import psf_torch
import utils
from Dataset import Dataset

INPUT_REAL = 0
INPUT_FOURIER = 1
INPUT_FOURIER_RATIO = 2

FRAME_DEPENDENCE_NONE = 0
FRAME_DEPENDENCE_GRU = 1
FRAME_DEPENDENCE_TRANSFORMER = 2

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        #n = m.in_features
        #y = 1.0/np.sqrt(n)
        #m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, activation, kernel=3, max_pooling=True, batch_normalization=True, num_convs=4):
        super(ConvLayer, self).__init__()

        self.batch_normalization = batch_normalization
        self.out_channels = out_channels
        
        self.layers = nn.ModuleList()
        
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
            self.layers.append(nn.ModuleList([conv1, conv2, act, bn]))
        if max_pooling:
            self.pool = nn.MaxPool2d(2)
        else:
            self.pool = None
    
    '''
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.layers = self.layers.to(*args, **kwargs)
        #for i in np.arange(len(self.layers)):
        #    layer = self.layers[i]
        #    conv1 = layer[0]
        #    conv2 = layer[1]
        #    act = layer[2]
        #    bn = layer[3]
        #    conv1 = conv1.to(*args, **kwargs)
        #    conv2 = conv2.to(*args, **kwargs)
        #    act = act.to(*args, **kwargs)
        #    if bn is not None:
        #        bn = bn.to(*args, **kwargs)
        #    self.layers[i] = nn.ModuleList([conv1, conv2, act, bn])
        if self.pool is not None:
            self.pool = self.pool.to(*args, **kwargs)
        return self
    '''

    def forward(self, x):
        for layer in self.layers:
            conv1 = layer[0]
            conv2 = layer[1]
            act = layer[2]
            bn = layer[3]
            x1 = conv1(x)
            x2 = conv2(x)
            x = x1 + act(x2)
            if bn is not None:
                x = bn(x)
        if self.pool is not None:
            x = self.pool(x)

        return x


class NN(nn.Module):
    def __init__(self, nx, pupil, modes, device, dir_name, state_file="state.tar"):
        super(NN, self).__init__()

        assert(nx == pupil.shape[0])

        self.device = device
        self.dir_name = dir_name
        self.state_file = state_file

        self.data_index = 0
        
        self.pupil = pupil
        self.modes = modes

        self.num_modes = len(self.modes)
        self.nx = nx # Should be part of config
        


        
    def save_state(self, state):
        date = datetime.datetime.now().strftime("%Y-%m-%dT%H")#:%M:%S.%f")
        state_file = f"state{date}.tar"
        torch.save(state, f"{self.dir_name}/{state_file}")
        state_file_link = f"{self.dir_name}/state.tar"
        try:
            os.remove(state_file_link)
        except:
            pass
        os.symlink(state_file, state_file_link)

    def load_state(self, state_file="state.tar"):
        try:
            state = torch.load(f"{self.dir_name}/{state_file}", map_location=self.device)
            self.load_state_dict(state["state_dict"])
            self.epoch = state["epoch"]
            self.val_loss = state["val_loss"]
            self.data_index = state["data_index"]
            self = self.to(self.device)
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            #self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        except Exception as e:
            print(e)
            print("No state found")
            self.epoch = 0
            self.val_loss = float("inf")
            self.apply(weights_init)
            self = self.to(self.device)
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            #self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)


    def create_conf(self, conf_file="conf"):
        conf = dict(
            num_modes = self.num_modes,
            nx = self.nx,
            num_epochs = 10,
            activation_fn = "ELU",
            learning_rate = 5e-5,
            weight_decay = 0.0,
            scheduler_decay = 1.0,
            scheduler_iterations = 20,
            momentum = .9,
            num_frames = 64,
            batch_size = 64,
            num_channels = 64,
            num_gpus=1,
            tt_weight = 0.0,#0.001
            sum_over_batch = True,
            zero_avg_tiptilt = False,
            tip_tilt_separated = False,
            input_type = INPUT_FOURIER_RATIO,
            tt_calib = False,
            pass_through = False,
            #use_lstm = True,
            use_neighbours=False,
            num_pix_apod = self.nx//4,
            num_pix_pad = 0,
            frame_dependence_model = FRAME_DEPENDENCE_GRU,
            num_latent = 128
            )
        
        file = open(f"{self.dir_name}/{conf_file}.json",'w')
        json.dump(conf, file)
        file.close()
        return conf
        

    def load_conf(self, conf_file="conf"):
        try:
            file = open(f"{self.dir_name}/{conf_file}.json",'r')
            conf = json.load(file)
            file.close()
        except Exception as e:
            print(e)
            print("No configuration found. Creating default configuration")
            conf = self.create_conf()
            

        assert(self.num_modes == conf["num_modes"])
        assert(self.nx == conf["nx"])
        self.num_epochs = conf["num_epochs"]
        self.learning_rate = conf["learning_rate"]
        self.weight_decay = conf["weight_decay"]
        self.scheduler_decay = conf["scheduler_decay"]
        self.scheduler_iterations = conf["scheduler_iterations"]
        self.momentum = conf["momentum"]
        self.num_frames = conf["num_frames"]
        self.batch_size = conf["batch_size"]
        self.num_channels = conf["num_channels"]
        self.num_gpus = conf["num_gpus"]
        self.tt_weight = conf["tt_weight"]
        self.sum_over_batch = conf["sum_over_batch"]
        self.zero_avg_tiptilt = conf["zero_avg_tiptilt"]
        self.tip_tilt_separated = conf["tip_tilt_separated"]
        self.input_type = conf["input_type"]
        self.tt_calib = conf["tt_calib"]
        self.pass_through = conf["pass_through"]
        #self.use_lstm = conf["use_lstm"]
        self.use_neighbours = conf["use_neighbours"]
        self.num_pix_apod = conf["num_pix_apod"]
        self.num_pix_pad = conf["num_pix_pad"]
        self.frame_dependence_model = conf["frame_dependence_model"]
        self.num_latent = conf["num_latent"]
            
        if conf["activation_fn"] == "ELU" :
            self.activation_fn = nn.ELU
        else:
            print("Using default activation function: ReLU")
            self.activation_fn = nn.ReLU

        self.hanning = utils.hanning(self.nx, self.num_pix_apod, num_pixel_padding=self.num_pix_pad)

    def init(self):
        self.load_conf()
        
        pa = psf_torch.phase_aberration_torch(self.num_modes, start_index=1, device=self.device)
        pa.set_terms(self.modes)
        ctf = psf_torch.coh_trans_func_torch(device=self.device)
        ctf.set_phase_aberr(pa)
        ctf.set_pupil(self.pupil)
        #ctf.set_diversity(diversity[i, j])
        batch_size_per_gpu = max(1, self.batch_size//max(1, self.num_gpus))
        self.psf = psf_torch.psf_torch(ctf, num_frames=1, batch_size=batch_size_per_gpu, set_diversity=True, 
                                       sum_over_batch=self.sum_over_batch, tt_weight=self.tt_weight, device=self.device)
        print("batch_size_per_gpu", batch_size_per_gpu)
        
        self.psf_test = psf_torch.psf_torch(ctf, batch_size=1, set_diversity=True, 
                                      sum_over_batch=self.sum_over_batch, device=self.device)
        
        num_in_channels = 2
        if self.use_neighbours:
            assert(self.frame_dependence_model == FRAME_DEPENDENCE_TRANSFORMER)
            #num_in_channels = 18#32
        if self.input_type == INPUT_FOURIER:
            num_in_channels = 6
            #if self.use_neighbours:
            #    num_in_channels = 6*9#32 + 64
        elif self.input_type == INPUT_FOURIER_RATIO:
            num_in_channels = 4
            
            #if self.use_neighbours:
            #    num_in_channels *= 9#16

        self.layers1 = nn.ModuleList()

        l = ConvLayer(in_channels=num_in_channels, out_channels=self.num_channels, activation=self.activation_fn, kernel=3, num_convs=2)
        self.layers1.append(l)
        l = ConvLayer(in_channels=l.out_channels, out_channels=self.num_channels, activation=self.activation_fn, kernel=3)
        self.layers1.append(l)
        l = ConvLayer(in_channels=l.out_channels, out_channels=self.num_channels, activation=self.activation_fn, kernel=3)
        self.layers1.append(l)
        l = ConvLayer(in_channels=l.out_channels, out_channels=self.num_channels, activation=self.activation_fn)
        self.layers1.append(l)

        self.layers2 = nn.ModuleList()
        
        size = self.num_latent#1024
        self.layers2.append(nn.Linear(l.out_channels*(self.nx//(2**len(self.layers1)))**2, size))
        self.layers2.append(self.activation_fn())

        #self.lstm = nn.LSTM(size, size//2, batch_first=True, bidirectional=True, dropout=0.0)
        if self.tip_tilt_separated:
            assert(self.frame_dependence_model == FRAME_DEPENDENCE_GRU)
            self.lstm_high = nn.GRU(size, size//2, batch_first=True, bidirectional=True, dropout=0.0)

            self.layers3_high = nn.ModuleList()
            self.layers3_high.append(nn.Linear(size, size))
            self.layers3_high.append(self.activation_fn())
            self.layers3_high.append(nn.Linear(size, size))
            self.layers3_high.append(self.activation_fn())
            self.layers3_high.append(nn.Linear(size, self.num_modes-2))
            #self.layers3_high.append(nn.Tanh())
            

            self.lstm_low = nn.GRU(size, size//2, batch_first=True, bidirectional=True, dropout=0.0)

            self.layers3_low = nn.ModuleList()
            self.layers3_low.append(nn.Linear(size, size))
            self.layers3_low.append(self.activation_fn())
            self.layers3_low.append(nn.Linear(size, size))
            self.layers3_low.append(self.activation_fn())
            self.layers3_low.append(nn.Linear(size, 2))
            #self.layers3_low.append(nn.Tanh())
        else:
            if self.frame_dependence_model == FRAME_DEPENDENCE_GRU:
                self.lstm = nn.GRU(size, size//2, batch_first=True, bidirectional=True, dropout=0.0)
            else:
                sys.path.append(os.path.join(os.path.dirname(__file__), "Transformer"))
                from ZernikeEncoder import ZernikeEncoder
                
                max_len = self.num_frames
                
                if self.use_neighbours:
                    max_len *= 9
                
                hyperparameters = {        
                    'input_dim' : size,                    # Embedding dimension for the Transformer Encoder
                    'num_heads' : 8,                      # Number of heads of the Encoder
                    'num_layers' : 6,                     # Number of layers of the Encoder
                    'ff_dim' : size*4,                     # Dimension of the internal fully connected networks in the Encoder (should be divisible by the number of heads)
                    'dropout' : 0.1,                      # Dropout in the Encoder
                    'norm_in' : True,                     # Order of the Layer Norm in the Encoder (always True for good convergence using PreNorm)
                    'weight_init_type' : 'xavier_normal', # Initialization of the Encoder
                    'max_len' : max_len,          # Use learning rate warmup during training (if PreNorm, not really necessary)        
                    }                
                self.zernike_encoder = ZernikeEncoder(hyperparameters, self.device)
                self.mask = torch.zeros((1, max_len), dtype=torch.bool).to(self.device)
            
            self.layers3 = nn.ModuleList()
            self.layers3.append(nn.Linear(size, self.num_modes))
        
        #######################################################################
        
        self.load_state(state_file=self.state_file)
        #self.loss_fn = nn.MSELoss().to(device)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.scheduler_iterations, gamma=self.scheduler_decay)
        #for p in self.parameters():
        #    print(p.name, p.numel())        
    
    '''
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.layers1 = self.layers1.to(*args, **kwargs)
        self.layers2 = self.layers2.to(*args, **kwargs)
        self.layers3 = self.layers3.to(*args, **kwargs)
        #for i in np.arange(len(self.layers1)):
        #    self.layers1[i] = self.layers1[i].to(*args, **kwargs)
        #for i in np.arange(len(self.layers2)):
        #    self.layers2[i] = self.layers2[i].to(*args, **kwargs)
        #for i in np.arange(len(self.layers3)):
        #    self.layers3[i] = self.layers3[i].to(*args, **kwargs)
        self.lstm = self.lstm.to(*args, **kwargs)
        return self
    '''
    
    def forward(self, data):

        image_input, diversity_input, tt_mean, alphas_input = data

        x = image_input
        if self.use_neighbours:
            neighbours = torch.flatten(image_input[:, 2:], start_dim=0, end_dim=1).view(-1, 2, self.nx, self.nx)
            image_input = image_input[:, :2]
            x = torch.cat([image_input, neighbours], dim=0)
        if self.input_type == INPUT_FOURIER:
            x_f = psf_torch.fft(psf_torch.to_complex(x))
            x = torch.cat([x, x_f[..., 0], x_f[..., 1]], dim=1)
        elif self.input_type == INPUT_FOURIER_RATIO:
            x_f = psf_torch.fft(psf_torch.to_complex(x))
            
            x = None
            
            for ch_ind in np.arange(x_f.size()[1], step=2):
                x_f_ch = x_f[:, ch_ind:ch_ind+2]
            
                x_f_mean = torch.mean(x_f_ch, dim=[0, 1], keepdim=True)
                #x_f = psf_torch.mul(x_f, psf_torch.to_complex(torch.from_numpy(self.filter2)).to(device, dtype=torch.float32))
                eps = psf_torch.to_complex(torch.tensor(1e-10)).to(self.device, dtype=torch.float32)
                x_f1 = psf_torch.div(x_f_ch[:, 0], x_f_mean[:, 0] + eps)
                x_f2 = psf_torch.div(x_f_ch[:, 1], x_f_mean[:, 0] + eps)

                #x_f3 = psf_torch.ifft(x_f1)
                #x_f4 = psf_torch.ifft(x_f2)
                
                x_f1 = torch.unsqueeze(x_f1, 1)
                x_f2 = torch.unsqueeze(x_f2, 1)

                #x_f3 = torch.unsqueeze(x_f3, 1)
                #x_f4 = torch.unsqueeze(x_f4, 1)
    
                x1 = torch.cat([x_f1[..., 0], x_f1[..., 1], x_f2[..., 0], x_f2[..., 1]], dim=1)
                #x1 = torch.cat([x1, x_f3[..., 0], x_f3[..., 1], x_f4[..., 0], x_f4[..., 1]], dim=1)
                if x is None:
                    x = x1
                else:
                    x = torch.cat([x, x1], dim=1)

            #x1 = torch.unsqueeze(x1, 1)
            
            #x = torch.cat([x1[..., 0], x1[..., 1], x_f1[..., 0], x_f1[..., 1]], dim=1)

        # Convolutional blocks
        for layer in self.layers1:
            x = layer(x)
            
        # Fully connected layers
        x = x.view(-1, x.size()[-1]*x.size()[-2]*x.size()[-3])
        for layer in self.layers2:
            x = layer(x)

        if self.tip_tilt_separated or self.frame_dependence_model != FRAME_DEPENDENCE_NONE:
            if self.pass_through:
                x1 = x
            # We want to use LSTM over the whole batch,
            # so we make first dimension of size 1
            #num_chunks = batch_size//16
            #x = x.view(num_chunks, x.size()[0]//num_chunks, x.size()[1])
            x = x.unsqueeze(dim=0)
        if self.tip_tilt_separated:
            x_high, _ = self.lstm_high(x)
            #x = x.reshape(x.size()[1]*num_chunks, x.size()[2])
            x_high = x_high.squeeze()
    
            # Fully connected layers
            i = len(self.layers3_high)
            for layer in self.layers3_high:
                x_high = layer(x_high)
                i -= 1
                #if i == 1:
                #    x_high = x_high*0.1
                #elif i == 0:
                #    x_high = x_high*2.0
                    

            x_low, _ = self.lstm_low(x[:, 1:, :])
            #x = x.reshape(x.size()[1]*num_chunks, x.size()[2])
            x_low = x_low.squeeze()
            
            i = len(self.layers3_low)
            # Fully connected layers
            for layer in self.layers3_low:
                x_low = layer(x_low)
                i -= 1
                #if i == 1:
                #    x_low = x_low*0.1
                #elif i == 0:
                #    x_low = x_low*4.0
                
            #x_low = x_low.view(-1, self.n_frames-1, 2)
            x_low = x_low.view(-1, 2)
            x_low = x_low.unsqueeze(dim=0)

            #x_high = x_high.view(-1, self.n_frames, (jmax-2)*num_frames_input)
            x_high = x_high.view(-1, self.num_modes-2)
            x_high = x_high.unsqueeze(dim=0)
    
            x_low = F.pad(x_low, (0,0,1,0,0,0), mode='constant', value=0.0)

            x = torch.cat([x_low, x_high], dim=-1)
            x = x.view(-1, self.num_modes)
                
        else:
            if self.frame_dependence_model == FRAME_DEPENDENCE_GRU:
                x, _ = self.lstm(x)
                #x = x.reshape(x.size()[1]*num_chunks, x.size()[2])
            elif self.frame_dependence_model == FRAME_DEPENDENCE_TRANSFORMER:
                x = self.zernike_encoder(x, self.mask)
            x = x.squeeze()
                
            
            if self.frame_dependence_model != FRAME_DEPENDENCE_NONE and self.pass_through:
                x = x + x1
            # Fully connected layers
            for layer in self.layers3:
                x = layer(x)
                
            x_low = F.pad(x[1:, :2], (0,0,1,0), mode='constant', value=0.0)
            x_high = x[:, 2:]
            x = torch.cat([x_low, x_high], dim=-1)
        
        alphas = x
        if self.use_neighbours:
            alphas = alphas[:self.num_frames]

        #################################################
        # To filter out only tip-tilt (for test purposes)
        #alphas = alphas[:, :2]
        #alphas = torch.cat([alphas, torch.zeros(alphas.size()[0], jmax-2).to(device, dtype=torch.float32)], axis = -1)
        #################################################


        #alphas = alphas.view(-1, num_frames_input, self.jmax)

        if self.zero_avg_tiptilt:
            tip_tilt_sums = torch.sum(alphas[:, :2], dim=(0), keepdims=True).repeat(alphas.size()[0], 1)
            
            tip_tilt_means = tip_tilt_sums / alphas.size()[0]    
            tip_tilt_means = torch.cat([tip_tilt_means, torch.zeros(alphas.size()[0], alphas.size()[1]-2).to(self.device, dtype=torch.float32)], axis=1)
            alphas_tt_zero = alphas - tip_tilt_means
            alphas = alphas_tt_zero
        elif tt_mean is not None:
            tt_mean = tt_mean.repeat(alphas.size()[0], 1)
            tt_mean = torch.cat([tt_mean, torch.zeros(alphas.size()[0], alphas.size()[1]-2).to(self.device, dtype=torch.float32)], axis=1)
            alphas_tt_zero = alphas - tt_mean
        else:
            alphas_tt_zero = alphas
        
        
        if alphas_input is not None:
            loss = torch.mean((alphas_input - alphas)**2)
            return loss
        else:
            loss, num, den, num_conj, psf, wf, DD = self.psf.mfbd_loss(image_input, alphas_tt_zero, diversity_input)
    
            loss = torch.mean(loss)#/nx/nx

            return loss, alphas, num, den, num_conj, psf, wf, DD
        


    # Inputs should be grouped per object (first axis)
    def deconvolve(self, Ds, alphas, diversity, do_fft=True, train=False):
        num_objs = Ds.shape[0]
        #assert(len(alphas) == len(Ds))
        #assert(Ds.shape[3] == 2) # Ds = [num_objs, num_frames, nx, nx, 2]
        if len(Ds.shape) == 4:
            # No batch dimension
            num_frames = Ds.shape[0]
        else:
            num_frames = Ds.shape[1]
        self.psf_test.set_num_frames(num_frames)
        self.psf_test.set_batch_size(num_objs)

        alphas = torch.tensor(alphas).to(self.device, dtype=torch.float32)
        diversity = torch.tensor(diversity).to(self.device, dtype=torch.float32)
        Ds = torch.tensor(Ds).to(self.device, dtype=torch.float32)
        #Ds = tf.reshape(tf.transpose(Ds, [0, 2, 3, 1, 4]), [num_objs, nx, nx, 2*num_frames])
        print("Ds", Ds.size())
        image_deconv, Ps, wf, loss = self.psf_test.deconvolve(Ds, alphas, diversity, do_fft=do_fft)
        return image_deconv, Ps, wf, loss
        
    def set_data(self, datasets, train_data=True):
        if train_data:
            self.Ds_train = Dataset(datasets, use_neighbours=self.use_neighbours)
        else:
            self.Ds_validation = Dataset(datasets, use_neighbours=self.use_neighbours)


    def do_batch(self, Ds, diversity, tt_mean=None, train=True, alphas_input=None):
        Ds = Ds.to(self.device)
        diversity = diversity.to(self.device)
        if tt_mean is not None:
            tt_mean = torch.from_numpy(tt_mean).to(self.device, dtype=torch.float32)
        #if alphas_input is not None:
        #    alphas_input = torch.from_numpy(alphas_input).to(device, dtype=torch.float32)
        if train:
            self.optimizer.zero_grad()
            result = self((Ds, diversity, tt_mean, alphas_input))
        else:
            with torch.no_grad():
                result = self((Ds, diversity, tt_mean, alphas_input))
        return result
    
    def do_epoch(self, data_loader, train=True, use_prefix=True, normalize=True, tt_mean=None):
        nx = self.nx
        prefix = None
        if train:
            prefix="Training error"
            self.train()
        else:
            if use_prefix:
                prefix="Validation error"
            self.eval()
        progress_bar = tqdm.tqdm(data_loader, file=sys.stdout)
        
        loss_sum = 0.
        count = 0.
        
        if not train:
            all_alphas = []
            all_num = []
            all_DP_conj = []
            all_den = []
            all_psf = []
            all_wf = []
            all_DD = []
        for batch_idx, (Ds, diversity) in enumerate(progress_bar):

            if self.tt_calib:
                dx = np.random.choice([1, -1, 2, -2])
                dy = np.random.choice([1, -1, 2, -2])
                Ds_shifted = torch.roll(Ds, shifts=(dx, dy), dims=(-2, -1))

            if normalize:
                med = np.array(data_loader.dataset.median)[None, None, None, None]
                if med is None:
                    med = np.median(Ds, axis=(0, 1, 2, 3), keepdims=True)
                    

                Ds -= med
                Ds = self.hanning.multiply(Ds, axis=2)
                Ds += med
                Ds /= med

                if self.tt_calib:
                    Ds_shifted -= med
                    Ds_shifted = self.hanning.multiply(Ds_shifted, axis=2)
                    Ds_shifted += med
                    Ds_shifted /= med
                
                # Mirror randomly for data augmentation purposes
                # Seemed not much useful
                #if random.choice([True, False]):
                #    Ds = torch.flip(Ds, [-1])
                #    diversity = torch.flip(diversity, [-1])
                #if random.choice([True, False]):
                #    Ds = torch.flip(Ds, [-2])
                #    diversity = torch.flip(diversity, [-2])
            
            ###################################################################
            # DEBUG
            ##if batch_idx == 0:
            ##    for i in np.arange(len(Ds)):
            #i = 0
            #my_test_plot = plot.plot(nrows=2, ncols=2)
            #my_test_plot.colormap(Ds.numpy()[i, 0], [0, 0], show_colorbar=True)
            #my_test_plot.colormap(Ds.numpy()[i, 1], [0, 1], show_colorbar=True)
            #my_test_plot.colormap(diversity.numpy()[i, 0], [1, 0], show_colorbar=True)
            #my_test_plot.colormap(diversity.numpy()[i, 1], [1, 1], show_colorbar=True)
            ##my_test_plot.save(f"{dir_name}/test_input_{i}.png")
            #my_test_plot.save(f"{dir_name}/test_input_{batch_idx}.png")
            #my_test_plot.close()
            ###################################################################
            result = self.do_batch(Ds, diversity, tt_mean=tt_mean, train=train)
            loss, alphas, num, den, DP_conj, psf, wf, DD = result
            
            
            ###################################################################
            
            if train:

                loss.backward(retain_graph=self.tt_calib)
                self.optimizer.step()

                if self.tt_calib:
                    loss1 = self.do_batch(Ds_shifted, diversity, tt_mean=tt_mean, train=train, alphas_input=alphas)
                    loss1.backward()
                    self.optimizer.step()

            else:
                all_alphas.append(alphas.cpu().numpy())
                all_num.append(num.cpu().numpy())
                all_den.append(den.cpu().numpy())
                all_DP_conj.append(DP_conj.cpu().numpy())
                all_psf.append(psf.cpu().numpy())
                all_wf.append(wf.cpu().numpy())
                all_DD.append(DD.cpu().numpy())
                
            loss_sum += loss.item()
            count += 1
            

            if train:        
                progress_bar.set_postfix({"lr": self.optimizer.param_groups[0]['lr'], prefix: loss_sum/count})
            else:
                if prefix is not None:
                    progress_bar.set_postfix({prefix: loss_sum/count})
            
        
        if train:        
            return loss_sum/count
        else:
            all_alphas = np.reshape(np.asarray(all_alphas), [-1, self.num_modes])
            tt_mean = np.mean(all_alphas[:, :2], axis=0, keepdims=True)
            tip_tilt_means = np.tile(tt_mean, (all_alphas.shape[0], 1))
            tip_tilt_means = np.concatenate((tip_tilt_means, np.zeros((all_alphas.shape[0], all_alphas.shape[1]-2))), axis=1)
            all_alphas = all_alphas - tip_tilt_means

            all_num = np.reshape(np.asarray(all_num), [-1, nx, nx])
            all_den = np.reshape(np.asarray(all_den), [-1, nx, nx])
            all_DP_conj = np.reshape(np.asarray(all_DP_conj), [-1, nx, nx, 2]) # Complex array
            all_psf = np.reshape(np.asarray(all_psf), [-1, 2, nx, nx, 2]) # Complex array
            all_wf = np.reshape(np.asarray(all_wf), [-1, nx, nx])
            all_DD = np.reshape(np.asarray(all_DD), [-1, nx, nx])
            
            return loss_sum/count, all_alphas, all_num, all_den, all_DP_conj, all_psf, all_wf, all_DD, tt_mean
            

    def do_train(self):
        self.test = False
        
        tt_mean = None


        shuffle_epoch = True
        if self.sum_over_batch and self.batch_size > 1:
            shuffle_epoch = False

        Ds_train_loader = torch.utils.data.DataLoader(self.Ds_train, batch_size=self.batch_size, shuffle=shuffle_epoch, drop_last=True)
        Ds_validation_loader = torch.utils.data.DataLoader(self.Ds_validation, batch_size=self.batch_size, shuffle=shuffle_epoch, drop_last=False)

        for epoch in np.arange(self.epoch, self.num_epochs):
            #self.do_epoch(Ds_train_loader)
            #_, _, _, _, _, _, _, _, tt_mean = self.do_epoch(Ds_train_loader, train=False, use_prefix=False)
            self.do_epoch(Ds_train_loader)#, tt_mean=np.random.normal(size=2))#tt_mean)
            self.scheduler.step()
            val_loss, _, _, _, _, _, _, _, _ = self.do_epoch(Ds_validation_loader, train=False)

            if True:#self.val_loss > history.history['val_loss'][-1]:
                self.save_state({
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'data_index': self.data_index,
                    'state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                })
            else:
                self.load_state()
                #print("Validation loss increased", self.val_loss, history.history['val_loss'][-1])
                ##self.val_loss = float("inf")
                #load_weights(model)
                break
            
        self.epoch = 0

        
        #######################################################################
        # Plot some of the training data results
        return
        n_test = 1

        _, pred_alphas, _, den, num_conj, psf, wf, DD, tt_mean = self.do_epoch(Ds_validation_loader, train=False, use_prefix=False)
            
        pred_Ds = None
    
        obj_ids_used = []
        objs_reconstr = []
        i = 0
        while len(obj_ids_used) < n_test and i < self.Ds_validation.length():
            
            obj_index, obj, _, _ = self.Ds_validation.get_obj_data(i)#np.reshape(self.objs[i], (self.nx, self.nx))
            found = False

            ###################################################################            
            # Just to plot results only for different objects
            for obj_id in obj_ids_used:
                if obj_id == obj_index:
                    found = True
                    break
            if found:
                i += 1
                continue
            ###################################################################            
            obj_ids_used.append(obj_index)

                
            Ds_ = []
            #DFs = []
            alphas = []
            if pred_alphas is not None:
                for j in range(i, self.Ds_validation.length()):
                    if self.Ds_validation.get_obj_data(j)[0] == obj_index:
                        for l in np.arange(1):
                            D = self.Ds_validation[j][0][2*l, :, :]
                            D_d = self.Ds_validation[j][0][2*l+1, :, :]
                            #D = misc.sample_image(Ds[j, :, :, 2*l], (2.*self.pupil.shape[0] - 1)/nx)
                            #D_d = misc.sample_image(Ds[j, :, :, 2*l+1], (2.*self.pupil.shape[0] - 1)/nx)
                            Ds_.append(np.array([D, D_d]))
                            #DF = fft.fft2(D)
                            #DF_d = fft.fft2(D_d)
                            #DFs.append(np.array([DF, DF_d]))
                            alphas.append(pred_alphas[j, l*self.num_modes:(l+1)*self.num_modes])
                            #if len(alphas) > 32:
                            #    break
                            #print("alphas", j, l, alphas[-1][0])
                            #if n_test_frames is not None and len(alphas) >= n_test_frames:
                            #    break
                    #if len(alphas) >= n_test_frames:
                    #    break
            Ds_ = np.asarray(Ds_)
            #DFs = np.asarray(DFs, dtype="complex")
            alphas = np.asarray(alphas)
                
            print("tip-tilt mean", np.mean(alphas[:, :2], axis=0))
            
            if pred_alphas is not None and obj is not None:
                obj_reconstr = self.deconvolve(Ds_, alphas=alphas, diversity=diversity, train=True)
                objs_reconstr.append(obj_reconstr)
                pred_Ds = self.psf_test.aberrate(obj, alphas=alphas)

            num_rows = 0
            if pred_alphas is not None and obj is not None:
                num_rows += 1
            if pred_Ds is not None:
                num_rows += 2
            my_test_plot = plot.plot(nrows=num_rows, ncols=3)
            #my_test_plot.colormap(np.reshape(self.objs[i], (self.nx+1, self.nx+1)), [0])
            #my_test_plot.colormap(np.reshape(pred_objs[i], (self.nx+1, self.nx+1)), [1])
            row = 0
            if pred_alphas is not None and obj is not None:
                my_test_plot.colormap(obj, [row, 0], show_colorbar=True)
                my_test_plot.colormap(obj_reconstr, [row, 1])
                my_test_plot.colormap(obj - obj_reconstr, [row, 2])
                #my_test_plot.colormap(misc.sample_image(obj, (2.*self.pupil.shape[0] - 1)/nx) - obj_reconstr, [row, 2])
                row += 1
            if pred_Ds is not None:
                Ds = self.Ds_validation[i][0]
                my_test_plot.colormap(Ds[0, :, :], [row, 0])
                my_test_plot.colormap(pred_Ds[0, 0, :, :], [row, 1])
                my_test_plot.colormap(np.abs(Ds[0, :, :] - pred_Ds[0, 0, :, :]), [row, 2])
                #my_test_plot.colormap(np.abs(misc.sample_image(self.Ds[i, :, :, 0], (2.*self.pupil.shape[0] - 1)/nx) - pred_Ds[0, :, :, 0]), [row, 2])
                row += 1
                my_test_plot.colormap(Ds[1, :, :], [row, 0])
                my_test_plot.colormap(pred_Ds[0, 1, :, :], [row, 1])
                my_test_plot.colormap(np.abs(Ds[1, :, :] - pred_Ds[0, 1, :, :]), [row, 2])
                #my_test_plot.colormap(np.abs(misc.sample_image(self.Ds[i, :, :, 1], (2.*self.pupil.shape[0] - 1)/nx) - pred_Ds[0, :, :, 1]), [row, 2])

            my_test_plot.save(f"{self.dir_name}/train{i}.png")
            my_test_plot.close()
            
            i += 1

 
        #######################################################################
    
    def coords_of_pos(self, coords, positions, pos):
        #print("pos", pos)
        max_pos = np.max(positions, axis = 0)
        min_pos = np.min(positions, axis = 0)
        if pos[0] < min_pos[0] or pos[1] < min_pos[1]:
            # extrapolate left coord
            coord0 = self.coords_of_pos(coords, positions, min_pos)
            if max_pos[0] == 0:
                if max_pos[1] == 0: # We have only single patch
                    coord1 = coord0 - [self.nx, self.nx] + [self.nx//10, self.nx//10]
                else:
                    coord1 = np.array([coord0[0] - self.nx + self.nx//10, 2*coord0[1] - self.coords_of_pos(coords, positions, min_pos + [0, 1])[1]])
            elif max_pos[1] == 0:
                coord1 = np.array([2*coord0[0] - self.coords_of_pos(coords, positions, min_pos + [1, 0])[0], coord0[1] - self.nx + self.nx//10])
            else:                
                coord1 = 2*coord0 - self.coords_of_pos(coords, positions, min_pos + [1, 1])
            if pos[0] < min_pos[0]:
                if pos[1] < min_pos[1]:
                    return coord1
                else:
                    coord0 = self.coords_of_pos(coords, positions, [min_pos[0], pos[1]])
                    return np.array([coord1[0], coord0[1]])
            else:
                coord0 = self.coords_of_pos(coords, positions, [pos[0], min_pos[1]])
                return np.array([coord0[0], coord1[1]])
        #print("max_pos", max_pos, positions)
        if pos[0] > max_pos[0] or pos[1] > max_pos[1]:
            # extrapolate left coord
            coord0 = self.coords_of_pos(coords, positions, max_pos)
            if max_pos[0] == 0:
                if max_pos[1] == 0: # We have only single patch
                    coord1 = coord0 - [self.nx, self.nx] + [self.nx//10, self.nx//10]
                else:
                    coord1 = np.array([coord0[0] - self.nx + self.nx//10, 2*coord0[1] - self.coords_of_pos(coords, positions, max_pos - [0, 1])[1]])
            elif max_pos[1] == 0:
                coord1 = np.array([2*coord0[0] - self.coords_of_pos(coords, positions, max_pos - [1, 0])[0], coord0[1] - self.nx + self.nx//10])
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
        return coords[filtr][0]
    
    def crop(self, obj_index, coords, positions):
        nx = self.nx
        coord = coords[obj_index]
        pos = positions[obj_index]
        top_left_coord = self.coords_of_pos(coords, positions, pos - [1, 1]) + [nx, nx]
        bottom_right_coord = self.coords_of_pos(coords, positions, pos + [1, 1])
        print("top_left_coord, bottom_right_coord", top_left_coord, bottom_right_coord)
        
        top_left_coord  = (top_left_coord + coord)//2
        bottom_right_coord = (bottom_right_coord + coord + [nx, nx])//2
        top_left_delta = top_left_coord - coord 
        bottom_right_delta = bottom_right_coord - coord - [nx, nx]
    
        print("pos, coord, obj_index", pos, coord, obj_index)
        return top_left_coord, bottom_right_coord, top_left_delta, bottom_right_delta
    
    def do_test(self, dataset, file_prefix, num_test_frames, true_coefs=None, benchmarking_level=0):
        
        self.test = True
        batch_size = num_test_frames
        self.batch_size = batch_size
        
        #num_frames = Ds_.shape[1]
        #num_objects = Ds_.shape[0]
        
        Ds_test = Dataset([dataset], use_neighbours=self.use_neighbours)

        estimate_full_image = True
        _, _, _, coord = Ds_test.get_obj_data(0)
        if coord is None:
            estimate_full_image = False

        Ds_test_loader = torch.utils.data.DataLoader(Ds_test, batch_size=batch_size, shuffle=False, drop_last=False)

        start = time.time()

            
        losses, pred_alphas, _, dens, nums_conj, psf_f, wf, DDs, tt_mean = self.do_epoch(Ds_test_loader, train=False)


        psf_f_np = psf_f[..., 0] + psf_f[..., 1]*1.j
        psf = fft.ifft2(psf_f_np).real
        psf = fft.ifftshift(psf, axes=(2, 3))
            
        end = time.time()
        print("Prediction time: " + str(end - start))

        obj_ids_test = []
        
        cropped_Ds = []
        cropped_objs = []
        cropped_reconstrs = []
        cropped_reconstrs_true = []
        cropped_coords = []
        
        loss_ratios = []
        
        full_shape = np.zeros(2, dtype="int")
        
        min_loss_ratio = float("inf")
        max_loss_ratio = 0.
        
        min_loss_plot = None
        max_loss_plot = None

    
        for i in range(Ds_test.length()):
            obj_index_i, obj, _, _ = Ds_test.get_obj_data(i)
            coords = Ds_test.get_coords()
            positions = Ds_test.get_positions()
            found = False
            ###################################################################
            # Just to plot results only for different objects
            # TODO: use set for obj_ids_test, instead of list
            for obj_id in obj_ids_test:
                if obj_id == obj_index_i:
                    found = True
                    break
            if found:
                continue
            ###################################################################
            obj_ids_test.append(obj_index_i)
            
            Ds, diversity = Ds_test[i]
            if estimate_full_image:
                top_left_coord, bottom_right_coord, top_left_delta, bottom_right_delta = self.crop(obj_index_i, coords, positions)
                print("Crop:", top_left_coord, bottom_right_coord, top_left_delta, bottom_right_delta)
                cropped_obj = obj[top_left_delta[0]:bottom_right_delta[0], top_left_delta[1]:bottom_right_delta[1]]
                cropped_objs.append(cropped_obj)
                
                cropped_coords.append(top_left_coord)
                cropped_Ds.append(Ds[0, :, :][top_left_delta[0]:bottom_right_delta[0], top_left_delta[1]:bottom_right_delta[1]])
                full_shape += cropped_obj.shape
                print("cropped_obj.shape", cropped_obj.shape, top_left_coord)

            # Find all other realizations of the same object
            #DFs = []
            Ds_ = []
            alphas = []
            DD = np.zeros_like(DDs[0])
            DP = np.zeros_like(nums_conj[0])
            DP1 = np.zeros((self.nx, self.nx), dtype="complex")
            PP = np.zeros_like(dens[0])
            psfs = []
            psfs_f = []
            wfs = []
            print("nums, dens, psf, wf", nums_conj.shape, dens.shape, psf.shape, wf.shape)
            if pred_alphas is not None:
                for j in range(i, Ds_test.length()):
                    obj_index_j, _, _, _ = Ds_test.get_obj_data(j)
                    if obj_index_j == obj_index_i:
                        Ds, _ = Ds_test[j]
                        for l in np.arange(1):
                            D = Ds[2*l:2*l+2, :, :]
                            #D_d = Ds[j, :, :, 2*l+1]
                            DF = fft.fft2(D)
                            #DF_d = fft.fft2(D_d)
                            Ds_.append(D)
                            #DFs.append(np.array([DF, DF_d]))
                            alphas.append(pred_alphas[j, l*self.num_modes:(l+1)*self.num_modes])
                            DP1 += np.sum(DF * psf_f_np[j].conj(), axis = 0)

                        if j % batch_size == 0:
                            DP += nums_conj[j//batch_size]
                            PP += dens[j//batch_size]
                            DD += DDs[j//batch_size]
                        psfs_f.append(psf_f[j])
                        psfs.append(psf[j])
                        wfs.append(wf[j])
            Ds_ = np.asarray(Ds_)
            #DFs = np.asarray(DFs, dtype="complex")
            alphas = np.asarray(alphas)
            psfs = np.asarray(psfs)
            psfs_f = np.asarray(psfs_f)
            wfs = np.asarray(wfs)
                            
            #print("alphas", alphas.shape, Ds_.shape)
            print("tip-tilt mean", np.mean(alphas[:, :2], axis=0))
            
            med = np.array(Ds_test.median)[None, None, None, None]
            if med is None:
                med = np.median(Ds_, axis=(0, 1, 2, 3), keepdims=True)
            Ds_ -= med
            Ds_ = self.hanning.multiply(Ds_, axis=2)
            Ds_ += med
            Ds_ /= med
            
            obj_reconstr, loss = self.psf_test.reconstr_(DP=torch.tensor(DP).to(self.device, dtype=torch.float32), 
                    PP=psf_torch.to_complex(torch.tensor(PP).to(self.device, dtype=torch.float32)), 
                    DD=torch.tensor(DD).to(self.device, dtype=torch.float32))
            
            obj_reconstr = obj_reconstr.cpu().numpy()
            wfs = wfs*self.pupil
            

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
                true_alphas = true_coefs[obj_index_i]
                nf = min(alphas.shape[0], true_alphas.shape[0])

                if benchmarking_level >= 2:
                    obj_reconstr_true, psf_true, wf_true, loss_true = self.deconvolve(Ds_[:nf], true_alphas[:nf]/utils.mode_scale, diversity)
                    obj_reconstr_true = obj_reconstr_true.cpu().numpy()
                    
                    loss_ratio = loss.cpu().numpy()/loss_true.cpu().numpy()
                    loss_ratios.append(loss_ratio)
                    
                    psf_true = psf_torch.real(psf_torch.ifft(psf_true))
                    psf_true = psf_true.cpu().numpy()
                    print("psf_true, obj_reconstr_true", psf_true.shape, obj_reconstr_true.shape)
                    wf_true = wf_true.cpu().numpy()*self.pupil
                    #psf_true = fft.ifftshift(fft.ifft2(fft.ifftshift(psf_true, axes=(1, 2))), axes=(1, 2)).real
                    psf_true = fft.ifftshift(psf_true, axes=(2, 3))
                    num_plot_frames = 5
                    frame_step = nf//num_plot_frames
                    my_test_plot = plot.plot(nrows=num_plot_frames, ncols=6, width=4.5, height=4)
                    my_test_plot.rescale_axis_title_font_size(1.3)
                    row = 0
                    zoom_start = psf_true.shape[2]//4
                    zoom_end = psf_true.shape[2] - zoom_start
                    for j in np.arange(nf):
                        if row < num_plot_frames and j % frame_step == 0:
                            print("psf_true[j]", np.max(psf_true[j]), np.min(psf_true[j]))
                            print("psf[j]", np.max(psfs[j]), np.min(psfs[j]))
                            print("psf MSE", np.sum((psf_true[j] - psfs[j])**2))
                            my_test_plot.colormap(utils.trunc(psf_true[j, 0, zoom_start:zoom_end, zoom_start:zoom_end], 1e-3), [row, 0], show_colorbar=False)
                            my_test_plot.colormap(utils.trunc(psfs[j, 0, zoom_start:zoom_end, zoom_start:zoom_end], 1e-3), [row, 1], show_colorbar=False)
                            my_test_plot.colormap(utils.trunc(psf_true[j, 1, zoom_start:zoom_end, zoom_start:zoom_end], 1e-3), [row, 2], show_colorbar=False)
                            my_test_plot.colormap(utils.trunc(psfs[j, 1, zoom_start:zoom_end, zoom_start:zoom_end], 1e-3), [row, 3], show_colorbar=False)
                            #my_test_plot.colormap(np.abs(psf_true[j, 0]-psfs[j, 0]), [0, 2], show_colorbar=True)
                            #my_test_plot.colormap(obj_reconstr_true, [0, 3], show_colorbar=True)
                            #my_test_plot.colormap(obj_reconstr, [0, 4], show_colorbar=True)
                            my_test_plot.colormap(wf_true[j], [row, 4], show_colorbar=False)
                            my_test_plot.colormap(wfs[j], [row, 5], show_colorbar=False)
                            #my_test_plot.colormap(np.abs(wf_true[j]-wfs[j]), [1, 2], show_colorbar=True)
                            row += 1
                    my_test_plot.set_axis_title("MOMFBD PSF (focus)", [0, 0])
                    my_test_plot.set_axis_title("NN PSF (focus)", [0, 1])
                    my_test_plot.set_axis_title("MOMFBD PSF (defocus)", [0, 2])
                    my_test_plot.set_axis_title("NN PSF (defocus)", [0, 3])
                    my_test_plot.set_axis_title("MOMFBD wavefront", [0, 4])
                    my_test_plot.set_axis_title("NN wavefront", [0, 5])
                    my_test_plot.toggle_axis()
                    my_test_plot.save(f"{self.dir_name}/psf{obj_index_i}.png")
                    my_test_plot.close()

                    if estimate_full_image:
                        cropped_reconstrs_true.append(obj_reconstr_true[top_left_delta[0]:bottom_right_delta[0], top_left_delta[1]:bottom_right_delta[1]])
                
                if benchmarking_level >= 1:
                    ncols = int(np.round(np.sqrt(self.num_modes)*0.7))
                    nrows = int(np.ceil(self.num_modes/ncols))
                    my_test_plot = plot.plot(nrows=nrows, ncols=ncols, smart_axis="x")
                    row = 0
                    col = 0
                    #xs = np.arange(modes_nn.shape[0]*modes_nn.shape[1])
                    xs = np.arange(nf)
                    for coef_index in np.arange(alphas.shape[1]):
                        scale = 1.//utils.mode_scale[coef_index]
                        #scale = np.std(alphas[:, coef_index])/np.std(true_alphas[:, coef_index])
                        #mean = np.mean(alphas[:, coef_index])
                        my_test_plot.plot(xs, np.reshape(alphas[:nf, coef_index], -1), "r-", [row, col])
                        my_test_plot.plot(xs, np.reshape(true_alphas[:nf, coef_index]*scale, -1), "b--", [row, col])
                        col += 1
                        if col >= ncols:
                            row += 1
                            col = 0
                    my_test_plot.save(f"{self.dir_name}/alphas{i // num_test_frames}.png")

                    if benchmarking_level >= 2:
                        if loss_ratio > max_loss_ratio or loss_ratio < min_loss_ratio:
                            if loss_ratio > max_loss_ratio:
                                if max_loss_plot is not None:
                                    max_loss_plot.close()
                                max_loss_ratio = loss_ratio
                                max_loss_plot = my_test_plot
            
                            if loss_ratio < min_loss_ratio:
                                if min_loss_plot is not None:
                                    min_loss_plot.close()
                                min_loss_ratio = loss_ratio
                                min_loss_plot = my_test_plot
                        else:
                            my_test_plot.close()
                    else:
                        my_test_plot.close()
                    
                
        if max_loss_plot is not None:
            max_loss_plot.save(f"{self.dir_name}/alphas_max.png")
            max_loss_plot.close()

        if min_loss_plot is not None:
            min_loss_plot.save(f"{self.dir_name}/alphas_min.png")
            min_loss_plot.close()

                

        if estimate_full_image:
            max_pos = np.max(positions, axis = 0)
            min_pos = np.min(positions, axis = 0)
            min_coord = np.min(cropped_coords, axis = 0)
            full_shape[0] = full_shape[0] // (max_pos[1] - min_pos[1] + 1)
            full_shape[1] = full_shape[1] // (max_pos[0] - min_pos[0] + 1)
            print("full_shape", full_shape)
            full_obj = np.zeros(full_shape)
            full_reconstr = np.zeros(full_shape)
            full_reconstr_true = np.zeros(full_shape)
            full_D = np.zeros(full_shape)
            
            zoomin = False
            num_rows = 1
            if len(cropped_objs) >= 100:
                zoomin = True
                num_rows = 2
                cropped_coords_2 = np.reshape(cropped_coords, (max_pos[0] - min_pos[0] + 1, max_pos[1] - min_pos[1] + 1, 2))
                zoom_start_patch = np.asarray(cropped_coords_2.shape) // 2 - 2
                zoom_end_patch = zoom_start_patch + 5

                zoom_x1 = cropped_coords_2[zoom_start_patch[0], zoom_start_patch[1], 0]-min_coord[0]
                zoom_y1 = cropped_coords_2[zoom_start_patch[0], zoom_start_patch[1], 1]-min_coord[1]
                zoom_x2 = cropped_coords_2[zoom_end_patch[0], zoom_end_patch[1], 0]-min_coord[0]
                zoom_y2 = cropped_coords_2[zoom_end_patch[0], zoom_end_patch[1], 1]-min_coord[1]
                
                zoom_obj = np.zeros((zoom_x2 - zoom_x1, zoom_y2 - zoom_y1))
                zoom_reconstr = np.zeros((zoom_x2 - zoom_x1, zoom_y2 - zoom_y1))
                zoom_reconstr_true = np.zeros((zoom_x2 - zoom_x1, zoom_y2 - zoom_y1))
                zoom_D = np.zeros((zoom_x2 - zoom_x1, zoom_y2 - zoom_y1))
                

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
                
                if zoomin:
                    if x >= zoom_x1 and x < zoom_x2 and y >= zoom_y1 and y < zoom_y2:
                        zoom_obj[x-zoom_x1:x+s[0]-zoom_x1,y-zoom_y1:y+s[1]-zoom_y1] = cropped_objs[i]
                        zoom_reconstr[x-zoom_x1:x+s[0]-zoom_x1,y-zoom_y1:y+s[1]-zoom_y1] = cropped_reconstrs[i]
                        if len(cropped_reconstrs_true) > 0:
                            zoom_reconstr_true[x-zoom_x1:x+s[0]-zoom_x1,y-zoom_y1:y+s[1]-zoom_y1] = cropped_reconstrs_true[i]
                        else:
                            zoom_reconstr_true[x-zoom_x1:x+s[0]-zoom_x1,y-zoom_y1:y+s[1]-zoom_y1] = cropped_objs[i]
                        zoom_D[x-zoom_x1:x+s[0]-zoom_x1,y-zoom_y1:y+s[1]-zoom_y1] = cropped_Ds[i]
                    

            plot_loss_ratios = len(loss_ratios) > 0
            num_cols = 2
            if plot_loss_ratios:
                num_cols += 1
                
            my_test_plot = plot.plot(nrows=num_rows, ncols=num_cols, size=plot.default_size(len(full_obj)*2, len(full_obj)*2))
            my_test_plot.set_default_cmap(cmap_name="Greys")
            #my_test_plot.colormap(utils.trunc(full_obj, 1e-3), [0], show_colorbar=True)
            #min_val = min(np.min(full_reconstr_true), np.min(full_reconstr))
            #max_val = max(np.max(full_reconstr_true), np.max(full_reconstr))
            #my_test_plot.colormap(utils.trunc(full_reconstr_true, 1e-3), [0])
            #my_test_plot.colormap(utils.trunc(full_reconstr, 1e-3), [1])
            my_test_plot.colormap(full_reconstr_true, [0, 0], show_colorbar=True)#, vmin=min_val, vmax=max_val)
            my_test_plot.colormap(full_reconstr, [0, 1])#, vmin=min_val, vmax=max_val)
            #my_test_plot.colormap(full_D, [2])
            
            my_test_plot.set_axis_title("MOMFBD", [0, 0])
            my_test_plot.set_axis_title("Neural network", [0, 1])
            #my_test_plot.set_axis_title([2], "Raw frame")

            if plot_loss_ratios:
                loss_ratios = np.reshape(loss_ratios, (max_pos[0] - min_pos[0] + 1, max_pos[1] - min_pos[1] + 1))
                loss_ratios10 = np.repeat(np.repeat(loss_ratios, 10, axis=1), 10, axis=0)
                max_loss_ratio = np.max(loss_ratios)
                min_loss_ratio = np.min(loss_ratios)                
                if min_loss_ratio < 1.:
                    min_loss_ratio = 2. - min_loss_ratio
                max_val = max(max_loss_ratio, min_loss_ratio)
                my_test_plot.set_default_cmap(cmap_name="bwr")
                my_test_plot.colormap(dat=loss_ratios10, ax_index=[0, num_cols-1], vmin=2.-max_val, vmax=max_val, show_colorbar=True, colorbar_prec="1.2")
                my_test_plot.set_axis_title(r"$L_{\rm NN}/L_{\rm MOMFBD}$", [0, num_cols-1])
            
            if zoomin:
                (x_low, x_high), (y_low, y_high) = my_test_plot.get_axis_limits(ax_index=[0, 0])
                width = x_high - x_low
                height = y_high - y_low
                #x and y swapped and y is reversed
                #TODO should actually change labels above
                zoom_x1, zoom_y1 = zoom_y1, zoom_x1
                zoom_x2, zoom_y2 = zoom_y2, zoom_x2
                zoom_y1 *= height/full_obj.shape[0]
                zoom_y2 *= height/full_obj.shape[0]
                zoom_x1 *= width/full_obj.shape[1]
                zoom_x2 *= width/full_obj.shape[1]
                zoom_y1 = y_high - zoom_y1
                zoom_y2 = y_high - zoom_y2
                zoom_x1 = x_low + zoom_x1
                zoom_x2 = x_low + zoom_x2
                my_test_plot.rectangle(zoom_x1, zoom_y1, zoom_x2, zoom_y2, ax_index=[0, 0], edgecolor="red", linestyle='--', linewidth=5.0, alpha=1.0)
                my_test_plot.rectangle(zoom_x1, zoom_y1, zoom_x2, zoom_y2, ax_index=[0, 1], edgecolor="red", linestyle='--', linewidth=5.0, alpha=1.0)

                my_test_plot.set_default_cmap(cmap_name="Greys")
                my_test_plot.colormap(zoom_reconstr_true, [1, 0], show_colorbar=True)#, vmin=min_val, vmax=max_val)
                my_test_plot.colormap(zoom_reconstr, [1, 1])#, vmin=min_val, vmax=max_val)
                if plot_loss_ratios:
                    loss_ratios_zoom = loss_ratios[zoom_start_patch[0]:zoom_end_patch[0], zoom_start_patch[1]:zoom_end_patch[1]]
                    loss_ratios10 = np.repeat(np.repeat(loss_ratios_zoom, 10, axis=1), 10, axis=0)
                    max_loss_ratio = np.max(loss_ratios_zoom)
                    min_loss_ratio = np.min(loss_ratios_zoom)                
                    if min_loss_ratio < 1.:
                        min_loss_ratio = 2. - min_loss_ratio
                    max_val = max(max_loss_ratio, min_loss_ratio)
                    my_test_plot.set_default_cmap(cmap_name="bwr")
                    my_test_plot.colormap(dat=loss_ratios10, ax_index=[1, num_cols-1], vmin=2.-max_val, vmax=max_val, show_colorbar=True, colorbar_prec="1.2")
                    
                    
            my_test_plot.toggle_axis()
            #my_test_plot.set_axis_title([0], "MOMFBD filtered")
            my_test_plot.tight_layout(pad=20, w_pad=15)
            my_test_plot.save(f"{self.dir_name}/{file_prefix}.png")
            my_test_plot.close()
            
            max_obj = np.max(full_obj)
            min_obj = np.min(full_obj)
            c1_obj = (max_obj-min_obj)/(max_obj+min_obj)

            max_reconstr = np.max(full_reconstr)
            min_reconstr = np.min(full_reconstr)
            c1_reconstr = (max_reconstr-min_reconstr)/(max_reconstr+min_reconstr)

            print("Contrasts 1", c1_obj, c1_reconstr)
            print("Contrasts 2", np.std(full_obj), np.std(full_reconstr))
            

            # Plot spectra            
            #h = utils.hanning(full_reconstr_true.shape[1], 50, n_dim=1)
            #my_test_plot = plot.plot(nrows=1, ncols=1)
            #my_test_plot.plot(np.arange(h.win.shape[0]), h.win, params="r-")
            #my_test_plot.save(f"{dir_name}/hanning.png")
            #my_test_plot.close()

            my_test_plot = plot.plot(nrows=1, ncols=1, size=plot.default_size(500, 200))
            my_test_plot.rescale_axis_utits_font_size(.5)
            
            
            #avg1 = np.mean(full_reconstr_true, axis=0)
            #avg2 = np.mean(full_reconstr, axis=0)
            #avg3 = np.mean(full_D, axis=0)
            #avg = h.multiply(np.array([avg1, avg2, avg3]), axis=1)
            #avg = np.array([avg1, avg2, avg3])
            spec = fft.fft2(np.array([full_obj, full_reconstr_true, full_reconstr, full_D]))
            #spec = fft.fft(avg)
            spec = np.real(spec*np.conj(spec))
            spec = np.mean(spec, axis=1)
            spec = spec/np.max(spec, axis=1, keepdims=True)
            freqs = fft.fftfreq(n=spec.shape[1], d=1.)
            spec = spec[:, :spec.shape[1]//2]
            freqs = freqs[:freqs.shape[0]//2]
            #spec = misc.normalize(spec, axis=1)
            #spec = fft.fftshift(spec)
            #freqs = fft.fftshift(freqs)
            
            #my_test_plot.plot(freqs, spec[0], params="k:")
            my_test_plot.plot(freqs, spec[1], params="r-")
            my_test_plot.plot(freqs, spec[2], params="g-")
            my_test_plot.plot(freqs, spec[3], params="b-")
            my_test_plot.set_log()
            my_test_plot.set_axis_limits(limits = [[0, np.max(freqs)], [1e-9, 1]])
            my_test_plot.legend(legends=["MOMFBD", "NN", "Raw"])
            #my_test_plot.legend(legends=["MOMFBD 1", "MOMFBD 2", "NN", "Raw"])

            my_test_plot.save(f"{self.dir_name}/spec.png")
            my_test_plot.close()
            
