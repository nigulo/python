import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.init as init

from conv_block import ConvBlock

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class Model(nn.Module):
    def __init__(self, n_frames, n_actions, nx=52, ny=40, n_channels=[16, 16], kernels=[5, 3], strides=[], pool_sizes=[2, 2], n_dense=[256]):
        super(Model, self).__init__()
        
        self.n_dense = n_dense

        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.dense = nn.ModuleList()
        in_frames = n_frames

        nx_out = nx
        ny_out = ny

        for i, n_chs in enumerate(n_channels):
            kernel = to_pair(kernels[i])
            stride = (1, 1)
            if len(strides) > i:
                stride = to_pair(strides[i])
            self.convs.append(ConvBlock(in_frames, n_chs, kernel_size=kernel, stride=stride))
            nx_out = nx_out//stride[0]
            ny_out = ny_out//stride[1]
            if len(pool_sizes) > i:
                pool_size = to_pair(pool_sizes[i])
                self.pools.append(nn.MaxPool2d(pool_size))
                nx_out = nx_out//pool_size[0]
                ny_out = ny_out//pool_size[1] 
            in_frames = n_channels[i]

        num_features = in_frames*nx_out*ny_out
        for n_d in n_dense:
            self.dense.append(nn.Linear(num_features, n_d))
            num_features = n_d
            nn.init.kaiming_normal_(self.dense[-1].weight)
            nn.init.constant_(self.dense[-1].bias, 0.01)
        
        self.dense.append(nn.Linear(num_features, n_actions))
        nn.init.kaiming_normal_(self.dense[-1].weight)
        nn.init.constant_(self.dense[-1].bias, 0.01)
        
        self.relu = nn.ReLU(inplace=True)


    def weights_init(self):
        pass

    def forward(self, state):
        x = state

        for i, conv in enumerate(self.convs):
            x = conv(x)
            if len(self.pools) > i:
                x = self.pools[i](x)

        x = torch.flatten(x, start_dim=1)
        for i, dense in enumerate(self.dense):
            x = self.relu(dense(x))

        return x

def to_pair(x):
    if type(x) is int:
        return x, x
    return x
    
