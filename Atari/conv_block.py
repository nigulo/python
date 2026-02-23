import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.init as init

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bn=False, activation=True):
        super(ConvBlock, self).__init__()

        self.use_bn = bn
        self.use_activation = activation

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)
        nn.init.kaiming_normal_(self.conv.weight)

        self.padding = nn.ZeroPad2d(get_padding(kernel_size))

        if bn:
            self.bn = nn.BatchNorm2d(in_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_bn:
            x = self.bn(x)
            x = self.relu(x)
            x = self.padding(x)
            x = self.conv(x)
        else:
            x = self.padding(x)
            x = self.conv(x)
            if self.use_activation:
                x = self.relu(x)
        return x

def get_padding(kernel):
    if type(kernel) is int:
        x_pad = kernel//2
        y_pad = x_pad
    else:
        x_pad = kernel[1]//2
        y_pad = kernel[0]//2
    return x_pad, x_pad, y_pad, y_pad
    