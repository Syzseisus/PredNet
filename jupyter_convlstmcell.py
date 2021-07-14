import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Gates = nn.Conv2d(in_channels + out_channels, 4*out_channels, 3, padding=1)
    
    def forward(self, x, prev_state): 
        # get batch and spatial sizes
        batch_size = x.data.size()[0]
        spatial_size = x.data.size()[2:]
        
        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.out_channels] + list(spatial_size)
            prev_state = (Variable(torch.zeros(state_size)).cuda(),
                          Variable(torch.zeros(state_size)).cuda())

        h0, c0 = prev_state
        if torch.cuda.is_available():
            if h0.is_cuda == False: h0 = h0.cuda()
            if c0.is_cuda == False: c0 = c0.cuda()
        
        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((x, h0), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        i, f, o, c = gates.chunk(4, 1)

        # apply non linearity
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(c)
        o = torch.sigmoid(o)
        
        c1 = (f * c0) + (i * g)
        h1 = o * torch.tanh(c1)
        
        return (h1, c1)
