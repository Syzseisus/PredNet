import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

from convlstmcell import ConvLSTMCell


class PredNet(nn.Module):
    def __init__(self, R_channels, A_channels, output_mode='error'):
        super(PredNet, self).__init__()
        self.r_channels = R_channels + (0, )  # for convenience iteration of ConvLSTM
        self.a_channels = A_channels
        self.n_layers = len(R_channels)
        self.lstm_channels = tuple([2*self.a_channels[l] + self.r_channels[l+1]
                                   for l in range(self.n_layers)])
        self.output_mode = output_mode
        self.prediction_all = []
#         self.error_all = []
        
        if output_mode == 'out_all':
            self.As = []
            self.Ahats = []
            self.Es = []
            self.Rs = []
        
        default_output_modes = ['prediction', 'error', 'prediction_all', 'error_all', 'out_all']
        assert output_mode in default_output_modes, 'Invalid output_mode: ' + str(output_mode)

        for i in range(self.n_layers):
            cell = ConvLSTMCell(self. lstm_channels[i], self.r_channels[i])
            setattr(self, 'cell{}'.format(i), cell)

        for i in range(self.n_layers):
            conv = nn.Sequential(nn.Conv2d(self.r_channels[i], self.a_channels[i], 3, padding=1), nn.ReLU())
            if i == 0:
                conv.add_module('satlu', SatLU())
            setattr(self, 'conv{}'.format(i), conv)

        self.upsample = nn.Upsample(scale_factor=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.loss_fn = nn.MSELoss().cuda()

        for l in range(self.n_layers - 1):
            update_A = nn.Sequential(nn.Conv2d(2* self.a_channels[l], self.a_channels[l+1], (3, 3), padding=1), self.maxpool)
            setattr(self, 'update_A{}'.format(l), update_A)

    def forward(self, input):
        
        error_all = []

        E_seq = [None] * self.n_layers
        R_seq = [None] * self.n_layers
        C_seq = [None] * self.n_layers
        
        w, h = input.size(-2), input.size(-1)
        batch_size = input.size(0)

        for l in range(self.n_layers):
            E_seq[l] = Variable(torch.zeros(batch_size, 2*self.a_channels[l], w, h)).cuda()
            R_seq[l] = Variable(torch.zeros(batch_size, self.r_channels[l], w, h)).cuda()
            C_seq[l] = Variable(torch.zeros(batch_size, self.r_channels[l], w, h)).cuda()
            w = w//2
            h = h//2
        time_steps = input.size(1)
        total_error = []
        
        for t in range(time_steps):
            A = input[:,t]
            A = A.type(torch.cuda.FloatTensor)
            # add As for out_all
            if self.output_mode == 'out_all':
                self.As.append(A.data.cpu().detach().numpy().tolist())
            
            for l in reversed(range(self.n_layers)):
                cell = getattr(self, 'cell{}'.format(l))
                
                if l == self.n_layers - 1:
                    inputs = E_seq[l]
                else:
                    temp = [E_seq[l], r_up]
                    inputs = torch.cat(temp,-3)
                _r, _c = cell(inputs, (R_seq[l], C_seq[l]))
                R_seq[l] = _r
                C_seq[l] = _c
                
                # add Rs for out_all
                if self.output_mode == 'out_all':
                    self.Rs.append(_r.data.cpu().detach().numpy().tolist())
                
                if l > 0:
                    r_up = self.upsample(_r)
            


            for l in range(self.n_layers):
                conv = getattr(self, 'conv{}'.format(l))
                A_hat = conv(R_seq[l])
                if l == 0:
                    frame_prediction = A_hat
                if self.output_mode == 'prediction_all':
                    self.prediction_all.append(A_hat)
                # add Ahats for out_all
                if self.output_mode == 'out_all':
                    self.Ahats.append(A_hat.data.cpu().detach().numpy().tolist())

                pos = F.relu(A_hat - A)
                neg = F.relu(A - A_hat)
                E = torch.cat([pos, neg],1)
                E_seq[l] = E
                # add Es for out_all
                if self.output_mode == 'out_all':
                    self.Es.append(E.data.cpu().detach().numpy().tolist())
                
                if l < self.n_layers - 1:
                    update_A = getattr(self, 'update_A{}'.format(l))
                    A = update_A(E)
                    # add As for out_all
                    if self.output_mode == 'out_all':
                        self.As.append(A.data.cpu().detach().numpy().tolist())
                         
                
                
            if self.output_mode == 'error':
                mean_error = torch.cat([torch.mean(e.view(e.size(0), -1), 1, keepdim=True)
                                        for e in E_seq], 1)
                # batch x n_layers
                total_error.append(mean_error)
            if self.output_mode == 'error_all':
                for e in E_seq[0]:
#                     self.error_all.append(e)
                    error_all.append(e)
    
    
        with torch.no_grad():
            torch.cuda.empty_cache()
            

        if self.output_mode == 'error':
            return torch.stack(total_error, 2) # batch x n_layers x nt
        elif self.output_mode == 'prediction':
            return frame_prediction
        elif self.output_mode == 'prediction_all':
            return self.prediction_all
        elif self.output_mode == 'error_all':
#             print("shape: ", len(self.error_all), self.error_all[0].shape)
#             return torch.cat(self.error_all,-3)
#             print("shape: ", len(error_all), error_all[0].shape)
#             return torch.cat(error_all,-3)
            errors = torch.cat(error_all,-3)
            targets = Variable(torch.zeros(errors.shape)).cuda()
            return self.loss_fn(errors, targets)
        elif self.output_mode == 'out_all':
            return self.As, self.Ahats, self.Es, self.Rs


class SatLU(nn.Module):

    def __init__(self, lower=0, upper=255, inplace=False):
        super(SatLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input):
        return F.hardtanh(input, self.lower, self.upper, self.inplace)


    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' ('\
            + 'min_val=' + str(self.lower) \
	        + ', max_val=' + str(self.upper) \
	        + inplace_str + ')'
