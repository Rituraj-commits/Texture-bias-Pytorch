#! /usr/bin/env python
""" 
script for training DOGLSTM 
"""
__authors__ = "Rituraj Dutta, Abdur R. Fayjie"
__emails__ = "fayjie92@gmail.com"


#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt
import utils as U
import numpy as np
from arguments import get_parser
import pickle
import cv2
from Encoder import get_encoder
import torch
import torch.nn as nn

## Get options
options = get_parser().parse_args()
t_l_path   = 'fss_test_set.txt'
Best_performance = 0.00
Valid_miou = []

# Parameter configuration
GPU = options.gpu                      ## default is gpu:0
CUDA = options.use_cuda                ## default is True
LEARNING_RATE = options.learning_rate  ## default is 0.0001
ITERATIONS = options.iterations        ## default is 1000
EPOCHS = options.epochs                ## default is 50

Train_list, Test_list = U.Get_tr_te_lists(options, t_l_path)

## DOG parameters
kernet_shapes = [3, 5, 7, 9]
k_value = np.power(2, 1/3)
sigma   = 1.6


class Lambda(nn.Module):
    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)


def get_kernel_gussian(kernel_size, Sigma=1, in_channels = 320):
    kernel_weights = cv2.getGaussianKernel(ksize=kernel_size, sigma= Sigma)
    kernel_weights = kernel_weights * kernel_weights.T
    kernel_weights = np.expand_dims(kernel_weights, axis=-1)
    kernel_weights = np.repeat(kernel_weights, in_channels, axis=-1)
    kernel_weights = np.expand_dims(kernel_weights, axis=-1)
    return kernel_weights


def GlobalAveragePooling2D_r(f):
    def func(x):
        repc = int(x.shape[1])
        m = torch.repeat_interleave(f,repc,dim=1)
        x = torch.mul(x,m)
        repx = int(x.shape[2])
        repy = int(x.shape[3])
        x = torch.sum(x,dim=[2,3],keepdim=True)/torch.sum(m,dim=[2,3],keepdim=True)
        x = torch.repeat_interleave(x,repx,dim=2)
        x = torch.repeat_interleave(x,repy,dim=3)
        return x
    return Lambda(func)  ##


def Rep_mask(f):
    def func(x):
        x = torch.repeat_interleave(x,f,dim=1)
        return x
    return Lambda(func)  ##


def common_representation(x1,x2):
### Not Implemented #####
    return


### time distributed layer
# Time distribution is in sequencial order 
class TimeDistributed(nn.Module):
    """
    Applies a module over tdim identically for each step.
    + Low memory condition.
    """
    def __init__(self, module, low_mem=False, tdim=1):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.low_mem = low_mem
        self.tdim = tdim
        

    def forward(self, *args, **kwargs):
        """
        Args: 
            - Input X : [bs, seq_len , C, W, H]
        Return:
            - Output X : [bs, seq_len , C, W, H]
        """
        if self.low_mem or self.tdim!=1: 
            return self.low_mem_forward(*args)
        else:
            #only support tdim=1
            inp_shape = args[0].shape
            bs, seq_len = inp_shape[0], inp_shape[1]   
            out = self.module(*[x.view(bs*seq_len, *x.shape[2:]) for x in args], **kwargs)
            out_shape = out.shape
            return out.view(bs, seq_len,*out_shape[1:])
    

    def low_mem_forward(self, *args, **kwargs):                                           
         """
        Args: 
            - Input X : [bs, seq_len , C, W, H]
        Return:
            - Output X : [bs, seq_len , C, W, H]
        """
        tlen = args[0].shape[self.tdim]
        args_split = [torch.unbind(x, dim=self.tdim) for x in args]
        out = []
        for i in range(tlen):
            out.append(self.module(*[args[i] for args in args_split]), **kwargs)
        return torch.stack(out,dim=self.tdim)


    def __repr__(self):
        """
        Return:
            - Time distribution over the input module
        """
        return f'TimeDistributed({self.module})'


class Texture:
    def __init__(self,s_encoded,q_encoded,s_mask):
        super(Texture,self).__init__()

        self.Sigma1_kernel = get_kernel_gussian(kernel_size = kernet_shapes[0],\
        Sigma = sigma*np.power(k_value, 1), in_channels = 320)
        self.Sigma2_kernel = get_kernel_gussian(kernel_size = kernet_shapes[1],\
        Sigma = sigma*np.power(k_value, 2), in_channels = 320)    
        self.Sigma3_kernel = get_kernel_gussian(kernel_size = kernet_shapes[2],\
        Sigma = sigma*np.power(k_value, 3), in_channels = 320)     
        self.Sigma4_kernel = get_kernel_gussian(kernel_size = kernet_shapes[3],\
        Sigma = sigma*np.power(k_value, 4), in_channels = 320)    

        ## Depthwise Convolution
        self.Sigma1_layer = nn.Conv2d(in_channels=320,out_channels=320,\
        kernel_size=kernet_shapes[0],groups=320,padding='same',bias=False)
        self.Sigma2_layer = nn.Conv2d(in_channels=320,out_channels=320,\
        kernel_size=kernet_shapes[1],groups=320,padding='same',bias=False)
        self.Sigma3_layer = nn.Conv2d(in_channels=320,out_channels=320,\
        kernel_size=kernet_shapes[2],groups=320,padding='same',bias=False)
        self.Sigma4_layer = nn.Conv2d(in_channels=320,out_channels=320,\
        kernel_size=kernet_shapes[3],groups=320,padding='same',bias=False)

        self.GlobalAveragePooling = GlobalAveragePooling2D_r(s_mask)

        ### Not Implemented #####


def train():
    """
    training part for the model
    """    
    encoder = get_encoder('efficientnet-b1',in_channels=3,weights=None)

    encoder.cuda(GPU)
    encoder_optim = torch.optim.Adam(encoder.parameters(),lr=LEARNING_RATE)

    for ep in range(EPOCHS):
        epoch_loss = 0.00

        for idx in range(ITERATIONS):
            samples, sample_labels, batches, batch_labels = U.get_episode(options,Test_list)

            S_input = encoder(samples).cuda(GPU)       ## out_channel = 320
            Q_input = encoder(batches).cuda(GPU)       ## out_channel = 320
            S_mask = encoder(sample_labels).cuda(GPU)  ## out_channel = 320

          

if __name__=="__main__":
    # train()
    print('--- Variable Tensor Time Distribution Layer ---\n')
    tdconv = TimeDistributed(nn.Conv2d(2, 5, 3, 1, 1), tdim=1)
    x = tdconv(torch.rand(3, 10, 2, 8, 8))
    print(x.shape)
    
            
         
     



