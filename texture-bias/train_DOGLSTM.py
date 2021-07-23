#! /usr/bin/env python
""" 
script for training DOGLSTM 
"""
__authors__ = "Rituraj Dutta, Abdur R. Fayjie"
__emails__ = "riturajdutta400@gmailcom, fayjie92@gmail.com"


import os
import re
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt
from numpy.lib.function_base import select
import utils as U
import numpy as np
from torch.autograd import Variable
from arguments import get_parser
import pickle
import cv2
from Encoder import *
import torch
import torch.nn as nn
from BDCLSTM import ConvLSTM

## Get options
options = get_parser().parse_args()
t_l_path   = 'fss_test_set.txt'
Best_performance = 0.00
Valid_miou = []

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

def get_kernel_gussian(kernel_size, Sigma=1, in_channels = 320):
    kernel_weights = cv2.getGaussianKernel(ksize=kernel_size, sigma= Sigma)
    kernel_weights = kernel_weights * kernel_weights.T
    kernel_weights = np.expand_dims(kernel_weights, axis=-1)
    kernel_weights = np.repeat(kernel_weights, in_channels, axis=-1)
    kernel_weights = np.expand_dims(kernel_weights, axis=-1)
    return kernel_weights

## Kernel weights for Gussian pyramid

Sigma1_kernel = get_kernel_gussian(kernel_size = kernet_shapes[0], Sigma = sigma*np.power(k_value, 1), in_channels = 128)
Sigma2_kernel = get_kernel_gussian(kernel_size = kernet_shapes[1], Sigma = sigma*np.power(k_value, 2), in_channels = 128)    
Sigma3_kernel = get_kernel_gussian(kernel_size = kernet_shapes[2], Sigma = sigma*np.power(k_value, 3), in_channels = 128)     
Sigma4_kernel = get_kernel_gussian(kernel_size = kernet_shapes[3], Sigma = sigma*np.power(k_value, 4), in_channels = 128)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class TimeDistributed(nn.Module):
    "Applies a module over tdim identically for each step" 
    def __init__(self, module, low_mem=False, tdim=1):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.low_mem = low_mem
        self.tdim = tdim
        
    def forward(self, *args, **kwargs):
        "input x with shape:(bs,seq_len,channels,width,height)"
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
        "input x with shape:(bs,seq_len,channels,width,height)"
        tlen = args[0].shape[self.tdim]
        args_split = [torch.unbind(x, dim=self.tdim) for x in args]
        out = []
        for i in range(tlen):
            out.append(self.module(*[args[i] for args in args_split]), **kwargs)
        return torch.stack(out,dim=self.tdim)
    def __repr__(self):
        return f'TimeDistributed({self.module})'



def GlobalAveragePooling2D_r(f):
    def func(x):
        repc = int(x.shape[1])
        m = torch.repeat_interleave(f,repc,dim=1)
        x = torch.mul(x,m)
        repx = int(x.shape[3])
        repy = int(x.shape[4])
        x = torch.sum(x,dim=[3,4],keepdim=True)/torch.sum(m,dim=[3,4],keepdim=True)
        x = torch.repeat_interleave(x,repx,dim=3)
        x = torch.repeat_interleave(x,repy,dim=4)
        return x
    return LambdaLayer(func)

def Rep_mask(f):
    def func(x):
        x = torch.repeat_interleave(x,f,dim=1)
        return x
    return LambdaLayer(func)

class common_representation(nn.Module):

    def __init__(self):
        super().__init__()

        self.Conv2dReLU = nn.Sequential(
            TimeDistributed(nn.Conv2d(in_channels=256,out_channels=128,kernel_size=(3,3),padding='same'),tdim=1),
            TimeDistributed(nn.BatchNorm2d(128),tdim=1),
            TimeDistributed(nn.ReLU(inplace=False),tdim=1)
        )
       

    def forward(self,x1,x2):
        repc = int(x1.shape[1])
        x2 = torch.reshape(x2,(5,1,128,56,56))
        x2 = Rep_mask(repc)(x2)
        x = torch.cat((x1,x2),dim=2)
        x = self.Conv2dReLU(x)

        return x

       
class Texture(nn.Module):
    def __init__(self):
        super(Texture,self).__init__()


        ## Depthwise Convolution

        self.Sigma1_layer = TimeDistributed(nn.Conv2d(in_channels=128,out_channels=128,kernel_size=kernet_shapes[0],groups=128,padding='same',bias=False),tdim=1)
        self.Sigma2_layer = TimeDistributed(nn.Conv2d(in_channels=128,out_channels=128,kernel_size=kernet_shapes[1],groups=128,padding='same',bias=False),tdim=1)
        self.Sigma3_layer = TimeDistributed(nn.Conv2d(in_channels=128,out_channels=128,kernel_size=kernet_shapes[2],groups=128,padding='same',bias=False),tdim=1)
        self.Sigma4_layer = TimeDistributed(nn.Conv2d(in_channels=128,out_channels=128,kernel_size=kernet_shapes[3],groups=128,padding='same',bias=False),tdim=1)

        ## Manually set Gaussian Weights and make them non-trainable

        with torch.no_grad():
            self.Sigma1_layer.weight = nn.parameter.Parameter(Sigma1_kernel,requires_grad=False)
            self.Sigma2_layer.weight = nn.parameter.Parameter(Sigma2_kernel,requires_grad=False)
            self.Sigma3_layer.weight = nn.parameter.Parameter(Sigma3_kernel,requires_grad=False)
            self.Sigma4_layer.weight = nn.parameter.Parameter(Sigma4_kernel,requires_grad=False)


        self.CommonRepresentation = common_representation()

        self.ConvLSTM2D = ConvLSTM(img_size=(56,56),input_dim=128,hidden_dim=128,kernel_size=(3,3),cnn_dropout=0.2,rnn_dropout=0.2,batch_first=True,bidirectional=True)


    def forward(self,s_encoded,s_mask,q_encoded):

        x1 = self.Sigma1_layer(s_encoded)
        x2 = self.Sigma1_layer(s_encoded)
        x3 = self.Sigma1_layer(s_encoded)
        x4 = self.Sigma1_layer(s_encoded)

        DOG1 = torch.subtract(s_encoded,x1)
        DOG2 = torch.subtract(x1,x2)
        DOG3 = torch.subtract(x2,x3)
        DOG4 = torch.subtract(x3,x4)

        s1 = GlobalAveragePooling2D_r(s_mask)(DOG1)
        s2 = GlobalAveragePooling2D_r(s_mask)(DOG2)
        s3 = GlobalAveragePooling2D_r(s_mask)(DOG3)
        s4 = GlobalAveragePooling2D_r(s_mask)(DOG4)

        s_1  = self.CommonRepresentation(s1, q_encoded)    
        s_2  = self.CommonRepresentation(s2, q_encoded)    
        s_3  = self.CommonRepresentation(s3, q_encoded)    
        s_4  = self.CommonRepresentation(s4, q_encoded)    

        s_3D = torch.cat(((torch.cat((torch.cat((s1,s2),dim=1),s3),dim=1)),s4),dim=1)  ## concatenate (s1,s2,s3,s4)

        ## Multi scale-space representations are fed into the Bi-Directional ConvLSTM

        x = self.ConvLSTM2D(s_3D)

        return x



class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.Conv2dReLU = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3,3),padding='same'),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=False)
        )

        self.UpSampling2D = nn.Upsample(scale_factor=(2,2),mode='bilinear',align_corners=True)

        self.final = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=(3,3),padding='same'),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64,out_channels=32,padding='same',kernel_size=(3,3)),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32,out_channels=1,padding='same',kernel_size=(3,3)),
            nn.Sigmoid()
        )
        


           

    def forward(self,x):

        x = self.Conv2dReLU(x)
        x = self.UpSampling2D(x)

        x = self.Conv2dReLU(x)
        x = self.UpSampling2D(x)

        x = self.Conv2dReLU(x)

        x = self.final(x)

        return x


def train():

    
    print("Training Started........")
    encoder = vgg_encoder()

    encoder = nn.Sequential(
    encoder,
    nn.Conv2d(in_channels=1280,out_channels=128,padding='same',kernel_size=(3,3)),
    nn.ReLU(inplace=False),
    )

    layers = []
    layers.append(TimeDistributed(encoder,tdim=1))
    kshot_encoder = torch.nn.Sequential(*layers)

    decoder = Decoder()
    texture_model = Texture()
   

    encoder.cuda(GPU)
    kshot_encoder.cuda(GPU)
    texture_model.cuda(GPU)
    decoder.cuda(GPU)
   

    
    encoder_optim = torch.optim.Adam(encoder.parameters(),lr=LEARNING_RATE)
    kshot_encoder_optim = torch.optim.Adam(kshot_encoder.parameters(),lr=LEARNING_RATE)
    texture_optim = torch.optim.Adam(texture_model.parameters(),lr=LEARNING_RATE)

    decoder_optim = torch.optim.Adam(decoder.parameters(),lr=LEARNING_RATE)


    for ep in range(EPOCHS):
        epoch_loss = 0.00

        for idx in range(ITERATIONS):

            encoder_optim.zero_grad()
            kshot_encoder_optim.zero_grad()
            texture_optim.zero_grad()
            decoder_optim.zero_grad()


            samples, sample_labels, batches, batch_labels = U.get_episode(options,Test_list)

            S_input = kshot_encoder(Variable(samples).cuda(GPU))      
            Q_input = encoder(Variable(batches).cuda(GPU))

            texture = texture_model(S_input,Variable(sample_labels),Q_input)

            output = decoder(texture)

            bce = nn.BCELoss().cuda(GPU)
            loss = bce(output,Variable(batch_labels).cuda(GPU))

            loss.backward()

            encoder_optim.step()
            kshot_encoder_optim.step()
            texture_optim.step()
            decoder_optim.step()

            epoch_loss += loss.cpu().data.numpy()


            if (idx % 50) == 0:
                print ('Epoch>>',(ep+1),'>> Itteration:', (idx+1),'/',options.iterations,' >>> Loss:', epoch_loss/(idx+1))
            
          
          

   



if __name__ == "main":

    train()
    
            
         
     



