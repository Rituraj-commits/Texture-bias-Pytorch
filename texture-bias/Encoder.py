__authors__ = "Rituraj Dutta, Abdur R. Fayjie"
__emails__ = "riturajdutta400@gmailcom, fayjie92@gmail.com"


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.pyplot import imread
import numpy as np

from skimage import transform
import os
import tarfile

import tensorflow as tf
import sys

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    # Not Python 3 - today, it is most likely to be Python 2
    # But note that this might need an update when Python 4
    # might be around one day
    from urllib import urlretrieve




class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        # conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding='same')
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding='same')

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding='same')

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding='same')
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding='same')
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding='same')

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding='same')
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding='same')
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding='same')

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding='same')
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding='same')
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding='same')

        # max pooling (kernel_size, stride)
        self.pool = nn.MaxPool2d(2, 2)

       

    def forward(self, x, training=True):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x3 = F.relu(self.conv3_3(x))
       
        x = F.relu(self.conv4_1(x3))
        x = F.relu(self.conv4_2(x))
        x4 = F.relu(self.conv4_3(x))
       
        x = F.relu(self.conv5_1(x4))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))

        x = torch.cat([torch.cat([x3,x4],dim=1),x],dim=1)
  
        return x

 



def vgg_encoder():
    
    # Download weights
    if not os.path.isdir('weights'):
        os.makedirs('weights')
    if not os.path.isfile('weights/vgg_16.ckpt'):
        print('Downloading the checkpoint ...')
        urlretrieve("http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz", "weights/vgg_16_2016_08_28.tar.gz")
        with tarfile.open('weights/vgg_16_2016_08_28.tar.gz', "r:gz") as tar:
            tar.extractall('weights/')
        os.remove('weights/vgg_16_2016_08_28.tar.gz')
        print('Download is complete !')

    reader = tf.train.load_checkpoint('weights/vgg_16.ckpt')
    debug_string = reader.debug_string()

    vgg16 = VGG16()

    # load the weights from the ckpt file (TensorFlow format)
    load_dic = {}
    for l in list(vgg16.state_dict()):
        if 'conv' in l:
            tensor_to_load = 'vgg_16/conv{}/{}/{}{}'.format(l[4], l[:7], l[8:], 's' if 'weight' in l else 'es')
            v_tensor = reader.get_tensor(tensor_to_load)
            if 'weight' in l:
                v_tensor = np.transpose(v_tensor, (3, 2, 1, 0))
            else:
                v_tensor = np.transpose(v_tensor)
            load_dic[l] = torch.from_numpy(v_tensor).float()
        if 'fc' in l:
            tensor_to_load = 'vgg_16/fc{}/{}{}'.format(l[2], l[4:], 's' if 'weight' in l else 'es')
            v_tensor = reader.get_tensor(tensor_to_load)
            if 'weight' in l:
                v_tensor = np.transpose(v_tensor, (3, 2, 1, 0))
            else:
                v_tensor = np.transpose(v_tensor)
            load_dic[l] = torch.from_numpy(v_tensor).float()

    vgg16.load_state_dict(load_dic)
    
    return  vgg16


