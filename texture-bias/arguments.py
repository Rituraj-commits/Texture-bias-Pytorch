__authors__ = "Rituraj Dutta, Abdur R. Fayjie"
__emails__ = "riturajdutta400@gmailcom, fayjie92@gmail.com"

import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-msp","--ModelSavePath",
                        type=str,
                        default='vgg16_5way_1shot')

    parser.add_argument('-img_p', '--data_path',
                        type=str,
                        help='path to image sources',
                        default = 'fewshot/fss1000/')

    parser.add_argument('-nw', '--nway',
                        type=int,
                        help='number of class in each episode of train, default=5',
                        default = 5)
                        
    parser.add_argument('-ks', '--kshot',
                        type=int,
                        help='number of sample for each class, default=1',
                        default = 1)

    parser.add_argument('-csample', '--class_samples',
                        type=int,
                        help='number of sample inside each class, default=10',
                        default=10)
                                                
    parser.add_argument('-ih', '--img_h',
                        type=int,
                        help='Image high, default=192',
                        default=224)
                        
    parser.add_argument('-iw', '--img_w',
                        type=int,
                        help='Image width, default=96',
                        default=224)

    parser.add_argument('-nep', '--epochs',
                        type=int,
                        help='number of epochs to train , default= 100',
                        default=50)

    parser.add_argument('-its', '--iterations',
                        type=int,
                        help='number of episodes per epoch, default=500',
                        default=1000)

    parser.add_argument('-ittests', '--it_test',
                        type=int,
                        help='number of episodes for evaluating test performance, default=1000',
                        default=100)                       

    parser.add_argument('-itval', '--it_eval',
                        type=int,
                        help='number of episodes for validation performance, default=100',
                        default=200)                                             
                                                
    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.0001',
                        default=0.0001)
                        
    parser.add_argument('-lm', '--learning_mode',
                        type=str,
                        help='learning rate for the model, default= only_train',
                        default='only_train') # can be one of only_train, evaluate or train_evaluate

    parser.add_argument("-g","--gpu",type=int, default=0)
    
    parser.add_argument("--use_cuda", type = bool, default = True)


    return parser
