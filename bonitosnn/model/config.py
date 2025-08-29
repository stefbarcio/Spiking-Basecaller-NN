"""Configuration file of a Bonito-CRF model
"""
import os
import sys
from model import BonitoModel as Model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from classes import BaseNanoporeDataset
import torch
from torch import nn
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type = 'crf' # ctc or crf
model_name = '500_cnv3_rnn5' + '_' + model_type
output_dir = '/hpc/compgen/projects/nanoxog/babe/analysis/mpages/models/bonito'

## TRAIN CONFIGURATION #############################################
num_epochs = 3
train_batch_size = 256
validation_batch_size = 256
validate_every = 100
checkpoint_every = 5000


##       DATA.         #############################################
data_dir = '/hpc/compgen/projects/nanoxog/babe/analysis/mpages/train_input/inter/500.0'
encoding_dict = {'A': 1 , 'C':  2 , 'G':  3 , 'T':  4 , '':  0}
decoding_dict = { 1 :'A',  2 : 'C',  3 : 'G',  4 : 'T', 0 : ''}
dataset = BaseNanoporeDataset(data_dir = data_dir, 
                              decoding_dict = decoding_dict, 
                              encoding_dict = encoding_dict, 
                              split = 0.95, 
                              shuffle = True, 
                              seed = 1)

dataloader_train = DataLoader(dataset, batch_size = train_batch_size, 
                              sampler = dataset.train_sampler, num_workers = 1)
dataloader_validation = DataLoader(dataset, batch_size = validation_batch_size, 
                                   sampler = dataset.validation_sampler, num_workers = 1)


##   MODEL PART1        #############################################
model = Model(device = device,
              dataloader_train = dataloader_train, 
              dataloader_validation = dataloader_validation, 
              model_type = model_type)


##    OPTIMIZATION     #############################################
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
schedulers = {'lr_scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, (len(dataset.train_idxs)*num_epochs)/train_batch_size, 
                                                                         eta_min=0.00001, last_epoch=-1, verbose=False),
             }
clipping_value = 2
use_sam = False


##   MODEL PART2        #############################################
model.optimizer = optimizer
model.schedulers = schedulers
model.clipping_value = clipping_value
model.use_sam = False

output_dir = os.path.join(output_dir, data_dir.split('/')[-2], model_name)