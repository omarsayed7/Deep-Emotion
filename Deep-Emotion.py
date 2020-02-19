'''
The following Code contains
1- Dataloader for the network with train_val split
2-class of the model layers
3-train and validation loop
4- testing loop and model evaluation
'''
import os
import numpy  as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import MultiLabelBinarizer

'''
Global variables
'''
validation_spli = 0.2
traincsv_file = 'Dataset/Kaggle/train.csv'
testcsv_file = 'Dataset/Kaggle/test.csv'
train_img_dir = 'Train/'
test_img_dir = 'test/'

class Train_dataset(Dataset):
    def __init__(self,csv_file,img_dir,transform):
        '''
        Documentation
        '''
        self.mlb = MultiLabelBinarizer() #one hot encoder object


    def __len__(self):
        return len()

    def __getitem__(self,idx):
        pass

class Test_dataset(Dataset):
    def __init__(self,):
        '''
        Documentation
        '''
        pass

    def __len__(self):
        return len()

    def __getitem__(self,idx):
        pass

def eval_train_dataloader(validation_Data = True):
    '''
    Documentation
    '''
    pass

class Deep_Emotion(nn.Module):
    def __init__(self):
        '''
        Documentation
        '''
        super.__init__()
    def forward():
        pass
    def STN():
        pass

def Train():
    '''
    Documentation
    '''
    pass
