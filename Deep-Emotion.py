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
from torchvision import transforms

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
        self.ohe = OneHotEncoder() #one hot encoder object
        self.train_csv = pd.read_csv(csv_file)
        self.hot_lables = self.ohe.fit_transform(self.train_csv[['emotion']]).toarray()
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.train_csv)
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = Image.open(self.img_dir+'train'+str(idx)+'.jpg')
        lables = self.hot_lables[idx]
        lables = torch.from_numpy(lables).float()
        if self.transform :
            img = self.transform(img)
        return img,lables

class Test_dataset(Dataset):
    def __init__(self,csv_file,img_dir,transform):
        '''
        Documentation
        '''
        self.ohe = OneHotEncoder() #one hot encoder object
        self.train_csv = pd.read_csv(csv_file)
        self.hot_lables = self.ohe.fit_transform(self.train_csv[['emotion']]).toarray()
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.train_csv)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(self.img_dir+'train'+str(idx)+'.jpg')
        lables = self.hot_lables[idx]
        lables = torch.from_numpy(lables).float()


        if self.transform :
            img = self.transform(img)


        return img,lables

def train_val_split(train_dataset,val_size= 0.25):
    '''
    Documentation
    '''
    print("===============================Train Validation Split===============================")
    data_size = len(train_dataset)
    print("data_size: ",data_size)

    indices = list(range(data_size))

    split_ammount = int(np.floor(val_size * data_size))
    print("number of val_set: ",split_ammount)

    np.random.seed(42)
    np.random.shuffle(indices)
    #print("shuffled training set: ",indices)

    train_indices, val_indices = indices[split_ammount:], indices[:split_ammount]

    print('number of training_indices: ',len(train_indices))
    print('number of training_indices: ',len(val_indices))

    print("========================================================")
    #print('training_indices: ',train_indices)
    #print('validation_indices: ',val_indices)
    return train_indices, val_indices





def eval_train_dataloader(validation = True):
    '''
    Documentation
    '''

    transformation = transforms.Compose([transforms.ToTensor()])

    dataset = Train_dataset(traincsv_file,'Train/',transformation)

    imgg = dataset.__getitem__(5)[0]
    lable = dataset.__getitem__(5)[1]

    print(lable)

    imgnumpy = imgg.numpy()
    imgt = imgnumpy.squeeze()
    plt.imshow(imgt)
    plt.show()
    #press Q to apply train_val split function
    if validation :
        train_val_split(dataset,0.2)


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
