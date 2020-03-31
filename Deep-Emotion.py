'''
[categories] : (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
The following Code contains
1- Dataloader for the network with train_val split
2-class of the model layers
3-train and validation loop
4- testing loop and model evaluation
'''
'''
[Note]
If running on Windows and you get a BrokenPipeError, try setting
the num_worker of torch.utils.data.DataLoader() to 0.
'''
import os
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import OneHotEncoder
from torchvision import transforms
from data_loaders import Plain_Dataset, eval_data_dataloader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
Global variables
'''
traincsv_file = 'Dataset/Kaggle/train.csv'
validationcsv_file = 'Dataset/Kaggle/val.csv'
testcsv_file = 'Dataset/Kaggle/test.csv'

train_img_dir = 'Train/'
validation_img_dir = 'validation/'
test_img_dir = 'test/'

epochs = 100
lr = 0.005
batchsize = 128

class Deep_Emotion(nn.Module):
    def __init__(self):
        '''
        Documentation
        '''
        super(Deep_Emotion,self).__init__()
        self.conv1 = nn.Conv2d(1,10,3)
        self.conv2 = nn.Conv2d(10,10,3)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(10,10,3)
        self.conv4 = nn.Conv2d(10,10,3)
        self.pool4 = nn.MaxPool2d(2,2)

        self.norm = nn.BatchNorm2d(10)
        #self.dropout = nn.Dropout2d()

        self.fc1 = nn.Linear(810,50)
        self.fc2 = nn.Linear(50,7)

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 640)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self,input):
        out = self.stn(input)

        out = F.relu(self.conv1(out))
        out = self.conv2(out)
        out = F.relu(self.pool2(out))

        out = F.relu(self.conv3(out))
        out = self.norm(self.conv4(out))
        out = F.relu(self.pool4(out))

        out = F.dropout(out)
        out = out.view(-1, 810) #####
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out

def Train():
    '''
    Documentation
    '''
    net = Deep_Emotion()
    net.to(device)
    print("===================================Start Training===================================")

    print("Model archticture: ", net)
    transformation = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

    train_dataset =      Plain_Dataset(csv_file=traincsv_file,img_dir = train_img_dir,datatype = 'train',transform = transformation)
    validation_dataset = Plain_Dataset(csv_file=validationcsv_file,img_dir = validation_img_dir,datatype = 'val',transform = transformation)

    train_loader = DataLoader(train_dataset,batch_size=batchsize,num_workers=0)
    val_loader =   DataLoader(validation_dataset,batch_size=batchsize,num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optmizer = optim.Adam(net.parameters(), lr = lr)

    for e in range(epochs):
        train_loss = 0
        val_loss = 0
        train_acc_epoch = []
        val_acc_epoch = []
        # Train the model  #
        net.train()
        for data, lables in train_loader:
            data, lables = data.to(device), lables.to(device)
            #lables = torch.max(lables, 1)[1] if you used onehot encoding
            optmizer.zero_grad()

            outputs = net(data)
            #calculate the accuarcy
            t_prediction = F.softmax(outputs,dim=1)
            t_classes = torch.argmax(t_prediction,dim=1)
            t_wrong = torch.where(t_classes != lables, torch.tensor([1.]).cuda(),torch.tensor([0.]).cuda())
            t_acc = 1- torch.sum(t_wrong) / batchsize

            train_acc_epoch.append(t_acc.item())
            #
            loss = criterion(outputs,lables)
            loss.backward()
            optmizer.step()
            train_loss += loss.item() * data.size(0)

        # validate the model #
        net.eval()
        for data, lables in val_loader:
            #
            data, lables = data.to(device), lables.to(device)
            #lables = torch.max(lables, 1)[1] if you used onehot encoding
            outputs = net(data)
            #calculate the accuarcy
            v_prediction = F.softmax(outputs,dim=1)
            v_classes = torch.argmax(v_prediction,dim=1)
            v_wrong = torch.where(v_classes != lables, torch.tensor([1.]).cuda(),torch.tensor([0.]).cuda())
            v_acc = 1- torch.sum(v_wrong) / batchsize

            val_acc_epoch.append(v_acc.item())
            #

            loss = criterion(outputs, lables)

            val_loss += loss.item() * data.size(0)

        train_loss = train_loss/len(train_loader.sampler)
        val_loss = val_loss/len(val_loader.sampler)
        print('Epoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tTraining Acuuarcy {:.3f}% \tValidation Acuuarcy {:.3f}%'
                                                            .format(e+1, train_loss,val_loss,np.mean(train_acc_epoch)*100,np.mean(val_acc_epoch)*100))

    torch.save(net,'model_noSTN-{}-{}-{}.pt'.format(epochs,batchsize,lr))
    print("===================================Training Finished===================================")


Train()
