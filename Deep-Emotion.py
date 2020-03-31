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

    train_loader = DataLoader(train_dataset,batch_size=batchsize,shuffle = True,num_workers=0)
    val_loader =   DataLoader(validation_dataset,batch_size=batchsize,shuffle = True,num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optmizer = optim.Adam(net.parameters(),lr= lr)

    for e in range(epochs):
        train_loss = 0
        val_loss = 0

        train_correct = 0
        val_correct = 0
        # Train the model  #
        net.train()
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optmizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs,labels)
            loss.backward()
            optmizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs,1)
            train_correct += torch.sum(preds == labels.data)

        net.eval()
        for data,labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            val_outputs = net(data)
            val_loss = criterion(val_outputs, labels)
            val_loss += val_loss.item()

            _, val_preds = torch.max(val_outputs,1)
            val_correct += torch.sum(val_preds == labels.data)

        train_loss = train_loss/len(train_dataset)
        train_acc = train_correct.double() / len(train_dataset)

        val_loss =  val_loss / len(validation_dataset)
        val_acc = val_correct.double() / len(validation_dataset)

        print('Epoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tTraining Acuuarcy {:.3f}% \tValidation Acuuarcy {:.3f}%'
                                                           .format(e+1, train_loss,val_loss,train_acc * 100, val_acc*100))

    torch.save(net.state_dict(),'model_noSTN-{}-{}-{}.pt'.format(epochs,batchsize,0.001))
    print("===================================Training Finished===================================")

if __name__ == '__main__':
    Train()
