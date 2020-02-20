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
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
Global variables
'''
validation_spli = 0.2
traincsv_file = 'Dataset/Kaggle/train.csv'
testcsv_file = 'Dataset/Kaggle/test.csv'
train_img_dir = 'Train/'
test_img_dir = 'test/'
epochs = 10
lr = 0.0001


class Plain_Dataset(Dataset):
    def __init__(self,csv_file,img_dir,datatype,transform):
        '''
        Documentation
        '''
        self.ohe = OneHotEncoder() #one hot encoder object
        self.train_csv = pd.read_csv(csv_file)
        self.hot_lables = self.ohe.fit_transform(self.train_csv[['emotion']]).toarray()
        self.img_dir = img_dir
        self.transform = transform
        self.datatype = datatype

    def __len__(self):
        return len(self.train_csv)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(self.img_dir+self.datatype+str(idx)+'.jpg')
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

    transformation = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

    dataset = Plain_Dataset(csv_file=traincsv_file,img_dir = 'Train/',datatype = 'train',transform = transformation)

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
        super(Deep_Emotion,self).__init__()
        self.conv1 = nn.Conv2d(1,10,3)
        self.conv2 = nn.Conv2d(10,10,3)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(10,10,3)
        self.conv4 = nn.Conv2d(10,10,3)
        self.pool4 = nn.MaxPool2d(2,2)

        #self.dropout = nn.Dropout2d()

        self.fc1 = nn.Linear(810,50)
        self.fc2 = nn.Linear(50,7)

    def forward(self,input):
        out = self.conv1(input)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.pool2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = F.relu(out)

        out = self.conv4(out)
        out = self.pool4(out)
        out = F.relu(out)

        out = F.dropout(out)
        out = out.view(-1, 810) #####
        out = self.fc1(out)
        out = self.fc2(out)

        return out

def Train():
    '''
    Documentation
    '''

    '''
    Load the data inform of iterator (Dataloader) but here we will use SubsetRandomSampler to split train into train and validation
    '''
    net = Deep_Emotion()
    net.cuda()

    transformation = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    train_dataset = Plain_Dataset(csv_file=traincsv_file,img_dir = 'Train/',datatype = 'train',transform = transformation)

    train_indices, validation_indices = train_val_split(train_dataset,0.2)

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler =  SubsetRandomSampler(validation_indices)

    train_loader = DataLoader(train_dataset,batch_size=64,num_workers=0,sampler=train_sampler)
    val_loader = DataLoader(train_dataset,batch_size=64,num_workers=0,sampler=val_sampler)


    criterion = nn.CrossEntropyLoss()
    optmizer = optim.Adam(net.parameters(), lr = lr)

    for e in tqdm(range(epochs)):
        train_loss = 0
        val_loss = 0
        # Train the model  #
        net.train()
        for data, lables in train_loader:
            data, lables = data.cuda(), lables.cuda()
            lables = torch.max(lables, 1)[1]

            optmizer.zero_grad()

            outputs = net(data)

            loss = criterion(outputs,lables)
            loss.backward()
            optmizer.step()
            train_loss += loss.item() * data.size(0)

        # validate the model #
        net.eval()
        for data, lables in val_loader:
            #
            data, lables = data.cuda(), lables.cuda()
            lables = torch.max(lables, 1)[1]
            outputs = net(data)

            loss = criterion(outputs, lables)

            val_loss += loss.item() * data.size(0)

        train_loss = train_loss/len(train_loader.sampler)
        val_loss = val_loss/len(val_loader.sampler)
        print('Epoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f}'.format(e+1, train_loss,val_loss))

    torch.save(net.state_dict(), 'model_noSTN.pt')

Train()
