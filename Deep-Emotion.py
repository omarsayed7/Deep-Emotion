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

epochs = 50
lr = 0.005
batchsize = 128


class Plain_Dataset(Dataset):
    def __init__(self,csv_file,img_dir,datatype,transform):
        '''
        Documentation
        NO OneHot encoding
        '''
        self.csv_file = pd.read_csv(csv_file)
        self.lables = self.csv_file['emotion']
        self.img_dir = img_dir
        self.transform = transform
        self.datatype = datatype

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(self.img_dir+self.datatype+str(idx)+'.jpg')
        lables = np.array(self.lables[idx])
        lables = torch.from_numpy(lables).long()


        if self.transform :
            img = self.transform(img)


        return img,lables



def eval_data_dataloader(csv_file,img_dir,datatype,sample_number):
    '''
    Documentation
    '''

    transformation = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    dataset = Plain_Dataset(csv_file=csv_file,img_dir = img_dir,datatype = datatype,transform = transformation)

    lable = dataset.__getitem__(sample_number)[1]
    print(lable)

    imgg = dataset.__getitem__(sample_number)[0]
    imgnumpy = imgg.numpy()
    imgt = imgnumpy.squeeze()
    plt.imshow(imgt)
    plt.show()

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
    print("===================================Start Training===================================")

    print("Model archticture: ", net)
    transformation = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

    train_dataset =      Plain_Dataset(csv_file=traincsv_file,img_dir = train_img_dir,datatype = 'train',transform = transformation)
    validation_dataset = Plain_Dataset(csv_file=validationcsv_file,img_dir = validation_img_dir,datatype = 'val',transform = transformation)
    test_dataset =       Plain_Dataset(csv_file=traincsv_file,img_dir = test_img_dir,datatype = 'test',transform = transformation)

    train_loader = DataLoader(train_dataset,batch_size=batchsize,num_workers=0)
    val_loader =   DataLoader(validation_dataset,batch_size=batchsize,num_workers=0)
    test_loader =  DataLoader(test_dataset,batch_size=batchsize,num_workers=0)

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
            data, lables = data.cuda(), lables.cuda()
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
            data, lables = data.cuda(), lables.cuda()
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

    torch.save(net.state_dict(), 'model_noSTN-{}-{}-{}.pt'.format(epochs,batchsize,lr))
    print("===================================Training Finished===================================")


Train()
