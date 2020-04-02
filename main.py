import numpy  as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from data_loaders import Plain_Dataset, eval_data_dataloader
from deep_emotion import Deep_Emotion
from generate_data import Generate_data

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



def Train(epochs,train_loader,val_loader,criterion,optmizer,device):
    '''
    Training Loop
    '''
    print("===================================Start Training===================================")
    for e in range(epochs):
        train_loss = 0
        validation_loss = 0
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

        #validate the model#
        net.eval()
        for data,labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            val_outputs = net(data)
            val_loss = criterion(val_outputs, labels)
            validation_loss += val_loss.item()
            _, val_preds = torch.max(val_outputs,1)
            val_correct += torch.sum(val_preds == labels.data)

        train_loss = train_loss/len(train_dataset)
        train_acc = train_correct.double() / len(train_dataset)
        validation_loss =  validation_loss / len(validation_dataset)
        val_acc = val_correct.double() / len(validation_dataset)
        print('Epoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tTraining Acuuarcy {:.3f}% \tValidation Acuuarcy {:.3f}%'
                                                           .format(e+1, train_loss,validation_loss,train_acc * 100, val_acc*100))

    torch.save(net.state_dict(),'deep_emotion-{}-{}-{}.pt'.format(epochs,batchsize,lr))
    print("===================================Training Finished===================================")


if __name__ == '__main__':
    generate_dataset = Generate_data()
    generate_dataset.split_test()
    generate_dataset.save_images()

    net = Deep_Emotion()
    net.to(device)
    print("Model archticture: ", net)

    transformation = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    train_dataset =      Plain_Dataset(csv_file=traincsv_file,img_dir = train_img_dir,datatype = 'train',transform = transformation)
    validation_dataset = Plain_Dataset(csv_file=validationcsv_file,img_dir = validation_img_dir,datatype = 'val',transform = transformation)
    train_loader = DataLoader(train_dataset,batch_size=batchsize,shuffle = True,num_workers=0)
    val_loader =   DataLoader(validation_dataset,batch_size=batchsize,shuffle = True,num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optmizer = optim.Adam(net.parameters(),lr= lr)
    Train(epochs, train_loader, val_loader, criterion, optmizer, device)
