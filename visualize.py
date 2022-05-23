from __future__ import print_function
import argparse
import os
import cv2
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from deep_emotion import Deep_Emotion
from data_loaders import Plain_Dataset, eval_data_dataloader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Configuration of testing process")
parser.add_argument('-d', '--data', type=str,required = True, help='Folder that contains the finaltest.csv and test images')
parser.add_argument('-m', '--model', type=str,required = True, help='Path to pretrained model')
parser.add_argument('-t', '--test_acc', type=bool, help='Only show test accuarcy')
parser.add_argument('-c', '--cam', type=bool, help='Test the model in real time with webcam connect via usb')
args = parser.parse_args()

transformation = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
dataset = Plain_Dataset(csv_file=args.data+'/test.csv',img_dir = args.data+'/'+'test/',datatype = 'finaltest',transform = transformation)
test_loader =  DataLoader(dataset,batch_size=64,num_workers=0)

net = Deep_Emotion()
print("Deep Emotion:-", net)
net.load_state_dict(torch.load(args.model))
net.to(device)
net.eval()
#Model Evaluation on test data
classes = ('Angry', 'Disgust', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral')
total = []
if args.test_acc:
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = net(data)
            pred = F.softmax(outputs,dim=1)
            classs = torch.argmax(pred,1)
            wrong = torch.where(classs != labels,torch.tensor([1.]).cuda(),torch.tensor([0.]).cuda())
            acc = 1- (torch.sum(wrong) / 64)
            total.append(acc.item())

    print('Accuracy of the network on the test images: %d %%' % (100 * np.mean(total)))


#helper_function for real time testing
def load_img(path):
    img = Image.open(path)
    img = transformation(img).float()
    img = torch.autograd.Variable(img,requires_grad = True)
    img = img.unsqueeze(0)
    return img.to(device)

if args.cam:
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('cascade_model/haarcascade_frontalface_default.xml')

    # To capture video from webcam.
    cap = cv2.VideoCapture(0)
    while True:
        # Read the frame
        _, img = cap.read()
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            roi = img[y:y+h, x:x+w]
            roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(roi,(48,48))
            cv2.imwrite("roi.jpg", roi)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        imgg = load_img("roi.jpg")
        out = net(imgg)
        pred = F.softmax(out)
        classs = torch.argmax(pred,1)
        wrong = torch.where(classs != 3,torch.tensor([1.]).cuda(),torch.tensor([0.]).cuda())
        classs = torch.argmax(pred,1)
        prediction = classes[classs.item()]

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        img = cv2.putText(img, prediction, org, font,
                       fontScale, color, thickness, cv2.LINE_AA)

        cv2.imshow('img', img)
        # Stop if (Q) key is pressed
        k = cv2.waitKey(30)
        if k==ord("q"):
            break

    # Release the VideoCapture object
    cap.release()
