import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def str_to_image(str_img = ' '):
    '''
    convert string pixels from the csv file into image object
    '''
    imgarray_str = str_img.split(' ')
    imgarray = np.asarray(imgarray_str,dtype=np.uint8).reshape(48,48)
    return Image.fromarray(imgarray)

def save_images(csvfile_path, column_pixels_name = ' ',foldername=''):
    '''
    csvfile_path == path to csv file of the training data e.g train.csv
    column_pixels_name == column name of the pixels images e.g 'pixels'
    foldername == name of the folder to save the images in
    '''
    if not os.path.exists(foldername):
        os.mkdir(foldername)

    data = pd.read_csv(csvfile_path)
    images = data[column_pixels_name] #dataframe to series pandas
    numberofimages = images.shape[0]
    for index in tqdm(range(numberofimages)):
        img = str_to_image(images[index])
        img.save(os.path.join(foldername,'train{}.jpg'.format(index)),'JPEG')
    print('Done saving {} data'.format((foldername)))

if __name__ == "__main__" :
    save_images(csvfile_path='F:/My_Stuff/1-MOOCS/AI/Deep Learning Notes/emotion detection/Dataset/Kaggle/train.csv',column_pixels_name='pixels',foldername='train')
    save_images(csvfile_path='F:/My_Stuff/1-MOOCS/AI/Deep Learning Notes/emotion detection/Dataset/Kaggle/test.csv',column_pixels_name='pixels',foldername='test')
