import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
'''
train = 28709 imgs
private val =
public test = 3589
'''
def split_test_file(csv_path):
    """
    note that you have to split the public and private from fer2013 file
    """
    test = pd.read_csv(csv_path)
    test_data = pd.DataFrame(test.iloc[:3589,:])
    validation_data = pd.DataFrame(test.iloc[3589:,:])
    test.to_csv("Test.csv")
    val.to_csv('val.csv')

def str_to_image(str_img = ' '):
    '''
    convert string pixels from the csv file into image object
    '''
    imgarray_str = str_img.split(' ')
    imgarray = np.asarray(imgarray_str,dtype=np.uint8).reshape(48,48)
    return Image.fromarray(imgarray)

def save_images(csvfile_path, column_pixels_name = ' ',foldername='',datatype='train'):
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
        img.save(os.path.join(foldername,'{}{}.jpg'.format(datatype,index)),'JPEG')
    print('Done saving {} data'.format((foldername)))

if __name__ == "__main__" :
    split_test_file("F:/My_Stuff/1-MOOCS/AI/Deep Learning Notes/emotion detection/Dataset/Kaggle/test.csv")
    save_images(csvfile_path='F:/My_Stuff/1-MOOCS/AI/Deep Learning Notes/emotion detection/Dataset/Kaggle/val.csv',column_pixels_name='pixels',foldername='validation',datatype='val')
    save_images(csvfile_path='F:/My_Stuff/1-MOOCS/AI/Deep Learning Notes/emotion detection/Dataset/Kaggle/test.csv',column_pixels_name='pixels',foldername='test',datatype='test')
