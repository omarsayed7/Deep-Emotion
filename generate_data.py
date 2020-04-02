import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm


class Generate_data():
    def __init__(self):
        """
        Generate_data class
        Two methods to be used
        1-split_test
        2-save_images
        [Note] that you have to split the public and private from fer2013 file
        """
        pass

    def split_test(self, csv_path,test_filename = 'test', val_filename= 'val'):
        """
        Helper function to split the validation and test data from general test file as it contains (Public test, Private test)
            params:-
                csv_path = path to csv file of the test data
                test_filename, val_filename = desired name of the new saved files
        """
        test = pd.read_csv(csv_path)
        test_data = pd.DataFrame(test.iloc[:3589,:])
        validation_data = pd.DataFrame(test.iloc[3589:,:])
        test.to_csv(test_filename+".csv")
        val.to_csv(val_filename+".csv")

    def str_to_image(self, str_img = ' '):
        '''
        Convert string pixels from the csv file into image object
            params:- take an image string
            return :- return PIL image object
        '''
        imgarray_str = str_img.split(' ')
        imgarray = np.asarray(imgarray_str,dtype=np.uint8).reshape(48,48)
        return Image.fromarray(imgarray)

    def save_images(self, csvfile_path, foldername='', datatype='train'):
        '''
        save_images is a function responsible for saving images from data files e.g(train, test) in a desired folder
            params:-
            csvfile_path = path to csv file of the data e.g train.csv, val.csv, test.csv
            column_pixels_name = column name of the pixels images e.g 'pixels'
            foldername = name of the folder to save the images in
        '''
        if not os.path.exists(foldername):
            os.mkdir(foldername)

        data = pd.read_csv(csvfile_path)
        images = data['pixels'] #dataframe to series pandas
        numberofimages = images.shape[0]
        for index in tqdm(range(numberofimages)):
            img = self.str_to_image(images[index])
            img.save(os.path.join(foldername,'{}{}.jpg'.format(datatype,index)),'JPEG')
        print('Done saving {} data'.format((foldername)))
