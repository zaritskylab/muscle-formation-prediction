import tqdm
import os
import cv2
import numpy as np
import pytrackmate as tm
from skimage import io
from sklearn.model_selection import train_test_split
import Scripts.Coordination.CoordinationValidations

"""
Created on 10/09/2020

The script receives a location of all the data (images),
the shape of the images and the size of the portion of the new_test group.
Then, it normalizes the data, shuffles it and divides it to new_test and training portions.

@author: Amit Shakarchy
"""


class DataPrep():

    def __init__(self, img_folder_path, IMG_HEIGHT, IMG_WIDTH, TEST_PERCENT):
        self.imgg_folder_path = img_folder_path
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.TEST_PERCENT = TEST_PERCENT
        # define temporal empty arrays:
        self.training = np.empty((3, 3), dtype=float, order='C')
        self.test = np.empty((3, 3), dtype=float, order='C')

    def load_dataset_from_images(self, normalize):
        '''
        receives a location of all the data (images),
        the shape of the images and the size of the portion of the new_test group.
        Then, it reads the data, shuffles it and divides it to new_test and training portions.
        :param normalize: if True, images will be normalized to a range of [0,255]
        :return: fills in the new_test and train np arrays of the class
        '''
        print("=== Reading Data ===")
        img_data_list = []
        for file in tqdm.tqdm(os.listdir(self.imgg_folder_path)):
            image_path = os.path.join(self.imgg_folder_path, file)
            if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                image = cv2.imread(image_path, -1)
                image = cv2.resize(image, (self.IMG_HEIGHT, self.IMG_WIDTH), interpolation=cv2.INTER_AREA)
                image = np.array(image)
                image = image.astype('float32')
                if normalize:
                    image = (image - image.min()) / (image.max() - image.min())

                img_data_list.append(image)

        print("=== Creating Dataset ===")
        np.random.shuffle(img_data_list)
        splitting_value = int(round(self.TEST_PERCENT * len(img_data_list)))
        test, training = img_data_list[:splitting_value], img_data_list[splitting_value:]
        self.training = np.array(training)
        self.test = np.array(test)

    def save_dataset(self, training_name, test_name):
        '''
        saves the dataset into np arrays
        :param training_name: training's file name
        :param test_name: new_test's file name
        :return:
        '''
        np.save(training_name + ".npy", self.training)
        np.save(test_name + ".npy", self.test)

    def load_dataset(self, training_path, test_path):
        '''
        loads the dataset
        :param training_path: training's file path
        :param test_path: new_test's file path
        :return:
        '''
        self.training = np.load(training_path + ".npy")
        self.test = np.load(test_path + ".npy")


if __name__ == '__main__':
    prep = DataPrep(r"C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Analysis\Autoencoder\all_images_dif",
                    32, 32, 0.2)
    prep.load_dataset_from_images(True)
    prep.save_dataset("train_32_norm", "test_32_norm")
