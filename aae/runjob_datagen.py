import os
from os import listdir
import cv2
from random import shuffle, random
import tensorflow as tf
import numpy as np


def load_images(directory, files, image_size):
    imageBatch = []
    for i in range(len(files)):
        sample = files[i]
        image = cv2.imread(directory + "/" + sample, -1)
        image = cv2.resize(image, (image_size, image_size))
        image = image.astype("float32")
        imageBatch.append(image)
    return np.asarray(imageBatch)


def aae_load_images(image_size):
    directory = "muscle-formation-diff/data/images/new_test"
    files = listdir(directory)
    imageBatch = []
    for i in range(300):
        sample = files[i]
        image = cv2.imread(directory + "/" + sample, -1)
        try:
            image = cv2.resize(image, (image_size, image_size))
            image = image.astype("float32")
            image = (image - image.min()) / (image.max() - image.min())
            imageBatch.append(image)
        except:
            print(directory + "/" + sample)
    return np.asarray(imageBatch)





def aae_generator():
    img_size = 64
    directory = "muscle-formation-diff/data/images/train"
    file_list = os.listdir(directory)
    # shuffle(file_list)
    for i in range(len(file_list)):
        sample = file_list[i]
        image = cv2.imread(directory + "/" + sample, -1)
        try:
            image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
            image = image.astype("float32")
            image = (image - image.min()) / (image.max() - image.min())
            yield image
        except:
            print(directory + "/" + sample)

def imageLoader(directory, files, batch_size, img_size):
    L = len(files)

    # this line is just to make the generator infinite, keras needs that
    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            X = load_images(directory, files[batch_start:limit], image_size=img_size)
            Y = X
            yield (X, Y)  # a tuple with two numpy arrays with batch_size samples
            batch_start += batch_size
            batch_end += batch_size

def generate_Data(directory, batch_size, img_size):
    i = 0
    file_list = os.listdir(directory)
    # shuffle(file_list)
    while True:
        image_batch = []

        for j in range(batch_size):
            if (i < len(file_list)):
                sample = file_list[i]
                i += 1
                image = cv2.imread(directory + "/" + sample, -1)
                image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
                image = image.astype("float32")
                image = (image - image.min()) / (image.max() - image.min())
                image_batch.append(image)
            else:
                i = 0

        image_batch = np.array(image_batch)
        yield image_batch, image_batch
