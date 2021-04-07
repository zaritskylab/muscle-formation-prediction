"""
Created on 08/09/2020

The script takes a tracking XML (based on the Widefield experiment videos), and its matching Brightfield video.
The output is a set of cropped images of the cells from each frame.

@author: Amit Shakarchy
"""
import collections

import cv2
import tensorflow as tf
import numpy as np
from skimage import io
from tensorflow.python.keras import Model
from tqdm import tqdm

from load_tracks_xml import load_tracks_xml
import pandas as pd

VIDEO_PATH = r"../../data/videos/Experiment1_w2Brightfield_s{}_all.tif"
XML_PATH = r"../../data/tracks_xml/Experiment1_w1Widefield550_s{}_all.xml"


def get_empty_encode_time_df():
    names = []
    for i in range(0, 50):
        names.append(i)
    names.append("time")
    enc_time_df = pd.DataFrame(columns=names)
    return enc_time_df


def crop_cells(normalized, bf_video, tm_xml, resize, resize_to, image_size, crop_single_cell=False):
    '''
    The function takes a tracking XML (based on the Widefield experiment videos), and its matching Brightfield video.
    The output is a set of cropped images of the cells from each frame.
    :param normalized: if True, images will be normalized to a scale of 0-1
    :param bf_video: bright field video path
    :param tm_xml: trackmate xml path
    :param resize: if true, images will be resized to resize_to
    :param resize_to: the wanted image size
    :param image_size: initial image size
    :return: numpy array of the cropped images, time stamps dataframe
    '''
    # Load the tracks XML (TrackMate's output)
    tracks01, df = load_tracks_xml(tm_xml)
    # Load the Brightfield matching video
    im = io.imread(bf_video)
    j = 0
    # Initialize dataset with all of the needed frames in it
    s01nuc = np.zeros((927, 1200 + image_size, 1200 + image_size), )
    s01nuc[:, (image_size // 2):(image_size // 2 + 1200), (image_size // 2):(image_size // 2 + 1200)] = im

    time_df = pd.DataFrame(columns=['time'])
    img_data_list = []

    # Iterate over all frames- of the tracked cells and their brightfeild's matching image
    for k, track in enumerate(tracks01):

        if (crop_single_cell):
            if k > 1:
                break
        start = int(np.min(np.asarray(track['t_stamp'])))
        stop = int(np.max(np.asarray(track['t_stamp'])))
        single_cell = np.zeros((stop - start + 1, image_size, image_size), )
        single_cell_crop = np.zeros((image_size, image_size), )

        # Skip in case the cell's path is too small
        if len(track) < 10:
            continue

        for frame, i in enumerate(track['t_stamp']):
            j = j + 1
            x = int(track['x'].iloc[frame] * 311.688)
            y = int(track['y'].iloc[frame] * 311.688)
            time = int(track['t_stamp'].iloc[frame])

            # Crop the needed image
            single_cell_crop = s01nuc[int(frame), y:y + image_size, x:x + image_size]

            # Skip in case the image has a black tip (For data-cleaning)
            if all(elem == 0 for elem in single_cell_crop[0]) or all(
                    elem == 0 for elem in single_cell_crop[image_size - 1]) or all(
                elem == 0 for elem in single_cell_crop[:, 0]) or all(
                elem == 0 for elem in single_cell_crop[:, image_size - 1]):
                continue
            single_cell_crop = np.array(single_cell_crop).astype('float32')
            if normalized:
                single_cell_crop = (single_cell_crop - single_cell_crop.min()) / (
                        single_cell_crop.max() - single_cell_crop.min())
            if resize:
                single_cell_crop = cv2.resize(single_cell_crop, (resize_to, resize_to), interpolation=cv2.INTER_AREA)

            # single_cell_crop = single_cell_crop.astype('float32')
            img_data_list.append(single_cell_crop)
            time_df.loc[len(time_df)] = [time]
    return np.array(img_data_list), time_df


def save_cells_images(images, path, prefix):
    '''
    saves the cropped images into a directory, as tif file
    :param images: numpy array with images to save
    :param path: path of directory to save to
    :return: -
    '''
    for ind, image in enumerate(images):
        # Save the cropped image
        io.imsave(path + str(prefix) + '_{0}.tif'.format(ind), image, check_contrast=False)


def build_encoded_cells_time_dataset(encoder_name, exp_list):
    # encoder = tf.keras.models.load_model("../models/" + encoder_name, compile=False)
    encoder = encoder_name
    enc_time_df = get_empty_encode_time_df()
    for i in tqdm(exp_list):  # ((3, 4, 6, 11, 12)):
        print(i)
        bf_video = r"../../data/videos/Experiment1_w2Brightfield_s{}_all.tif".format(
            i)
        tm_xml = r"../../data/tracks_xml/Experiment1_w1Widefield550_s{}_all.xml".format(
            i)
        cells_arr, time_df = crop_cells(
            normalized=True, bf_video=bf_video, tm_xml=tm_xml, resize=True, resize_to=64, image_size=64,
            crop_single_cell=False)

        encoded_images = encoder.predict(cells_arr)
        df = pd.DataFrame(data=encoded_images[2])
        df["time"] = time_df["time"]
        enc_time_df = enc_time_df.append(df)
    return enc_time_df


def build_encoded_labeled_dataset(to_normalize, img_size, encoder_name, exp_list):
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    # config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # encoder = tf.keras.models.load_model("../models/" + encoder_name, compile=False)
    encoder = encoder_name
    enc_time_df = get_empty_encode_time_df()
    for i in tqdm(exp_list):  # ((3, 4, 6, 11, 12)):  # video number 5 will be for testing
        print(i)
        cells_arr, time_df = crop_cells(normalized=to_normalize, bf_video=VIDEO_PATH.format(i),
                                        tm_xml=XML_PATH.format(i), resize=True, resize_to=img_size,
                                        image_size=img_size, crop_single_cell=False)

        encoded_images = encoder.predict(cells_arr)
        df = pd.DataFrame(data=encoded_images[2])
        df["time"] = time_df["time"]
        # Take images from the last 5 frames an from the first 5 frames
        df = df.drop(df[(df.time < (np.max(df["time"]) - 5)) & (5 < df.time)].index)

        # set low time values to 0
        df.loc[df['time'] <= 5, 'time'] = 0

        # set high time values to 1
        df.loc[df['time'] > 5, 'time'] = 1

        enc_time_df = enc_time_df.append(df)
    print("False is 1, True is 0 :", collections.Counter(df['time'] == 0))
    return enc_time_df


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def single_cell_time_df(encoder_name, exp_num):
    exp_ind = exp_num
    # encoder = tf.keras.models.load_model("../models/" + encoder_name, compile=False)
    encoder = encoder_name
    names = []
    for i in range(0, 50):
        names.append(i)
    names.append("time")
    enc_time_df = pd.DataFrame(columns=names)
    bf_video = r"../../data/videos/Experiment1_w2Brightfield_s{}_all.tif".format(
        exp_ind)
    tm_xml = r"../../data/tracks_xml/Experiment1_w1Widefield550_s{}_all.xml".format(
        exp_ind)
    cells_arr, time_df = crop_cells(normalized=True, bf_video=bf_video, tm_xml=tm_xml, resize=True, resize_to=64,
                                    image_size=64, crop_single_cell=True)
    encoded_images = encoder.predict(cells_arr)
    df = pd.DataFrame(data=encoded_images[2])
    df["time"] = time_df["time"]
    enc_time_df = enc_time_df.append(df)
    return enc_time_df


def get_data_for_classifier(encoder_name):
    # build train and test sets for classifier
    enc_lables_test = build_encoded_labeled_dataset(to_normalize=True, img_size=64,
                                                    encoder_name=encoder_name, exp_list=[5])
    print(len(enc_lables_test))
    enc_lables_train = build_encoded_labeled_dataset(to_normalize=True, img_size=64,
                                                     encoder_name=encoder_name,
                                                     exp_list=[3, 4, 6, 11, 12])
    print(len(enc_lables_train))
    enc_lables_test.to_pickle("encoded_labeled - test normalized")
    enc_lables_train.to_pickle("encoded_labeled - train normalized")
    # build real time data for test (video 5)
    all_times_encodings = build_encoded_cells_time_dataset(encoder_name=encoder_name,
                                                           exp_list=[5])
    print((all_times_encodings))
    all_times_encodings.to_pickle("test - encoded video 5 with timestamp")

    # build real time data for test (video 7)
    all_times_encodings = build_encoded_cells_time_dataset(encoder_name=encoder_name,
                                                           exp_list=[7])
    print((all_times_encodings))
    all_times_encodings.to_pickle("test - encoded video 7 with timestamp")
    # get single cell tracks to classify:
    # diff:
    single_cell_df = single_cell_time_df(encoder_name=encoder_name,
                                         exp_num=5)
    print((single_cell_df))
    single_cell_df.to_pickle("single cell - exp 5 - diff")
    # control:
    single_cell_df = single_cell_time_df(encoder_name=encoder_name,
                                         exp_num=7)
    print((single_cell_df))
    single_cell_df.to_pickle("single cell - exp 7 - con")


if __name__ == '__main__':
    from skimage import io
    image_size = 60

    bf_video = r"../../data/videos/Experiment1_w2Brightfield_s3_all.tif"
    im = io.imread(bf_video)
    s01nuc = np.zeros((927, 1200 + image_size, 1200 + image_size), )
    s01nuc[:, (image_size // 2):(image_size // 2 + 1200), (image_size // 2):(image_size // 2 + 1200)] = im

    # encoder_name = "encoder_vae_exp_261"
    bf_video = r"../../data/videos/Experiment1_w2Brightfield_s3_all.tif"
    tm_xml = r'../../data/tracks_xml/Experiment1_w1Widefield550_s3_all.xml'
    crop_cells(normalized=False, bf_video=bf_video, tm_xml=tm_xml, resize=True, resize_to=64,
               image_size=64, crop_single_cell=False)

    path = "../aae/my_vae_exp_256"
    vae = tf.keras.models.load_model(path, compile=False)
    encoder = Model(vae.input, vae.get_layer("origin_encoder").output)



    get_data_for_classifier(encoder)
    # for i in ((8, 9, 10, 11, 12)):  # 8 ,9
    #     print(i)
    #     bf_video = r"../../data/videos/Experiment1_w2Brightfield_s{}_all.tif".format(i)
    #     tm_xml = r'../../data/tracks_xml/Experiment1_w1Widefield550_s{}_all.xml'.format(i)
    #     images, time_df = crop_cells(False, bf_video, tm_xml, False, resize_to=64, image_size=64)
    #     print("cropped cells from video #{}".format(i))
    #     save_cells_images(images, "../../data/images/train/".format(i), i)
