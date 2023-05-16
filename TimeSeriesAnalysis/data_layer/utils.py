import pickle

import pandas as pd
import joblib
import re
import numpy as np

import sys, os

sys.path.append('/sise/home/shakarch/muscle-formation-diff')
sys.path.append(os.path.abspath('..'))

sys.path.append(os.path.abspath('../..'))
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from configuration import params, consts


def load_data(dir_path, load_clf=True, load_x_train=True, load_x_test=True, load_y_train=True, load_y_test=True):
    """
    Loads a classifier, train & test data of a given model (by its path)
    :param dir_path: (str) Directory path where the files are located.
    :param load_clf: (bool, optional) Whether to load the classifier. Default is True.
    :param load_x_train: (bool, optional) Whether to load the training input data. Default is True.
    :param load_x_test: (bool, optional) Whether to load the testing input data. Default is True.
    :param load_y_train: (bool, optional) Whether to load the training target data. Default is True.
    :param load_y_test: (bool, optional) Whether to load the testing target data. Default is True.
    :return: A tuple containing the loaded data in the following order:
        (Classifier object (clf), Training input data (x_train),
        Testing input data (x_test), Training target data (y_train), Testing target data (y_test))
    """
    clf = joblib.load(dir_path + "/clf.joblib") if load_clf else None
    x_train = pd.read_csv(dir_path + "/" + "X_train", encoding="cp1252") if load_x_train else None
    x_test = pd.read_csv(dir_path + "/" + "X_test", encoding="cp1252") if load_x_test else None
    y_train = pd.read_csv(dir_path + "/" + "y_train", encoding="cp1252") if load_y_train else None
    y_test = pd.read_csv(dir_path + "/" + "y_test", encoding="cp1252") if load_y_test else None

    return clf, x_train, x_test, y_train, y_test


def save_data(dir_path, clf=None, X_train=None, X_test=None, y_train=None, y_test=None):
    """
    Saves a classifier, train & test data of a given model in a directory.
    :param dir_path: (str) Directory path where the files will be saved
    :param clf: (bool, optional) Whether to save the classifier. Default is True.
    :param X_train: (bool, optional) Whether to save the training input data. Default is True.
    :param X_test: (bool, optional) Whether to save the testing input data. Default is True.
    :param y_train: (bool, optional) Whether to save the training target data. Default is True.
    :param y_test: (bool, optional) Whether to save the testing target data. Default is True.
    :return: 
    """
    if X_train is not None:
        X_train.to_csv(dir_path + "/" + "X_train")
    if X_test is not None:
        X_test.to_csv(dir_path + "/" + "X_test")
    if y_test is not None:
        y_test.to_csv(dir_path + "/" + "y_test")
    if y_train is not None:
        y_train.to_csv(dir_path + "/" + "y_train")
    if clf is not None:
        joblib.dump(clf, dir_path + "/" + "clf.joblib")


def downcast_df(data_copy, fillna=True):
    """
    Downcasts the data types of a DataFrame to reduce memory usage.
    :param data_copy: (pd.DataFrame) The DataFrame to be downcasted.
    :param fillna: (bool, optional) Whether to fill NaN values with zero. Default is True.
    :return: (pd.DataFrame) The downcasted DataFrame.
    """
    data_copy = data_copy.copy()
    if fillna:
        data_copy = data_copy.fillna(0)
    data_copy = data_copy.dropna(axis=1)
    cols = list(data_copy.drop(columns="Spot track ID").columns) if "spot track ID" in data_copy.columns else list(
        data_copy.columns)
    for col in cols:
        try:
            if data_copy[col].sum().is_integer():
                data_copy[col] = pd.to_numeric(data_copy[col], downcast='integer')
            else:
                data_copy[col] = pd.to_numeric(data_copy[col], downcast='float')

            if np.isinf(data_copy[col]).sum() > 0:
                data_copy[col] = data_copy[col]
        except:
            continue
    return data_copy


def get_tracks_list(int_df, target):
    """
    Returns a list of tracks, Grouped by their 'Spot track ID'.
    :param int_df: (pd.DataFrame) The DataFrame containing the tracks.
    :param target: (bool, optional) The target value to assign to each track.
    :return: (list) A list of tracks (DataFrames) grouped by 'Spot track ID'
    """
    if target:
        int_df['target'] = target
    tracks = list()
    for label, labeled_df in int_df.groupby('Spot track ID'):
        tracks.append(labeled_df)
    return tracks


def load_clean_rows(file_path):
    """
    Loads and cleans rows from a CSV file.
    :param file_path: (str) The path to the CSV file.
    :return: (pd.DataFrame) The cleaned DataFrame.
    """
    df = pd.DataFrame()
    chunk_size = 10 ** 3
    with pd.read_csv(file_path, encoding="cp1252",
                     chunksize=chunk_size) as reader:
        for chunk in reader:
            chunk = downcast_df(chunk)
            df = df.append(chunk)
            df = pd.concat([df, chunk], ignore_index=True)
    try:
        df = df.drop(labels=range(0, 2), axis=0)
    except Exception as e:
        print(e)

    return df


def get_tracks(file_path, tagged_only=False, manual_tagged_list=True, target=1):
    """
    Loads and processes singe cells tracks from a csv file, returning theire DataFrame and a list of tracks.
    :param file_path:  (str) The path to the file containing the tracks.
    :param tagged_only: (bool, optional) If True, only include manual annotated tracks only. Default is False.
    :param manual_tagged_list: (bool, optional) If True, returns manual tagged list. Default is True.
    :param target: (bool, optional) The target value to assign to each track.
    :return: (tuple) A tuple containing the DataFrame and a list of tracks.
    """
    df_s = load_clean_rows(file_path)
    df_s = df_s.rename(columns=lambda x: re.sub('[^A-Za-z0-9 _.]+', '', x))
    df_s = df_s.rename(columns={"Spot position": "Spot position X", "Spot position.1": "Spot position Y",
                                "Spot position X m": "Spot position X", "Spot position Y m": "Spot position Y"})
    cols = ["Spot position X", "Spot position Y", "manual", "Spot frame", "Spot track ID"]
    df_s = df_s[cols]
    df_s = df_s.astype(float)
    df_s = downcast_df(df_s)
    df_s = df_s[df_s["manual"] == 1] if tagged_only else df_s
    data = df_s[df_s["manual"] == 1] if manual_tagged_list else df_s
    tracks_s = get_tracks_list(data, target=target)
    return df_s, tracks_s


def get_all_properties_df(modality, con_train_vid_num, diff_train_vid_num, vid_num):
    """
    Loads and returns the DataFrame containing all properties previously calculated for data from a specific video
    and a specific modality.
    :param modality: (str) The modality of the properties ("actin_intensity" or "motility").
    :param con_train_vid_num: (int) The DMSO (control) train video number.
    :param diff_train_vid_num: (int) The differentiated (ERKi) train video number.
    :param vid_num: (int): The video number to get its properties.
    :return: (pd.DataFrame) The DataFrame containing all properties.
    """
    properties_data_path = consts.intensity_model_path if modality == "actin_intensity" else consts.motility_model_path
    properties_data_path = properties_data_path % (con_train_vid_num, diff_train_vid_num)
    properties_df = pickle.load(
        open(properties_data_path + f"/S{vid_num}_properties_{params.registration_method}" + ".pkl", 'rb'))
    return properties_df
