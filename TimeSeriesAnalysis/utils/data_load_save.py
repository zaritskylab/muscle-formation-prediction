import pickle

import pandas as pd
import joblib
import re
import numpy as np
import sys

sys.path.append('/sise/home/shakarch/muscle-formation-diff')
import TimeSeriesAnalysis.params as params


def load_clean_rows(file_path):
    # df = pd.read_csv(file_path, encoding="cp1252")

    df = pd.DataFrame()
    chunksize = 10 ** 3
    with pd.read_csv(file_path, encoding="cp1252",
                     chunksize=chunksize) as reader:  # , dtype=np.float32,  index_col=[0],
        for chunk in reader:
            # try:
            #     chunk = chunk.drop(labels=range(0, 2), axis=0)
            # except Exception as e:
            #     continue

            chunk = downcast_df(chunk)
            df = df.append(chunk)

    # df = df.drop(labels=range(0, 2), axis=0)
    try:
        df = df.drop(labels=range(0, 2), axis=0)
    except Exception as e:
        print(e)
    print(df.info(memory_usage='deep'), flush=True)
    # print(df.head())

    return df


def load_data(dir_name, load_clf=True, load_x_train=True, load_x_test=True, load_y_train=True, load_y_test=True):
    clf = joblib.load(dir_name + "/clf.joblib") if load_clf else None
    x_train = pd.read_csv(dir_name + "/" + "X_train", encoding="cp1252") if load_x_train else None
    x_test = pd.read_csv(dir_name + "/" + "X_test", encoding="cp1252") if load_x_test else None
    y_train = pd.read_csv(dir_name + "/" + "y_train", encoding="cp1252") if load_y_train else None
    y_test = pd.read_csv(dir_name + "/" + "y_test", encoding="cp1252") if load_y_test else None

    return clf, x_train, x_test, y_train, y_test


def get_tracks_list(int_df, target):
    if target:
        int_df['target'] = target
    tracks = list()
    for label, labeled_df in int_df.groupby('Spot track ID'):
        tracks.append(labeled_df)
    return tracks


def save_data(dir_name, clf=None, X_train=None, X_test=None, y_train=None, y_test=None):
    if X_train is not None:
        X_train.to_csv(dir_name + "/" + "X_train")
    if X_test is not None:
        X_test.to_csv(dir_name + "/" + "X_test")
    if y_test is not None:
        y_test.to_csv(dir_name + "/" + "y_test")
    if y_train is not None:
        y_train.to_csv(dir_name + "/" + "y_train")
    if clf is not None:
        joblib.dump(clf, dir_name + "/" + "clf.joblib")


def downcast_df(data_copy, fillna=True):
    # data_copy = data.copy()
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

    # print(data_copy.info(memory_usage='deep'))
    return data_copy


def get_tracks(file_path, tagged_only=False, manual_tagged_list=True, target=1):
    df_s = load_clean_rows(file_path)

    df_s = df_s.rename(columns=lambda x: re.sub('[^A-Za-z0-9 _.]+', '', x))

    df_s = df_s.rename(columns={"Spot position": "Spot position X", "Spot position.1": "Spot position Y",
                                "Spot position X m": "Spot position X", "Spot position Y m": "Spot position Y",
                                "Spot center intensity": "Spot center intensity Center ch1 (Counts)",
                                "Spot intensity": "Spot intensity Mean ch1 (Counts)",
                                "Spot intensity.1": "Spot intensity Std ch1 (Counts)",
                                "Spot intensity.2": "Spot intensity Min ch1 (Counts)",
                                "Spot intensity.3": "Spot intensity Max ch1 (Counts)",
                                "Spot intensity.4": "Spot intensity Median ch1 (Counts)",
                                "Spot intensity.5": "Spot intensity Sum ch1 (Counts)",
                                })
    cols = ["Spot position X", "Spot position Y", "manual", "Spot frame", "Spot track ID"]
    df_s = df_s[cols]
    df_s = df_s.astype(float)
    df_s = downcast_df(df_s)
    df_s = df_s[df_s["manual"] == 1] if tagged_only else df_s
    data = df_s[df_s["manual"] == 1] if manual_tagged_list else df_s
    tracks_s = get_tracks_list(data, target=target)
    return df_s, tracks_s


def get_all_properties_df(modality, con_train_vid_num, diff_train_vid_num, scores_vid_num,
                          reg_method=params.registration_method):
    properties_data_path = fr"/storage/users/assafzar/Muscle_Differentiation_AvinoamLab/30-07-2022-{modality} local dens-False, s{con_train_vid_num}, s{diff_train_vid_num} train" + (
        " win size 16" if modality != "motility" else "") + fr"/track len 30, impute_func-{params.impute_methodology}_{params.impute_func} reg {reg_method}/S{scores_vid_num}_properties_{reg_method}"
    properties_df = pickle.load(open(properties_data_path + ".pkl", 'rb'))
    return properties_df
