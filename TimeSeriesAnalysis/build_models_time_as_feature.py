import pickle

import diff_tracker_utils as utils
import pandas as pd
from mast_intensity import get_intensity_measures_df

from diff_tracker import DiffTracker
from imblearn.over_sampling import RandomOverSampler
from intensity_erk_compare import get_tracks_list
import numpy as np

from mast_intensity import prep_data


def get_data_to_run(to_run):
    if to_run == "intensity":
        diff_df_train = get_intensity_measures_df(csv_path=path + r"/data/mastodon/train/Nuclei_5-vertices.csv",
                                                  video_actin_path=path + r"/data/videos/train/S5_Actin.tif",
                                                  window_size=window_size)
        con_df_train = get_intensity_measures_df(csv_path=path + r"/data/mastodon/train/Nuclei_1-vertices.csv",
                                                 video_actin_path=path + r"/data/videos/train/S1_Actin.tif",
                                                 window_size=window_size)
        con_df_test = get_intensity_measures_df(csv_path=path + r"/data/mastodon/test/Nuclei_2-vertices.csv",
                                                video_actin_path=path + r"/data/videos/test/S2_Actin.tif",
                                                window_size=window_size)
        diff_df_test = get_intensity_measures_df(csv_path=path + r"/data/mastodon/test/Nuclei_3-vertices.csv",
                                                 video_actin_path=path + r"/data/videos/test/S3_Actin.tif",
                                                 window_size=window_size)
        normalize_func = utils.normalize_intensity
        drop_columns_func = utils.drop_columns_intensity

    if to_run == "motility":
        diff_df_train = pd.read_csv(path + "/data/mastodon/train/Nuclei_5-vertices.csv", encoding="cp1252").dropna()
        con_df_train = pd.read_csv(path + "/data/mastodon/train/Nuclei_1-vertices.csv", encoding="cp1252").dropna()
        diff_df_test = pd.read_csv(path + "/data/mastodon/test/Nuclei_3-vertices.csv", encoding="cp1252").dropna()
        con_df_test = pd.read_csv(path + "/data/mastodon/test/Nuclei_2-vertices.csv", encoding="cp1252").dropna()

        normalize_func = utils.normalize_motility
        drop_columns_func = utils.drop_columns_motility

    return diff_df_train, con_df_train, con_df_test, diff_df_test, normalize_func, drop_columns_func


def prep_data_TIME(diff_df, con_df):
    print("concatenating control data & ERKi data")
    diff_df['target'] = np.array([True for i in range(len(diff_df))])
    con_df['target'] = np.array([False for i in range(len(con_df))])
    df = pd.concat([diff_df, con_df], ignore_index=True)

    print("dropping irrelevant columns")
    df = diff_tracker_model.drop_columns(df)

    print("normalizing data")
    df = diff_tracker_model.normalize(df)

    print("add time stamp")
    df["time_stamp"] = df["Spot frame"]

    df = df.sample(frac=1).reset_index(drop=True)

    y = pd.Series(df['target'])
    y.index = df['Spot track ID']
    y = utils.get_unique_indexes(y)
    df = df.drop("target", axis=1)
    return df, y


if __name__ == '__main__':
    cluster_path = "muscle-formation-diff"
    local_path = ".."
    path = cluster_path
    window_size = 40
    to_run = "motility"

    diff_df_train, con_df_train, con_df_test, diff_df_test, normalize_func, drop_columns_func = get_data_to_run(to_run)

    diff_window = [140, 170]
    con_windows = [[[70, 100], [100, 130], [140, 170], [180, 210], [210, 240]]]
    # diff_window =[0,258]
    # con_windows = [[[0,258]]]
    track_length = 30

    for w in con_windows:
        dir_path = f"14-03-2022-manual_mastodon_{to_run} time as feature"
        second_dir = f"{diff_window} frames ERK, {w} frames con"
        utils.open_dirs(dir_path, second_dir)
        dir_path += "/" + second_dir

        diff_tracker_model = DiffTracker(normalize=normalize_func, drop_columns=drop_columns_func,
                                         concat_dfs=utils.concat_dfs, dir_path=dir_path,
                                         diff_window=diff_window, con_windows=w)

        # X_train, y_train = prep_data_TIME(diff_df_train, con_df_train)
        # X_test, y_test = prep_data_TIME(diff_df_test, con_df_test)
        #
        X_train, y_train = diff_tracker_model.prep_data(diff_df=diff_df_train, con_df=con_df_train,
                                     diff_t_window=diff_window, con_t_windows=w, add_time=True)
        X_test, y_test = diff_tracker_model.prep_data(diff_df=diff_df_test, con_df=con_df_test,
                                   diff_t_window=diff_window, con_t_windows=w, add_time=True)

        print("fit into ts-fresh dataframe")
        X_train, X_test = diff_tracker_model.fit_transform_tsfresh(X_train, y_train, X_test)

        print("training")
        clf = utils.train(X_train, y_train)
        diff_tracker_model.clf = clf
        utils.save_data(diff_tracker_model.dir_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                        clf=clf)

        diff_tracker_model.evaluate_clf(X_test, y_test)

        motility = True
        intensity = False
        wt_cols = [wt for wt in range(0, 260, track_length)]

        csv_path = path + fr"/data/mastodon/test/Nuclei_2-vertices.csv"
        print(csv_path)
        df = pd.read_csv(csv_path, encoding="ISO-8859-1")
        df = df[df["manual"] == 1]

        tracks = get_tracks_list(df, target=0)[:15]
        prob_ = utils.calc_prob_delta(window=track_length, tracks=tracks, clf=clf, X_test=X_test, motility=motility,
                                      visual=intensity,
                                      wt_cols=wt_cols, moving_window=True, aggregate_windows=False, calc_delta=False,
                                      add_time=True)
        pickle.dump(prob_, open(dir_path + "/" + f"df_prob_w={track_length}, video_num={2}", 'wb'))

        csv_path = path + fr"/data/mastodon/test/Nuclei_3-vertices.csv"
        print(csv_path)
        df = pd.read_csv(csv_path, encoding="ISO-8859-1")
        df = df[df["manual"] == 1]

        tracks = get_tracks_list(df, target=1)[:15]
        prob_ = utils.calc_prob_delta(window=track_length, tracks=tracks, clf=clf, X_test=X_test, motility=motility,
                                      visual=intensity,
                                      wt_cols=wt_cols, moving_window=True, aggregate_windows=False, calc_delta=False,
                                      add_time=True)
        pickle.dump(prob_, open(dir_path + "/" + f"df_prob_w={track_length}, video_num={3}", 'wb'))
