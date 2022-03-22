import pickle

import diff_tracker_utils as utils
import pandas as pd
from mast_intensity import get_intensity_measures_df

from diff_tracker import DiffTracker
from imblearn.over_sampling import RandomOverSampler
from intensity_erk_compare import get_tracks_list


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


if __name__ == '__main__':
    cluster_path = "muscle-formation-diff"
    local_path = ".."
    path = cluster_path
    window_size = 40
    to_run = "motility"

    diff_df_train, con_df_train, con_df_test, diff_df_test, normalize_func, drop_columns_func = get_data_to_run(to_run)

    diff_windows = [[140, 170]]
    con_windows = [
        [[140, 170], [180, 210], [220, 250]],
        [[0, 30], [40, 70], [90, 120], [140, 170], [180, 210], [220, 250]],
        [[0, 30], [30, 60], [60, 90], [90, 120], [120, 150], [150, 180], [180, 210], [210, 240]],
        [[10, 40], [40, 70], [70, 100], [100, 130], [130, 160], [160, 190], [190, 220], [220, 250]]
    ]
    for diff_window in diff_windows:
        for w in con_windows:
            tracks_len = 30
            dir_path = f"14-03-2022-manual_mastodon_{to_run}"
            second_dir = f"{diff_window} frames ERK, {w} frames con xgboost"
            utils.open_dirs(dir_path, second_dir)
            dir_path += "/" + second_dir

            diff_tracker_model = DiffTracker(normalize=normalize_func, drop_columns=drop_columns_func,
                                             concat_dfs=utils.concat_dfs, dir_path=dir_path,
                                             diff_window=diff_window, con_windows=w)

            diff_tracker_model.train(diff_df_train=diff_df_train, con_df_train=con_df_train,
                                     diff_df_test=diff_df_test, con_df_test=con_df_test)

            clf, X_train, X_test, y_train, y_test = utils.load_data(dir_path)
            diff_tracker_model.evaluate_clf(X_test, y_test)

            motility = True
            intensity = False

            wt_cols = [wt for wt in range(0, 260, tracks_len)]

            csv_path = path + fr"/data/mastodon/test/Nuclei_2-vertices.csv"
            print(csv_path)
            df = pd.read_csv(csv_path, encoding="ISO-8859-1")
            df = df[df["manual"] == 1]

            tracks = get_tracks_list(df, target=0)[:15]
            prob_ = utils.calc_prob_delta(window=tracks_len, tracks=tracks, clf=clf, X_test=X_test, motility=motility,
                                          visual=intensity,
                                          wt_cols=wt_cols, moving_window=True, aggregate_windows=False,
                                          calc_delta=False)
            pickle.dump(prob_, open(dir_path + "/" + f"df_prob_w={30}, video_num={2}", 'wb'))

            csv_path = path + fr"/data/mastodon/test/Nuclei_3-vertices.csv"
            print(csv_path)
            df = pd.read_csv(csv_path, encoding="ISO-8859-1")
            df = df[df["manual"] == 1]

            tracks = get_tracks_list(df, target=1)[:15]
            prob_ = utils.calc_prob_delta(window=tracks_len, tracks=tracks, clf=clf, X_test=X_test, motility=motility,
                                          visual=intensity,
                                          wt_cols=wt_cols, moving_window=True, aggregate_windows=False,
                                          calc_delta=False)
            pickle.dump(prob_, open(dir_path + "/" + f"df_prob_w={30}, video_num={3}", 'wb'))
