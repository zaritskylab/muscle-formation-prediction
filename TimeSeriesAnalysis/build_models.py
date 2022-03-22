import pickle

import diff_tracker_utils as utils
import pandas as pd
from mast_intensity import get_intensity_measures_df

from diff_tracker import DiffTracker
from imblearn.over_sampling import RandomOverSampler
from intensity_erk_compare import get_tracks_list
from consts import *
if __name__ == '__main__':
    path = cluster_path
    to_run = "motility"

    if to_run == "intensity":
        diff_df_train = get_intensity_measures_df(csv_path=path + csv_path_s5,
                                                  video_actin_path=path + vid_path_s5_actin, window_size=window_size)
        con_df_train = get_intensity_measures_df(csv_path=path + csv_path_s1,
                                                 video_actin_path=path + vid_path_s1_actin, window_size=window_size)
        con_df_test = get_intensity_measures_df(csv_path=path + csv_path_s2,
                                                video_actin_path=path + vid_path_s2_actin, window_size=window_size)
        diff_df_test = get_intensity_measures_df(csv_path=path + csv_path_s3,
                                                 video_actin_path=path + vid_path_s3_actin, window_size=window_size)
        normalize_func = utils.normalize_intensity
        drop_columns_func = utils.drop_columns_intensity

    if to_run == "motility":
        diff_df_train = pd.read_csv(path + csv_path_s5, encoding="cp1252").dropna()
        con_df_train = pd.read_csv(path + csv_path_s1, encoding="cp1252").dropna()
        diff_df_test = pd.read_csv(path + csv_path_s3, encoding="cp1252").dropna()
        con_df_test = pd.read_csv(path + csv_path_s2, encoding="cp1252").dropna()

        normalize_func = utils.normalize_motility
        drop_columns_func = utils.drop_columns_motility

    # diff_windows = [[130, 160], [135, 165], [140, 170], [145, 175], [150, 180], [155, 185]]
    diff_windows = [[140, 170]]
    tracks_len = 30
    con_windows = [

        [[0, 30], [40, 70], [90, 120], [140, 170], [180, 210], [220, 250]],
        # [[10, 40], [40, 70], [70, 100], [100, 130], [130, 160], [160, 190], [190, 220], [220, 250]]

        # [[0, 50], [50, 100], [100, 150], [150, 200], [200, 250]],
        # [[0, 50], [60, 110], [110, 160], [160, 210], [210, 260]],
        # [[10, 60], [60, 110], [110, 160], [160, 210], [210, 260], [150, 180], [180, 210], [210, 240]],
    ]
    for diff_window in diff_windows:
        for w in con_windows:

            dir_path = f"20-03-2022-manual_mastodon_{to_run} shifted tracks"
            second_dir = f"{diff_window} frames ERK, {w} frames con track len {tracks_len}"
            utils.open_dirs(dir_path, second_dir)
            dir_path += "/" + second_dir

            diff_tracker_model = DiffTracker(normalize=normalize_func, drop_columns=drop_columns_func,
                                             concat_dfs=utils.concat_dfs, dir_path=dir_path,
                                             diff_window=diff_window, con_windows=w)

            # diff_tracker_model.train(diff_df_train=diff_df_test, con_df_train=con_df_test,
            #                          diff_df_test=diff_df_train, con_df_test=con_df_train,
            #                          vid_path_dif_train=vid_path_s5_nuc, vid_path_con_train=vid_path_s1_nuc,
            #                          vid_path_dif_test=vid_path_s3_nuc, vid_path_con_test=vid_path_s2_nuc,
            #                          )

            clf, X_train, X_test, y_train, y_test = utils.load_data(dir_path)
            diff_tracker_model.clf = clf
            diff_tracker_model.evaluate_clf(X_test, y_test)

            motility = True
            intensity = False

            wt_cols = [wt for wt in range(0, 260, tracks_len)]

            csv_path = path + csv_path_s2
            print(csv_path)
            df = pd.read_csv(csv_path, encoding="ISO-8859-1")
            df = df[df["manual"] == 1]

            tracks = get_tracks_list(df, target=0)[:14]
            print(len(tracks))

            prob_ = utils.calc_prob_delta(window=tracks_len, tracks=tracks, clf=clf, X_test=X_test, motility=motility,
                                          visual=intensity,
                                          wt_cols=wt_cols, moving_window=True, aggregate_windows=False,
                                          calc_delta=False,
                                          shift_tracks=True, vid_path = vid_path_s2_nuc)
            pickle.dump(prob_, open(dir_path + "/" + f"df_prob_w={30}, video_num={2}", 'wb'))
            #
            # csv_path = path + csv_path_s3
            # print(csv_path)
            # df = pd.read_csv(csv_path, encoding="ISO-8859-1")
            # df = df[df["manual"] == 1]
            #
            # tracks = get_tracks_list(df, target=1)[:30]
            # print(len(tracks))
            # prob_ = utils.calc_prob_delta(window=tracks_len, tracks=tracks, clf=clf, X_test=X_test, motility=motility,
            #                               visual=intensity,
            #                               wt_cols=wt_cols, moving_window=True, aggregate_windows=False,
            #                               calc_delta=False,
            #                               shift_tracks=True, vid_path=vid_path_s3_nuc)
            # pickle.dump(prob_, open(dir_path + "/" + f"df_prob_w={30}, video_num={3}", 'wb'))
