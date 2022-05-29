import pickle

import diff_tracker_utils as utils
import pandas as pd

from diff_tracker import DiffTracker
from imblearn.over_sampling import RandomOverSampler
import consts


def get_intensity_df(path, tagged_csv, csv_all, vid_path_actin, winsize, local_density):
    df_all_tracks, tracks_s_train = utils.get_tracks(path + csv_all, target=1)

    df_tagged = pd.read_csv(path + tagged_csv, encoding="ISO-8859-1")
    df_tagged = df_tagged[df_tagged["manual"] == 1]
    df_tagged = df_all_tracks[df_all_tracks["manual"] == 1]

    # df_tagged = df_tagged[(df_tagged["Spot frame"] > 138) & df_tagged["Spot track ID"].isin(df_tagged["Spot track ID"][5:100])]  # TODO

    df_tagged = utils.add_features_df(df_tagged, df_all_tracks, local_density=local_density)
    df_tagged = utils.get_intensity_measures_df(df=df_tagged,
                                                video_actin_path=path + vid_path_actin,
                                                window_size=winsize, local_density=local_density)
    return df_tagged


def get_to_run(to_run, local_density=False):
    if to_run == "intensity":
        diff_df_train = get_intensity_df(path, consts.csv_path_s5, consts.csv_all_s5, consts.vid_path_s5_actin,
                                         consts.window_size,
                                         local_density)
        con_df_train = get_intensity_df(path, consts.csv_path_s1,consts.csv_all_s1, consts.vid_path_s1_actin, consts.window_size,
                                        local_density)
        con_df_test = get_intensity_df(path, consts.csv_path_s2,consts.csv_all_s2, consts.vid_path_s2_actin, consts.window_size,
                                       local_density)
        diff_df_test = get_intensity_df(path, consts.csv_path_s3,consts.csv_all_s3, consts.vid_path_s3_actin, consts.window_size,
                                        local_density)

        normalize_func = utils.normalize_intensity
        drop_columns_func = utils.drop_columns_intensity

    if to_run == "motility":
        diff_df_train, tracks_s_train = utils.get_tracks(path + consts.csv_all_s5, target=1)
        con_df_train, tracks_s_train = utils.get_tracks(path + consts.csv_all_s1, target=0)

        con_df_test, tracks_s_train = utils.get_tracks(path + consts.csv_all_s2, target=0)
        diff_df_test, tracks_s_train = utils.get_tracks(path + consts.csv_all_s3, target=1)

        normalize_func = utils.normalize_motility
        drop_columns_func = utils.drop_columns_motility

    return diff_df_train, con_df_train, con_df_test, diff_df_test, normalize_func, drop_columns_func


if __name__ == '__main__':
    diff_window = [140, 170]
    tracks_len = 30
    con_windows = [[0, 30], [40, 70], [90, 120], [140, 170], [180, 210], [220, 250]]

    path = consts.cluster_path
    to_run = "intensity"
    motility = False
    intensity = True
    local_density = True

    dir_path = f"20-03-2022-manual_mastodon_{to_run} local density"
    second_dir = f"{diff_window} frames ERK, {con_windows} frames con track len {tracks_len}"
    utils.open_dirs(dir_path, second_dir)
    dir_path += "/" + second_dir

    diff_df_train, con_df_train, con_df_test, diff_df_test, normalize_func, drop_columns_func = get_to_run(to_run,
                                                                                                           local_density)

    diff_tracker_model = DiffTracker(normalize=normalize_func, drop_columns=drop_columns_func,
                                     concat_dfs=utils.concat_dfs, dir_path=dir_path,
                                     diff_window=diff_window, con_windows=con_windows)
    print("start training")
    diff_tracker_model.train(
        # diff_df_train=diff_df_test, con_df_train=con_df_test,
        # diff_df_test=diff_df_train, con_df_test=con_df_train,
        diff_df_train=diff_df_train, con_df_train=con_df_train,
        diff_df_test=diff_df_test, con_df_test=con_df_test,

        vid_path_dif_train=consts.vid_path_s5_nuc,
        vid_path_con_train=consts.vid_path_s1_nuc,
        vid_path_dif_test=consts.vid_path_s3_nuc, vid_path_con_test=consts.vid_path_s2_nuc,
        local_density=local_density)

    clf, X_train, X_test, y_train, y_test = utils.load_data(dir_path)
    diff_tracker_model.clf = clf
    diff_tracker_model.evaluate_clf(X_test, y_test)
