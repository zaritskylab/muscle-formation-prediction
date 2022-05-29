import pickle

import pandas as pd
import sys
from diff_tracker_utils import calc_prob_delta
from calc_delta_mastodon import  plot_avg_diff_prob
from mast_intensity import plot_intensity_over_time, \
     load_data, open_dirs, get_intensity_measures_df_df


def get_fusion_time(fusion_times, label):
    lab_df = fusion_times[fusion_times["Spot track ID"] == label]
    cols = list(lab_df.columns)
    cols.remove("manual")
    cols.remove("last_position")
    fusion_time = 0
    for col in cols:
        if lab_df[col].mean() == 1:
            fusion_time = int(col)
    return fusion_time


def get_single_measure_vector_df(intentsity_measures_df, measure_name, fusion_times):
    single_measure_df = pd.DataFrame(columns=[i for i in range(262)])
    for lable, lable_df in intentsity_measures_df.groupby("label"):
        fusion_t = get_fusion_time(fusion_times, int(lable))
        fusion_t = fusion_t if fusion_t != 0 else lable_df["frame"].max()
        frame_label_df = lable_df[["frame", measure_name]]
        frame_label_df.index = frame_label_df["frame"].astype(int)
        frame_label_df = frame_label_df[~frame_label_df.index.duplicated()]
        frame_label_df = frame_label_df[frame_label_df["frame"] < fusion_t]
        frame_label_df["mean"] = frame_label_df["mean"] - frame_label_df["mean"].iloc[0]
        single_measure_df = single_measure_df.append(frame_label_df["mean"])

    single_measure_df.index = [i for i in range(len(single_measure_df))]
    return single_measure_df


# pos_df = int_measures_s3_fusion[int_measures_s3_fusion["label"].isin(
#     [8987, 19929, 506, 8787, 11135, 8966, 22896, 8594, 14202, 14646, 20010, 7641, 17902, 20043])]
# neg_df = int_measures_s3_fusion[
#     int_measures_s3_fusion["label"].isin([12421, 20455, 26219, 27930, 15645, 14625, 17513, 2590, 30007, 15281, 29878])]


def trim_fusion(df):
    fused_df = pd.DataFrame()
    not_fused_df = pd.DataFrame()
    for label, label_df in df.groupby("Spot track ID"):
        fusion_t = get_fusion_time(label_df, int(label))
        if fusion_t != 0:  # fused
            label_df = label_df[label_df["Spot frame"] <= fusion_t]
            fused_df = fused_df.append(label_df)
        else:  # not fused at the end of the video
            not_fused_df = not_fused_df.append(label_df)
    return fused_df, not_fused_df


def get_tracks_list(int_df, target):
    int_df['target'] = target
    tracks = list()
    for label, labeld_df in int_df.groupby('Spot track ID'):
        tracks.append(labeld_df)
    return tracks


def run(dir_name, fused_df, not_fused_df, bounded_window_size):
    clf, X_train, X_test, y_train, y_test = load_data(dir_name)

    fused_int_df = get_intensity_measures_df_df(fused_df, path + fr"/data/videos/test/S3_Actin.tif",
                                                bounded_window_size)
    not_fused_int_df = get_intensity_measures_df_df(not_fused_df,
                                                    path + fr"/data/videos/test/S3_Actin.tif",
                                                    bounded_window_size)
    fused_tracks = get_tracks_list(fused_int_df)
    not_fused_tracks = get_tracks_list(not_fused_int_df)

    prob_fuse = calc_prob_delta(30, fused_tracks, clf, X_test, False, visual=True, wt_cols=wt_cols,
                                calc_delta=False,
                                moving_window=True)
    pickle.dump(prob_fuse, open(dir_name + "/" + f"df_prob_fuse_w={30}, video_num={3}", 'wb'))

    prob_not_fuse = calc_prob_delta(30, not_fused_tracks, clf, X_test, False, visual=True, wt_cols=wt_cols,
                                    calc_delta=False,
                                    moving_window=True)
    pickle.dump(prob_not_fuse, open(dir_name + "/" + f"df_prob_not_fuse_w={30}, video_num={3}", 'wb'))


if __name__ == '__main__':
    # # load the model & train set & test set
    # wt_cols = [wt for wt in range(0, 260, 30)]
    # cluster_path = "muscle-formation-diff"
    # local_path = ".."
    # path = cluster_path
    #
    # bounded_window_size = 40
    # diff_window = [140, 170]
    # # con_windows = [[[0, 30], [40, 70], [90, 120], [140, 170], [180, 210], [220, 250]],
    # #                [[0, 30], [40, 70], [70, 100], [110, 140], [150, 180], [190, 220], [238, 258]],
    # #                [[0, 30], [30, 60], [60, 90], [90, 120], [120, 150], [150, 180], [180, 210], [210, 240]],
    # #                [[10, 40], [40, 70], [70, 100], [100, 130], [130, 160], [160, 190], [190, 220], [220, 250]]]
    #
    # con_windows = [
    #     [[70, 100], [130, 100], [140, 170]],
    #     # [[70, 100], [100, 130], [140, 170], [180, 210], [210, 240]],
    #
    #                ]
    #
    # motility_or_intensity = "motility"
    # to_run = "not_fuse"
    # print(motility_or_intensity, to_run)
    #
    # csv_path = path + fr"/data/mastodon/test/s3_fusion-vertices.csv"
    # fusion_times = pd.read_csv(csv_path, encoding="ISO-8859-1")
    # fused_df, not_fused_df = trim_fusion(fusion_times)
    # fused_tracks = get_tracks_list(fused_df, target=1)
    # not_fused_tracks = get_tracks_list(not_fused_df, target=1)
    #
    # if motility_or_intensity == "motility":
    #     dir_path = f"27-02-2022-manual_mastodon_motility"
    #     motility = True
    #     intensity = False
    # if motility_or_intensity == "intensity":
    #     dir_path = f"27-02-2022-manual_mastodon_intensity"
    #     motility = False
    #     intensity = True
    #
    # for w in con_windows:
    #     second_dir = f"{diff_window} frames ERK, {w} frames con"
    #     iteration_dir_path = dir_path + "/" + second_dir
    #
    #     clf, X_train, X_test, y_train, y_test = load_data(iteration_dir_path)
    #
    #     if to_run == "fuse":
    #         if motility_or_intensity == "intensity":
    #             fused_int_df = get_intensity_measures_df_df(fused_df, path + fr"/data/videos/test/S3_Actin.tif",
    #                                                         bounded_window_size)
    #             fused_tracks = get_tracks_list(fused_int_df, target=1)[:15]
    #         prob_fuse = calc_prob_delta(window=30, tracks=fused_tracks, clf=clf, X_test=X_test, motility=motility,
    #                                     visual=intensity,
    #                                     wt_cols=wt_cols, moving_window=True, aggregate_windows=False, calc_delta=False)
    #         pickle.dump(prob_fuse, open(iteration_dir_path + "/" + f"df_prob_fuse_w={30}, video_num={3}", 'wb'))
    #
    #     elif to_run == "not_fuse":
    #         if motility_or_intensity == "intensity":
    #             not_fused_int_df = get_intensity_measures_df_df(not_fused_df,
    #                                                             path + fr"/data/videos/test/S3_Actin.tif",
    #                                                             bounded_window_size)
    #             not_fused_tracks = get_tracks_list(not_fused_int_df, target=1)[:15]
    #
    #         prob_not_fuse = calc_prob_delta(window=30, tracks=not_fused_tracks, clf=clf, X_test=X_test,
    #                                         motility=motility,
    #                                         visual=intensity,
    #                                         wt_cols=wt_cols, moving_window=True, aggregate_windows=False,
    #                                         calc_delta=False)
    #         pickle.dump(prob_not_fuse, open(iteration_dir_path + "/" + f"df_prob_not_fuse_w={30}, video_num={3}", 'wb'))
    #
    #     elif to_run == "S2":
    #         if motility_or_intensity == "intensity":
    #             df = get_intensity_measures_df(csv_path=path + r"/data/mastodon/test/Nuclei_2-vertices.csv",
    #                                            video_actin_path=path + r"/data/videos/test/S2_Actin.tif",
    #                                            window_size=bounded_window_size)
    #         else:
    #             csv_path = path + fr"/data/mastodon/test/Nuclei_2-vertices.csv"
    #             print(csv_path)
    #             df = pd.read_csv(csv_path, encoding="ISO-8859-1")
    #             df = df[df["manual"] == 1]
    #
    #         tracks = get_tracks_list(df)
    #         prob_ = calc_prob_delta(window=30, tracks=tracks, clf=clf, X_test=X_test, motility=motility,
    #                                 visual=intensity,
    #                                 wt_cols=wt_cols, moving_window=True, aggregate_windows=False, calc_delta=False)
    #         pickle.dump(prob_, open(iteration_dir_path + "/" + f"df_prob_w={30}, video_num={2}", 'wb'))
    #
    #     elif to_run == "S3":
    #         if motility_or_intensity == "intensity":
    #             df = get_intensity_measures_df(csv_path=path + r"/data/mastodon/test/Nuclei_3-vertices.csv",
    #                                            video_actin_path=path + r"/data/videos/test/S3_Actin.tif",
    #                                            window_size=bounded_window_size)
    #         else:
    #             csv_path = path + fr"/data/mastodon/test/Nuclei_3-vertices.csv"
    #             print(csv_path)
    #             df = pd.read_csv(csv_path, encoding="ISO-8859-1")
    #             df = df[df["manual"] == 1]
    #
    #         tracks = get_tracks_list(df)
    #         prob_ = calc_prob_delta(window=30, tracks=tracks, clf=clf, X_test=X_test, motility=motility,
    #                                 visual=intensity,
    #                                 wt_cols=wt_cols, moving_window=True, aggregate_windows=False, calc_delta=False)
    #         pickle.dump(prob_, open(iteration_dir_path + "/" + f"df_prob_w={30}, video_num={3}", 'wb'))

    """PLOT"""
    import numpy as np
    import matplotlib.pyplot as plt




    # def get_delta_prob(df):
    #     delta_df = pd.DataFrame(columns=[i for i in range(1,260)])
    #     for i in range(len(df)):
    #         list_row = list(df.iloc[i])
    #         prob = [(list_row[i] - list_row[i - 1]) for i in range(1, len(list_row))]
    #         delta_df.loc[len(delta_df)] = prob
    #     delta_df[0] = None
    #     return delta_df
    #
    # delta_fuse_df = get_delta_prob(fuse_df)
    # delta_not_fuse_df = get_delta_prob(not_fuse_df)
    # plot_avg_conf(delta_fuse_df, "DarkOrange", "Orange")
    # plot_avg_conf(delta_not_fuse_df, "slategrey", "slategrey")
    # plt.legend(["fused avg","fused std", "not fused avg", "not fused std"])
    # plt.xlabel("time (h)")
    # plt.ylabel("avg delta in confidence")
    # plt.title("avg delta of differentiation confidence over time (motility)")
    # plt.plot([i * 5 / 60 for i in range(260)], [0 for i in range(260)], color="black", linestyle="--")
    # plt.show()
    #
