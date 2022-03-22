import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from TimeSeriesAnalysis.diff_tracker_utils import load_data
from TimeSeriesAnalysis.intensity_erk_compare import trim_fusion, get_tracks_list
from TimeSeriesAnalysis.mastodon import open_dirs
from TimeSeriesAnalysis.mastodon_interpretation import calc_motility_measures
import seaborn as sns

if __name__ == '__main__':

    csv_path = fr"../data/mastodon/test/s3_fusion-vertices.csv"
    fusion_times = pd.read_csv(csv_path, encoding="ISO-8859-1")
    fused_df, not_fused_df = trim_fusion(fusion_times)
    fused_tracks = get_tracks_list(fused_df)
    not_fused_tracks = get_tracks_list(not_fused_df)[:30]

    dir_name = f"16-02-2022-manual_mastodon_motility-True_intensity-False"
    second_dir = f"140,170 frames ERK, [[0, 30], [40, 70], [90, 120], [140, 170], [180, 210], [220, 250]] frames con"  # ,40 winsize
    open_dirs(dir_name, second_dir)
    dir_name += "/" + second_dir

    clf, X_train, X_test, y_train, y_test = load_data(dir_name)

    all_data = calc_motility_measures(fused_tracks, 30, clf, X_test)

    fuse_df_motility = pickle.load(open(f"df_prob_fuse_w={30}, video_num={3}", 'rb'))

    fig, axs = plt.subplots(2, 2)
    """motility measurements over confidence"""
    # for track, i in zip(fused_tracks, fuse_df_motility):
    #     motility_measurements = all_data[all_data["Spot track ID"] == track["Spot track ID"].iloc[0]]
    #
    #     confidence = fuse_df_motility.iloc[i]
    #     confidence = confidence[:len(
    #         motility_measurements)]  # confidence[motility_measurements["Spot frame"].min():motility_measurements["Spot frame"].max()] # + 1
    #     confidence = confidence  # * 5 / 60
    #
    #     axs[0, 0].scatter(confidence, motility_measurements["avg_total"])
    #     axs[0, 0].set(xlabel='confidence', ylabel='y-label')
    #     axs[0, 1].scatter(confidence, motility_measurements["linearity"], c='tab:orange')  #
    #     axs[0, 1].set(xlabel='confidence', ylabel='linearity')
    #     axs[1, 0].scatter(confidence, motility_measurements["net_total_distance"], c='tab:green')  # c='tab:green'
    #     axs[1, 0].set(xlabel='confidence', ylabel='net_total_distance')
    #     axs[1, 1].scatter(confidence, motility_measurements["monotonicity"], c="tab:red")
    #     axs[1, 1].set(xlabel='confidence', ylabel='monotonicity')
    # plt.show()

    """fusion times over motility measurements"""
    fusions = []
    a_total = []
    lin = []
    net = []
    mono = []

    data = pd.DataFrame()
    for track, i in zip(fused_tracks, fuse_df_motility):
        motility_measurements = all_data[all_data["Spot track ID"] == track["Spot track ID"].iloc[0]]
        fusion_time = track["Spot frame"].max()
        measures_fusion_time = motility_measurements[motility_measurements["Spot frame"] == fusion_time]
        fusions.append(fusion_time)
    #     data = data.append({"fusion_time": fusion_time, "avg_total": measures_fusion_time["avg_total"],
    #                         'linearity': measures_fusion_time["linearity"],
    #                         "net_total_distance": measures_fusion_time["net_total_distance"],
    #                         "monotonicity": measures_fusion_time["monotonicity"]}, ignore_index=True)
    # data = data.astype(float)

    # plt.figure(figsize=(14, 6))
    # ax1 = plt.subplot(2, 4, 1)
    # ax2 = plt.subplot(2, 4, 2)
    # ax3 = plt.subplot(2, 4, 3)
    # ax4 = plt.subplot(2, 4, 4)
    # axes = [ax1, ax2, ax3, ax4]
    #
    # sns.scatterplot(ax=ax1, x="fusion_time", y="avg_total",palette="crest",data=data, legend="brief")
    # sns.scatterplot(ax=ax2, x="fusion_time", y="linearity",palette="crest",data=data, legend="brief")
    # sns.scatterplot(ax=ax3, x="fusion_time", y="net_total_distance",palette="crest",data=data, legend="brief")
    # sns.scatterplot(ax=ax4, x="fusion_time", y="monotonicity",palette="crest",data=data, legend="brief")
    # plt.show()

    conf = pd.DataFrame()
    for ind, track in zip(fuse_df_motility, fused_tracks):
        fusion_time = track["Spot frame"].max()
        confidence = fuse_df_motility.iloc[ind]
        conf_fusion_time = confidence[fusion_time]
        conf = conf.append({"fusion_time": fusion_time, "confidence": conf_fusion_time}, ignore_index=True)
    fig = plt.Figure(figsize=(8, 8))
    sns.scatterplot(x="fusion_time", y="confidence", palette="crest", data=conf, legend="brief",
                    hue="fusion_time").set_title(
        "cross correlation- confidence & cooredination")
    plt.show()
