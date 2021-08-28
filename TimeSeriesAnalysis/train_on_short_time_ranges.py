import itertools
import os
import pickle
import numpy as np
import joblib
from sklearn.metrics import plot_roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns

from DataPreprocessing.load_tracks_xml import load_tracks_xml
from ts_interpretation import build_pca, feature_importance, plot_roc, plot_pca

from ts_fresh import short_extract_features, extract_distinct_features, train, drop_columns, \
    normalize_tracks, get_prob_over_track, save_data, load_data, get_unique_indexes, get_x_y, get_path

from calc_delta import calc_prob_delta, get_df_delta_sums


def plot_cls_distribution(diff_video_num, con_video_num, end_of_file_name, diff_t_windows):
    # df_sums = get_df_delta_sums(diff_video_num, con_video_num, end_of_file_name)
    df_probs = pd.DataFrame()

    for t_window in diff_t_windows:
        time_frame = f"{diff_t_w[0]},{diff_t_w[1]} frames ERK, 1,120 frames con"
        dir_name = f"manual_tracking_1,3_ motility-{motility}_intensity-{intensity}/{time_frame}"

        df_prob_diff = pickle.load(
            open(dir_name + "/" + f"df_prob_w=999, video_num={diff_video_num}" + end_of_file_name, 'rb'))
        df_prob_con = pickle.load(
            open(dir_name + "/" + f"df_prob_w=999, video_num={con_video_num}" + end_of_file_name, 'rb'))

        df_probs = pd.concat([df_probs, pd.DataFrame(
            {"prob": df_prob_diff, "time_window": str(t_window), "diff_con": "ERK"})])
        df_probs = pd.concat([df_probs, pd.DataFrame(
            {"prob": df_prob_con, "time_window": str(t_window), "diff_con": "con"})])

    sns.set_theme(style="darkgrid")
    # Plot the responses for different events and regions
    sns.lineplot(x="time_window", y="prob", hue="diff_con", style="event", data=df_probs)
    plt.title("Distribution of the probability to be differentiated for each time window classifier")
    plt.ylabel("probability")
    plt.xlabel("time window")
    plt.legend()


if __name__ == '__main__':
    print(
        "Let's go! In this script, we will train random forest + tsfresh,on manual tracked cells (videos 1,3), with short time frames")

    # params
    window = 160
    wt_cols = [wt * 90 for wt in range(0, 950, window)]
    video_num = 1
    motility = False
    intensity = True
    lst_videos = [1, 3]
    con_t_window = [0, 120]
    diff_t_windows = [[0, 120], [80, 200], [160, 280], [240, 360], [320, 440], [400, 520],
                      [480, 600], [560, 680], [640, 760], [720, 840], [800, 920], [880, 925]]

    for diff_t_w in diff_t_windows:
        time_frame = f"{diff_t_w[0]},{diff_t_w[1]} frames ERK, 1,120 frames con"

        # open a new directory to save the outputs in

        dir_name = f"manual_tracking_1,3_ motility-{motility}_intensity-{intensity}/{time_frame}"
        print(dir_name)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        # load ERK's tracks and dataframe
        xml_path = get_path(
            fr"data/tracks_xml/manual_tracking/Experiment1_w1Widefield550_s{video_num}_all_manual_tracking.xml")
        tracks, df = load_tracks_xml(xml_path)

        clf, X_train, X_test, y_train, y_test = load_data(dir_name)

        df_delta_prob = calc_prob_delta(window, tracks, clf, X_test, motility, intensity, wt_cols,
                                        calc_delta=False)
        pickle.dump(df_delta_prob,
                    open(dir_name + "/" + f"df_prob_w={window}, video_num={video_num}", 'wb'))

        # # generate train data & extract features using tsfresh
        # X, y = get_x_y(lst_videos=lst_videos, motility=motility, intensity=intensity, time_window=True,
        #                con_t_window=con_t_window, diff_t_window=diff_t_w)
        # X = short_extract_features(X, y)
        #
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        #
        # # train the classifier
        # clf, report, auc_score = train(X_train, X_test, y_train, y_test)
        #
        # # save the model & train set & test set
        # save_data(dir_name, clf, X_train, X_test, y_train, y_test)
        #
        # # plot ROC curve
        # plot_roc(clf=clf, X_test=X_test, y_test=y_test, path=dir_name)

        # # perform PCA analysis
        # principal_df, pca = build_pca(3, X_test)
        # plot_pca(principal_df, pca, dir_name)

        # # calculate feature importance
        # feature_importance(clf, X_train.columns, dir_name)

        # # save classification report & AUC score
        # txt_file = open(dir_name + '/info.txt', 'a')
        # txt_file.write(f"classification report: {report}\n auc score: {auc_score}")
        # txt_file.close()
