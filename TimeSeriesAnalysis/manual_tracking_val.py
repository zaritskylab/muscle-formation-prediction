import itertools
import os
import pickle
import numpy as np
import joblib
from sklearn.metrics import plot_roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from DataPreprocessing.load_tracks_xml import load_tracks_xml
from TimeSeriesAnalysis.ts_interpretation import build_pca, feature_importance, plot_roc, plot_pca

from TimeSeriesAnalysis.ts_fresh import get_x_y, short_extract_features, extract_distinct_features, train, drop_columns, \
    normalize_tracks, get_prob_over_track, save_data, load_data, get_path, evaluate
from multiprocessing import Process
import math

from Motility.MotilityMeasurements import get_linearity, get_distance, get_total_distance, get_net_total_proportion, \
    get_monotonicity, get_msd
from TimeSeriesAnalysis.interpretation import get_cell_speed


def train_and_eval_procedoure(
        lst_videos, motility, intensity, dir_name, min_length=0, max_length=950, min_time_diff=0, crop_start=False,
        crop_end=False, time_window=False, diff_t_window=None, con_t_windows=None, columns_to_drop=None):
    # generate train data & extract features using tsfresh
    X, y = get_x_y(min_length=min_length, max_length=max_length, min_time_diff=min_time_diff, lst_videos=lst_videos,
                   motility=motility, intensity=intensity, columns_to_drop=columns_to_drop, crop_start=crop_start,
                   crop_end=crop_end, time_window=time_window, diff_t_window=diff_t_window, con_t_windows=con_t_windows)
    extracted_features = extract_features(X, column_id="label", column_sort="t")
    impute(extracted_features)
    features_filtered = select_features(extracted_features, y)
    if features_filtered.empty:
        X = extracted_features
    else:
        X = short_extract_features(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)  # random_state=42

    # Free ram we dont use anymore
    del (X)
    del (y)

    # train the classifier
    clf = train(X_train, y_train)
    save_data(dir_name, clf=clf, X_train=X_train, y_train=y_train)
    del (X_train)
    del (y_train)

    report, auc_score = evaluate(clf, X_test, y_test)

    # save the model & train set & test set
    save_data(dir_name, X_test=X_test, y_test=y_test)

    # load the model & train set & test set
    clf, X_train, X_test, y_train, y_test = load_data(dir_name)

    # plot ROC curve
    plot_roc(clf=clf, X_test=X_test, y_test=y_test, path=dir_name)

    # perform PCA analysis
    principal_df, pca = build_pca(3, X_test)
    plot_pca(principal_df, pca, dir_name)

    # calculate feature importance
    feature_importance(clf, X_train.columns, dir_name)

    # save classification report & AUC score
    txt_file = open(dir_name + '/info.txt', 'a')
    txt_file.write(f"classification report: {report}\n auc score: {auc_score}")
    txt_file.close()


def plot_test_prob_distribution(dir_name, t_window_con, t_window_diff):
    clf, X_train, X_test, y_train, y_test = load_data(dir_name)

    probs = clf.predict_proba(X_test)[:, 1]
    p_list = list(probs)
    y = y_test.copy()
    y_ = np.asarray(y)

    df = pd.DataFrame({"p": probs, "con_diff": y})
    import seaborn as sns
    sns.displot(data=df, x="p", hue="con_diff")

    # markerline, stemlines, baseline = plt.stem(probs, linefmt='orange')
    plt.show()

    # get tracks according to the x_test tracks:
    xml_path = get_path(
        fr"data/tracks_xml/0104/Experiment1_w1Widefield550_s{7}_all_0104.xml")
    _, df_c = load_tracks_xml(xml_path)
    # df_con = df[df["label"].isin(list(X_test.index))]

    # max_val = df["label"].max()
    # indices = sorted(i for i in list(X_test.index) if i >= max_val)
    # indices = np.asarray(indices) - 2 * max_val

    xml_path = get_path(
        fr"data/tracks_xml/0104/Experiment1_w1Widefield550_s{5}_all_0104.xml")
    _, df_d = load_tracks_xml(xml_path)
    # df_diff = df_d[df_d["label"].isin(list(indices))]

    hist_p_c_120 = get_prob_histogram_list(df_c, clf, t_window_con[1] - t_window_con[0], X_test, t_window_con)
    hist_p_d_120 = get_prob_histogram_list(df_d, clf, t_window_con[1] - t_window_con[0], X_test, t_window_con)
    plot_histogram(hist_p_c_120, hist_p_d_120, "distribution 0-120, videos 5,7", dir_name)

    hist_p_c_640 = get_prob_histogram_list(df_c, clf, t_window_diff[1] - t_window_diff[0], X_test, t_window_diff)
    hist_p_d_640 = get_prob_histogram_list(df_d, clf, t_window_diff[1] - t_window_diff[0], X_test, t_window_diff)
    plot_histogram(hist_p_c_640, hist_p_d_640, "distribution 550-650, videos 5,7", dir_name)


def plot_histogram(l_c, l_d, title, dir_name):
    plt.stem(l_c, linefmt='-')
    markerline, stemlines, baseline = plt.stem(l_d, linefmt='orange')
    markerline.set_markerfacecolor('orange')
    plt.title(title)
    plt.xlabel("prob (%)")
    plt.ylabel("count- how many cells have the prob x of being differentiated")
    plt.xticks(range(0, 10, 1), labels=np.arange(0, 100, 10))
    plt.legend(["Control", "ERK"])
    plt.show()
    plt.savefig(
        dir_name + "/" + f"{title}.png")


def get_prob_histogram_list(df, clf, window, X_test, t_windows):
    probs = get_prob(df, clf, window, X_test, t_windows)
    counter = np.zeros(10)
    if probs:
        for p in probs:
            if p:
                bin = math.floor(p * 10)
                counter[bin] += 1
    return list(counter)


def get_prob(df, clf, _window, X_test, t_windows):
    true_probs = []
    for label, track in df.groupby('label'):
        cropped_track = track[(track["t_stamp"] <= t_windows[1]) & (track["t_stamp"] >= t_windows[0])]
        true_prob = get_prob_over_track(clf, cropped_track, _window, X_test)
        true_probs.append(None if not true_prob else true_prob[0])
    return true_probs


def train_all_but_one_feature(lst_videos, motility, intensity):
    # open a new directory to save the outputs in
    dir_name = f"all but one feature manual_tracking_1,3"
    print(dir_name)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    # train on all but one column:
    intensity_features = ['median_intensity', 'min_intensity', 'max_intensity', 'mean_intensity',
                          'total_intensity', 'std_intensity', 'contrast', 'snr', 'w', 'q']
    for col in intensity_features:
        dir_all_but_one = f"{dir_name}/all but {col}"
        print(dir_all_but_one)
        if not os.path.exists(dir_all_but_one):
            os.mkdir(dir_all_but_one)

        # p = Process(target=train_and_eval_procedoure,
        #             args=(lst_videos, motility, intensity, [col]))
        # p.start()
        train_and_eval_procedoure(dir_name=dir_all_but_one, lst_videos=lst_videos, motility=motility,
                                  intensity=intensity, columns_to_drop=[col])


def train_one_feature_at_a_time():
    # train on one feature:
    # open a new directory to save the outputs in
    dir_name = f"one feature manual_tracking_1,3"
    print(dir_name)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    # tm_columns = ['x', 'y', 'mean_intensity', 'w', 'q',
    #               'median_intensity', 'min_intensity', 'max_intensity', 'total_intensity',
    #               'std_intensity', 'contrast', 'snr']
    tm_columns = ['median_intensity', 'min_intensity', 'max_intensity', 'total_intensity',
                  'std_intensity', 'contrast', 'snr']

    for col in tm_columns:
        dir_name_one_col = f"{dir_name}/{col}"
        print(dir_name_one_col)
        if not os.path.exists(dir_name_one_col):
            os.mkdir(dir_name_one_col)

        cols_to_drop = tm_columns.copy()
        cols_to_drop.remove(col)
        # p = Process(target=train_and_eval_procedoure,
        #             args=(lst_videos, motility, intensity, cols_to_drop))
        # p.start()
        train_and_eval_procedoure(dir_name=dir_name_one_col, lst_videos=lst_videos, motility=motility,
                                  intensity=intensity, columns_to_drop=cols_to_drop)


def calc_motility_measures(video_num):
    # xml_path = get_path(fr"data/tracks_xml/260721/S{video_num}_Nuclei.xml")
    xml_path = get_path(fr"../data/tracks_xml/0104/Experiment1_w1Widefield550_s{video_num}_all_0104.xml")
    clf, X_train, X_test, y_train, y_test = load_data(dir_name)
    # load tracks and dataframe
    tracks, _df = load_tracks_xml(xml_path)

    all_data = pd.DataFrame()
    for track in tracks[:3]:
        if len(track) > window_size * 2:
            # speed
            avg_x, avg_y, avg_total = get_cell_speed(track, window_size)

            linearity = []
            # linearity = [get_linearity(track[i:i + window_size]) for i in range(0, len(track), window_size)]
            net_distance = []
            total_distance = []
            net_total_distance = []
            monotonicity = []
            msd_alpha = []
            for i in range(0, len(track), window_size):
                track_portion = track[i:i + window_size]
                # linearity (to calculate persistence)
                linearity.append(get_linearity(track_portion))
                net_distance.append(
                    get_distance(x1=track_portion[track_portion["t"] == int(np.min(track_portion["t"]))]["x"].values[0],
                                 y1=track_portion[track_portion["t"] == int(np.min(track_portion["t"]))]["y"].values[0],
                                 x2=track_portion[track_portion["t"] == int(np.max(track_portion["t"]))]["x"].values[0],
                                 y2=track_portion[track_portion["t"] == int(np.max(track_portion["t"]))]["y"].values[
                                     0]))
                total_distance.append(get_total_distance(track_portion))
                for net, tot in zip(net_distance, total_distance):
                    net_total_distance.append(get_net_total_proportion(net, tot))
                monotonicity.append(get_monotonicity(track_portion))
                msd_alpha.append(get_msd(track_portion))

            # calculate list of probabilities per window
            track = drop_columns(track, motility=motility, intensity=intensity)
            # track = normalize_tracks(track, motility=motility, intensity=intensity)
            track_diff_confidence = get_prob_over_track(clf=clf, track=track, window=window_size, features_df=X_test)
            length = len(track_diff_confidence)
            tmp_df = pd.DataFrame(
                {"confidence": track_diff_confidence, "avg_x": avg_x[:length], "avg_y": avg_y[:length],
                 "avg_total": avg_total[:length], "linearity": linearity[:length],
                 "net_distance": net_distance[:length], "total_distance": total_distance[:length],
                 "net_total_distance": net_total_distance[:length], "monotonicity": monotonicity[:length],
                 "msd_alpha": msd_alpha[:length]})
            all_data = pd.concat([all_data, tmp_df], axis=0)
    pickle.dump(all_data, open(dir_name + "/" + f"all_data , video {video_num}", 'wb'))


if __name__ == '__main__':

    print(
        "Let's go! In this script, we will train random forest + tsfresh, on the 210726 experiment")

    # params
    motility = True
    intensity = False
    window_size = 30
    wt_cols = [wt * 300 for wt in range(0, 350, window_size)]
    lst_videos = [3, 8, 4, 11]  # 3,4 - control; 8,11 - ERKi
    t_windows_con = [[0, 30], [140, 170], [180, 210]]  # , [140, 170], [180, 210]
    t_window_diff = [140, 170]

    dir_name = f"tmp_ motility-{motility}_intensity-{intensity}"
    time_frame = f"{t_window_diff[0]},{t_window_diff[1]} frames ERK, {t_windows_con} frames con"

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    print(dir_name + "/" + time_frame)
    if not os.path.exists(dir_name + "/" + time_frame):
        os.mkdir(dir_name + "/" + time_frame)

    # train_and_eval_procedoure(lst_videos=lst_videos, motility=motility, intensity=intensity,
    #                           dir_name=dir_name + "/" + time_frame, time_window=True, diff_t_window=t_window_diff,
    #                           con_t_windows=t_windows_con)

    dir_name = dir_name + "/" + time_frame
    dir_name = f"tmp_  motility-True_intensity-False/140,170 frames ERK, {t_windows_con} frames con"

    vid_3_pkl = pickle.load(open(dir_name + "/" + "all_data , video 3", 'rb'))

    print("video 3")
    calc_motility_measures(video_num=3)
    print("video 8")
    calc_motility_measures(video_num=8)
