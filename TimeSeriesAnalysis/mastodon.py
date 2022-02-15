import glob
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features, select_features, extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute
import os

import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore")


def concat_dfs(lst_videos, csv_path, diff_t_window=None, con_t_windows=None):
    erk_videos = [5, 8, 3]
    control_videos = [1, 2]
    total_df = pd.DataFrame()
    max_val = 0
    for video_name in lst_videos:
        i = int(list(filter(str.isdigit, video_name))[0])
        _csv_path = csv_path + fr"/Nuclei_{i}-vertices.csv"

        df = pd.read_csv(_csv_path, encoding="cp1252")
        df = df.dropna()

        # get only manual tracked cells:
        df = df[df["manual"] == 1]

        labels_to_keep = []
        diff_start, diff_end = diff_t_window
        window_size = diff_end - diff_start

        if i in erk_videos:
            # Cut the needed time window
            df = df[(df["Spot frame"] >= diff_start) & (
                    df["Spot frame"] < diff_end)]


        elif i in control_videos:  # control video
            # Cut the needed time window
            my_labels = []
            control_df = pd.DataFrame()
            new_label = max(df['Spot track ID'].unique()) + 1
            for start, end in con_t_windows:
                tmp_df = df[(df["Spot frame"] >= start) & (df["Spot frame"] < end)]
                for label, label_df in tmp_df.groupby('Spot track ID'):
                    if len(label_df) == window_size:
                        new_label += 1
                        label_df["Spot track ID"] = new_label
                        control_df = control_df.append(label_df)
            #         if len(df[df["Spot track ID"] == label]) >= window_size:
            #             # if label in my_labels:
            #             new_label = max(df['Spot track ID'].unique()) + 1
            #             df.loc[((df["Spot track ID"] == label) & (df["Spot frame"] >= start) & (
            #                     df["Spot frame"] < end)), "Spot track ID"] = new_label
            #             # label = new_label
            #             my_labels.append(new_label)
            # df = df[df['Spot track ID'].isin(my_labels)]
            df = control_df.copy()

        for label, label_df in df.groupby('Spot track ID'):
            if len(label_df) >= window_size:
                labels_to_keep.append(label)
        df = df[df['Spot track ID'].isin(labels_to_keep)]

        df["Spot track ID"] = df["Spot track ID"] + max_val
        max_val = df["Spot track ID"].max() + 1
        target = True if i in erk_videos else False

        df['target'] = np.array([target for i in range(len(df))])
        total_df = pd.concat([total_df, df], ignore_index=True)

    return total_df


def extract_distinct_features(df, feature_list):
    df = extract_features(df, column_id='Spot track ID', column_sort="Spot frame") #, show_warnings=False
    impute(df)
    return df[feature_list]


def get_prob_over_track(clf, track, window, features_df, moving_window=False, aggregate_windows=False):
    '''
    Returns a list of the probability of being differentiated, for each track portion
    :param clf: classifier
    :param track: cell's track
    :param window: the size of the track's portion
    :param features_df: dataframe to take its features, fir using the same features on the tested data
    :return: list of probabilities
    '''
    true_prob = []

    step_size = 1 if moving_window or aggregate_windows else window

    if aggregate_windows:
        for i in range(1, len(track), step_size):
            track_portion = track[0:i * window]
            X = extract_distinct_features(df=track_portion, feature_list=features_df.columns)
            probs = clf.predict_proba(X)
            true_prob.append(probs[0][1])
            print(f"track portion: [{0}:{i * window}]")
            print(clf.classes_)
            print(probs)
            if len(track_portion) >= len(track):
                return true_prob

    else:
        for i in range(0, len(track), step_size):
            if i + window > len(track):
                break
            track_portion = track[i:i + window]
            X = extract_distinct_features(df=track_portion, feature_list=features_df.columns)
            probs = clf.predict_proba(X)
            true_prob.append(probs[0][1])
            print(f"track portion: [{i}:{i + window}]")
            print(clf.classes_)
            print(probs)

    return true_prob


def drop_columns(df, motility, visual):
    to_keep = ['Spot frame', 'Spot track ID', 'target']  # , 'Track N spots'
    visual_features = ['Spot center intensity (Counts)', 'Detection quality', 'Spot intensity Mean ch1 (Counts)',
                       'Spot intensity Std ch1 (Counts)', 'Spot intensity Min ch1 (Counts)',
                       'Spot intensity Max ch1 (Counts)', 'Spot intensity Median ch1 (Counts)',
                       'Spot intensity Sum ch1 (Counts)', 'Spot quick mean (Counts)', 'Spot radius (µm)', ]
    motility_features = ['Spot position X (µm)', 'Spot position Y (µm)', ]

    to_keep.extend(visual_features) if visual else to_keep.extend([])
    to_keep.extend(motility_features) if motility else to_keep.extend([])
    to_keep = [value for value in to_keep if value in df.columns]
    return df[to_keep]


def normalize_tracks(df, motility, visual):
    if motility:
        if 'Spot position X (µm)' in df.columns:
            for label in df['Spot track ID']:
                to_reduce_x = df[df['Spot track ID'] == label].sort_values(by=['Spot frame']).iloc[0][
                    "Spot position X (µm)"]

                df.loc[df['Spot track ID'] == label, 'Spot position X (µm)'] = df[df['Spot track ID'] == label][
                    'Spot position X (µm)'].apply(lambda num: num - to_reduce_x)

        if 'Spot position Y (µm)' in df.columns:
            for label in df['Spot track ID']:
                to_reduce_y = df[df['Spot track ID'] == label].sort_values(by=['Spot frame']).iloc[0][
                    "Spot position Y (µm)"]

                df.loc[df['Spot track ID'] == label, 'Spot position Y (µm)'] = df[df['Spot track ID'] == label][
                    'Spot position Y (µm)'].apply(lambda num: num - to_reduce_y)

    if visual:
        columns = [e for e in list(df.columns) if e not in ('Spot frame', 'Spot track ID', 'target')]

        # create a scaler
        scaler = StandardScaler()
        # transform the feature
        df[columns] = scaler.fit_transform(df[columns])

    return df


def get_unique_indexes(y):
    idxs = y.index.unique()
    lst = [y[idx] if isinstance(y[idx], np.bool_) else y[idx].iloc[0] for idx in idxs]
    y_new = pd.Series(lst, index=idxs).sort_index()
    return y_new


def train(X_train, y_train):
    clf = RandomForestClassifier(max_depth=8)
    clf.fit(X_train, y_train, )
    return clf


def prep_data(lst_videos, csv_path, motility, visual, diff_t_window, con_t_windows):
    df = concat_dfs(lst_videos, csv_path, diff_t_window, con_t_windows)
    df = drop_columns(df, motility, visual)
    df = normalize_tracks(df, motility, visual)

    df = df.sample(frac=1).reset_index(drop=True)

    y = pd.Series(df['target'])
    y.index = df['Spot track ID']
    y = get_unique_indexes(y)
    df = df.drop("target", axis=1)
    return df, y


def save_data(dir_name, clf=None, X_train=None, X_test=None, y_train=None, y_test=None):
    # save the model & train set & test set
    if X_train is not None:
        pickle.dump(X_train, open(dir_name + "/" + "X_train", 'wb'))
    if X_test is not None:
        pickle.dump(X_test, open(dir_name + "/" + "X_test", 'wb'))
    if y_test is not None:
        pickle.dump(y_test, open(dir_name + "/" + "y_test", 'wb'))
    if y_train is not None:
        pickle.dump(y_train, open(dir_name + "/" + "y_train", 'wb'))
    if clf is not None:
        joblib.dump(clf, dir_name + "/" + "clf.joblib")


def load_data(dir_name):
    # load the model & train set & test set
    clf = joblib.load(dir_name + "/clf.joblib")
    X_train = pickle.load(open(dir_name + "/" + "X_train", 'rb'))
    X_test = pickle.load(open(dir_name + "/" + "X_test", 'rb'))
    y_train = pickle.load(open(dir_name + "/" + "y_train", 'rb'))
    y_test = pickle.load(open(dir_name + "/" + "y_test", 'rb'))
    return clf, X_train, X_test, y_train, y_test


def evaluate(clf, X_test, y_test):
    predicted = cross_val_predict(clf, X_test, y_test, cv=5)
    report = classification_report(y_test, predicted)
    auc_score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(report)
    print(auc_score)
    return report, auc_score


def plot_roc(clf, X_test, y_test, path):
    # plt.figure(figsize=(20, 6))
    # roc curve for models
    fpr1, tpr1, thresh1 = roc_curve(y_test, clf.predict_proba(X_test)[:, 1], pos_label=1)

    # roc curve for tpr = fpr
    random_probs = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

    plt.style.use('seaborn')
    # plot roc curves
    plt.plot(fpr1, tpr1, linestyle='--', color='orange', label='Random Forest')
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    # title
    plt.title('ROC curve')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')

    plt.legend(loc='best')
    plt.savefig(path + "/" + 'ROC', dpi=300)
    plt.show()


def build_pca(num_of_components, df):
    '''
    The method creates component principle dataframe, with num_of_components components
    :param num_of_components: number of desired components
    :param df: encoded images
    :return: PCA dataframe
    '''
    pca = PCA(n_components=num_of_components)
    principal_components = pca.fit_transform(df)
    colomns = ['principal component {}'.format(i) for i in range(1, num_of_components + 1)]
    principal_df = pd.DataFrame(data=principal_components, columns=colomns)
    return principal_df, pca


def plot_pca(principal_df, pca, path):
    '''
    The method plots the first 3 dimensions of a given PCA
    :param principal_df: PCA dataframe
    :return: no return value
    '''
    variance = pca.explained_variance_ratio_
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="principal component 1", y="principal component 2",
        hue='principal component 3',
        palette=sns.color_palette("hls", len(principal_df['principal component 3'].unique())),
        data=principal_df,
        legend=False,
        alpha=0.3
    )
    plt.xlabel(f"PC1 ({variance[0]}) %")
    plt.ylabel(f"PC2 ({variance[1]}) %")
    plt.title("PCA")
    plt.savefig(path + "/pca.png")
    plt.show()


def feature_importance(clf, feature_names, path):
    # Figure Size
    top_n = 30
    fig, ax = plt.subplots(figsize=(16, 9))

    sorted_idx = clf.feature_importances_.argsort()

    ax.barh(feature_names[sorted_idx[-top_n:]], clf.feature_importances_[sorted_idx[-top_n:]])
    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=50)

    plt.xlabel("Random Forest Feature Importance")
    plt.title('Feature Importance Plot')
    plt.savefig(path + "/feature importance.png")
    plt.show()


def train_evaluate(lst_videos, csv_path, motility, visual, dir_name, diff_t_window, con_t_windows):
    train_path = csv_path + "/train"
    test_path = csv_path + "/test"

    lst_videos_train = glob.glob(train_path + "/*")
    lst_videos_test = glob.glob(test_path + "/*")

    X_train, y_train = prep_data(lst_videos_train, train_path, motility, visual, diff_t_window, con_t_windows)

    extracted_features = extract_features(X_train, column_id="Spot track ID", column_sort="Spot frame",
                                          show_warnings=False)

    impute(extracted_features)
    features_filtered = select_features(extracted_features, y_train, show_warnings=False)
    if features_filtered.empty:
        X_train = extracted_features
    else:
        X_train = extract_relevant_features(X_train, y_train, column_id="Spot track ID", column_sort='Spot frame',
                                            show_warnings=False)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)  # random_state=42

    X_test, y_test = prep_data(lst_videos_test, test_path, motility, visual, diff_t_window, con_t_windows)
    extracted_features = extract_features(X_test, column_id="Spot track ID", column_sort="Spot frame",
                                          show_warnings=False)
    impute(extracted_features)
    X_test = extracted_features[X_train.columns]

    save_data(dir_name, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    # train the classifier
    clf = train(X_train, y_train)
    save_data(dir_name, clf=clf)

    del (X_train)
    del (y_train)

    report, auc_score = evaluate(clf, X_test, y_test)

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


def open_dirs(main_dir, inner_dir):
    if not os.path.exists(main_dir):
        os.mkdir(main_dir)
    print(main_dir + "/" + inner_dir)
    if not os.path.exists(main_dir + "/" + inner_dir):
        os.mkdir(main_dir + "/" + inner_dir)


def calc_prob_delta(window, tracks, clf, X_test, motility, intensity, wt_cols, moving_window=False,
                    aggregate_windows=False, calc_delta=True):
    df_prob = pd.DataFrame(columns=wt_cols)

    for ind, t in enumerate(tracks):
        track = tracks[ind]
        if window == 926:
            _window = len(track)
        elif len(track) < window:
            continue
        else:
            _window = window

        step_size = 1 if moving_window or aggregate_windows else window
        time_windows = [(track.iloc[val]['Spot frame']) for val in range(0 + window, len(track), step_size)]

        # track = track[:10]

        # normalize track:
        track = drop_columns(track, motility=motility, visual=visual)
        track = normalize_tracks(track, motility=motility, visual=visual)

        # calculate list of probabilities per window
        true_prob = get_prob_over_track(clf, track, _window, X_test, moving_window, aggregate_windows)
        if calc_delta:
            # calculate the difference in probability
            prob = [(true_prob[i] - true_prob[i - 1]) for i in range(1, len(true_prob))]
            time_windows = time_windows[1:]
        else:
            prob = true_prob

        dic = {}
        for (wt, prob) in zip(time_windows, prob):
            dic[int(wt)] = prob
        data = [dic]
        df_prob = df_prob.append(data, ignore_index=True, sort=False)

    return df_prob


def plot_avg_diff_prob(diff_video_num, con_video_num, end_of_file_name, dir_name, title):
    windows = [30, 30, 30]
    df = pd.DataFrame()
    avg_vals_diff = []
    std_vals_diff = []
    avg_vals_con = []
    std_vals_con = []
    window_times = []
    for w in windows:
        wt_c = [wt * 1 for wt in range(0, 260, w)]
        window_times.append(np.array(wt_c) / 3600 * 300)
        df_delta_prob_diff = pickle.load(
            open(dir_name + "/" + f"df_prob_w={w}, video_num={diff_video_num}" + end_of_file_name, 'rb'))
        df_delta_prob_con = pickle.load(
            open(dir_name + "/" + f"df_prob_w={w}, video_num={con_video_num}" + end_of_file_name, 'rb'))

        avg_vals_diff.append([df_delta_prob_diff[col].mean() for col in wt_c])
        std_vals_diff.append([df_delta_prob_diff[col].std() for col in wt_c])
        avg_vals_con.append([df_delta_prob_con[col].mean() for col in wt_c])
        std_vals_con.append([df_delta_prob_con[col].std() for col in wt_c])

    # plot it!
    fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(8, 4), dpi=140)

    def plot_window(ax, window_times, avg_vals_con, avg_vals_diff, ax_0, ax_1, std_vals_con, std_vals_diff, w):
        ax[ax_0, ax_1].plot(window_times, avg_vals_con)
        ax[ax_0, ax_1].plot(window_times, avg_vals_diff)
        ax[ax_0, ax_1].set_title(f"window size={windows[w]}")

        p_std = np.asarray(avg_vals_con) + np.asarray(std_vals_con)
        m_std = np.asarray(avg_vals_con) - np.asarray(std_vals_con)
        ax[ax_0, ax_1].fill_between(window_times, m_std, p_std, alpha=0.5)

        p_std = np.asarray(avg_vals_diff) + np.asarray(std_vals_diff)
        m_std = np.asarray(avg_vals_diff) - np.asarray(std_vals_diff)
        ax[ax_0, ax_1].fill_between(window_times, m_std, p_std, alpha=0.5)
        ax[ax_0, ax_1].grid()

    fig, ax = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(8, 4), dpi=140)
    ax.plot(window_times[1], avg_vals_con[1])
    ax.plot(window_times[1], avg_vals_diff[1])
    ax.set_title(f"window size={windows[1]}")

    p_std = np.asarray(avg_vals_con[1]) + np.asarray(std_vals_con[1])
    m_std = np.asarray(avg_vals_con[1]) - np.asarray(std_vals_con[1])
    ax.fill_between(window_times[1], m_std, p_std, alpha=0.5)

    p_std = np.asarray(avg_vals_diff[1]) + np.asarray(std_vals_diff[1])
    m_std = np.asarray(avg_vals_diff[1]) - np.asarray(std_vals_diff[1])
    ax.fill_between(window_times[1], m_std, p_std, alpha=0.5)
    ax.grid()

    plt.suptitle(title, wrap=True)
    fig.legend(['control', 'ERK'], loc="lower left")

    for ax in fig.get_axes():
        ax.set_xlabel('Time [h]')
        ax.set_ylabel(' Avg p delta')
        ax.label_outer()
    plt.savefig(
        dir_name + "/" + f"{title} 30.png")
    plt.show()
    plt.close(fig)


def run_calc(dir_name, tracks, video_num, motility, intensity):
    print(dir_name)
    window = 30
    wt_cols = [wt * 300 for wt in range(0, 350, window)]
    clf, X_train, X_test, y_train, y_test = load_data(dir_name)

    if not os.path.exists(dir_name + "/" + f"df_prob_w={window}, video_num={video_num}"):
        df_prob = calc_prob_delta(window, tracks, clf, X_test, motility, intensity, wt_cols, calc_delta=False)
        pickle.dump(df_prob, open(dir_name + "/" + f"df_prob_w={window}, video_num={video_num}", 'wb'))


def run_plot_delta(dir_name, intensity, motility, diff_vid_num, con_vid_num):
    title = f"Averaged diff probability (intensity= {intensity}, motility={motility}, video #{con_vid_num},{diff_vid_num})"
    plot_avg_diff_prob(diff_video_num=diff_vid_num, con_video_num=con_vid_num, end_of_file_name="",
                       dir_name=dir_name, title=title)


def run_delta(diff_t_w, motility, visual, video_diff_num, video_con_num, con_windows, path, csv_path):
    # open a new directory to save the outputs in
    # time_frame = f"{diff_t_w[0]},{diff_t_w[1]} frames ERK, {con_windows} frames con"
    # complete_path = path + "/" + time_frame

    for video_num in [video_con_num, video_diff_num]:
        csv_path = csv_path + f"/Nuclei_{video_num}-vertices.csv"
        df = pd.read_csv(csv_path, encoding="ISO-8859-1")

        # if there is manual data- keep only it:
        if 'manual' in df.columns:
            df = df[df["manual"] == 1]

        tracks = list()
        for label, labeld_df in df.groupby('Spot track ID'):
            tracks.append(labeld_df)

        run_calc(path, tracks, video_num, motility, visual)

    run_plot_delta(path, visual, motility, video_diff_num, video_con_num)


if __name__ == '__main__':

    lst_v = [1, 5]  # 1 = Control, 5 = ERK
    video_diff_num = 5  # 8
    video_con_num = 1  # 3
    dif_window = [140, 170]

    motility = True
    visual = False
    # con_windows = [[0, 30], [140, 170], [180, 210]]
    # con_windows = [[0, 30], [30, 60], [60, 90], [90, 120], [120, 150], [150, 180],
    #                [180, 210], [210, 240], [240, 270], [270, 300], [300, 330]]
    con_windows = [[0, 30], [140, 170], [180, 210], [240, 270], [300, 330]]

    # # csv_path = fr"muscle-formation-diff/data/mastodon/Nuclei_{video_diff_num}-vertices.csv"
    csv_path_d = fr"../data/mastodon/Nuclei_{video_diff_num}-vertices.csv"
    csv_path_c = fr"../data/mastodon/Nuclei_{video_con_num}-vertices.csv"

    csv_path = fr"muscle-formation-diff/data/mastodon/" if os.path.exists(
        "muscle-formation-diff/data/mastodon/") else fr"../data/mastodon/"

    dir_name = f"14-02-2022-manual_mastodon_motility-{motility}_intensity-{visual}"
    time_frame = f"{dif_window[0]},{dif_window[1]} frames ERK, {con_windows} frames con"
    complete_path = dir_name + "/" + time_frame

    open_dirs(main_dir=dir_name, inner_dir=time_frame)
    # check if cls exists already
    if not os.path.exists(complete_path + "/clf.joblib"):
        train_evaluate(lst_videos=lst_v, csv_path=csv_path, motility=motility, visual=visual, dir_name=complete_path,
                       diff_t_window=dif_window, con_t_windows=con_windows)

    # run_delta(dif_window, motility, visual, video_diff_num, video_con_num, con_windows, path=complete_path,
    #           csv_path=csv_path)

    df = pd.read_csv(csv_path, encoding="cp1252")
    df = drop_columns(df, motility, visual)
    df = normalize_tracks(df, motility, visual)
    df = df.sample(frac=0.2, replace=True, random_state=1)

    # # load the model & train set & test set
    clf, X_train, X_test, y_train, y_test = load_data(complete_path)

    new_X_test = extract_distinct_features(df, X_test.columns)
    new_X_test["target"] = True
    new_y_test = new_X_test["target"]
    new_X_test.drop(columns=["target"])

    report, auc_score = evaluate(clf, new_X_test, new_y_test)

    # plot ROC curve
    plot_roc(clf=clf, X_test=new_X_test, y_test=new_y_test, path=dir_name)

    # perform PCA analysis
    principal_df, pca = build_pca(3, new_X_test)
    plot_pca(principal_df, pca, dir_name)

    # calculate feature importance
    feature_importance(clf, new_X_test.columns, dir_name)

    # save classification report & AUC score
    txt_file = open(dir_name + '/info.txt', 'a')
    txt_file.write(f"classification report: {report}\n auc score: {auc_score}")
    txt_file.close()
