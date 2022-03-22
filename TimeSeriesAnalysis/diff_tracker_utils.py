import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
import numpy as np
import numpy as np
import pandas as pd
import os
from sklearn import metrics

from experimentation import get_shifts


def extract_distinct_features(df, feature_list, column_id="Spot track ID", column_sort="Spot frame"):
    df = extract_features(df, column_id=column_id, column_sort=column_sort)  # , show_warnings=False
    impute(df)
    return df[feature_list]

def correct_shifts(df, vid_path):
    shifts = get_shifts(vid_path, df["Spot frame"].max() + 1)
    shifts.insert(0, (0, 0))
    df["Spot position X (µm)"] = df.apply(lambda x: x["Spot position X (µm)"] + shifts[int(x["Spot frame"])][0], axis=1)
    df["Spot position Y (µm)"] = df.apply(lambda x: x["Spot position Y (µm)"] + shifts[int(x["Spot frame"])][0], axis=1)
    return df

def calc_prob_delta(window, tracks, clf, X_test, motility, visual, wt_cols, moving_window=False,
                    aggregate_windows=False, calc_delta=False, add_time=False, shift_tracks=False, vid_path=None):
    wt_cols = [i for i in range(260)] if moving_window else wt_cols
    df_prob = pd.DataFrame(columns=wt_cols)

    # tracks = tracks[:5] #todo: remove

    for ind, t in enumerate(tracks):
        track = tracks[ind]
        if len(track) < window:
            continue
        else:
            _window = window

        step_size = 1 if moving_window or aggregate_windows else window

        time_windows = [(track.iloc[val]['Spot frame']) for val in range(0 + window, len(track), step_size)]
        time_windows.sort()
        track = track.sort_values("Spot frame")

        # track = track[:33]

        # normalize track:
        if motility:
            track = drop_columns_motility(track)
            track = normalize_motility(track)
        elif visual:
            track = drop_columns_intensity(track)
            track = normalize_intensity(track)
        track.dropna(inplace=True)

        if shift_tracks:
            track = correct_shifts(track, vid_path)

        if add_time:
            track["time_stamp"] = track["Spot frame"]
        # calculate list of probabilities per window
        true_prob = get_prob_over_track(clf, track, _window, X_test.columns, moving_window)
        if calc_delta:
            # calculate the difference in probability
            prob = [(true_prob[i] - true_prob[i - 1]) for i in range(1, len(true_prob))]
            time_windows = time_windows[1:]
        else:
            prob = true_prob

        # dic = {}
        # for (wt, prob) in zip(time_windows, prob):
        #     dic[int(wt)] = prob
        # data = [dic]
        # df_prob = df_prob.append(data, ignore_index=True, sort=False)
        df_prob = df_prob.append(prob, ignore_index=True, sort=False)

    return df_prob


def concat_dfs(diff_df, con_df, diff_t_window=None, con_t_windows=None):
    def set_indexes(df, target, max_val):
        df["Spot track ID"] = df["Spot track ID"] + max_val
        max_val = df["Spot track ID"].max() + 1
        df['target'] = np.array([target for i in range(len(df))])
        return df, max_val

    max_val = 0
    diff_start, diff_end = diff_t_window
    window_size = diff_end - diff_start

    # Erk video
    # Cut the needed time window
    new_diff_df = pd.DataFrame()
    diff_df = diff_df[(diff_df["Spot frame"] >= diff_start) & (diff_df["Spot frame"] < diff_end)]
    for label, label_df in diff_df.groupby('Spot track ID'):
        if len(label_df) == window_size:
            new_diff_df = new_diff_df.append(label_df)

    # control video
    # Cut the needed time window
    control_df = pd.DataFrame()
    new_label = max(con_df['Spot track ID'].unique()) + 1
    for start, end in con_t_windows:
        tmp_df = con_df[(con_df["Spot frame"] >= start) & (con_df["Spot frame"] < end)]
        for label, label_df in tmp_df.groupby('Spot track ID'):
            if len(label_df) == window_size:
                new_label += 1
                label_df["Spot track ID"] = new_label
                control_df = control_df.append(label_df)
    con_df = control_df.copy()

    new_diff_df, max_val = set_indexes(new_diff_df, target=True, max_val=max_val)
    con_df, _ = set_indexes(con_df, target=False, max_val=max_val)
    total_df = pd.concat([new_diff_df, con_df], ignore_index=True)
    return total_df


def get_prob_over_track(clf, track, window, feature_list, moving_window=False):
    '''
    Returns a list of the probability of being differentiated, for each track portion
    :param clf: classifier
    :param track: cell's track
    :param window: the size of the track's portion
    :param features_df: dataframe to take its features, fir using the same features on the tested data
    :return: list of probabilities
    '''
    true_p = {}
    step_size = 1 if moving_window else window

    for i in range(0, len(track), step_size):
        if i + window > len(track):
            break
        track_portion = track[i:i + window]
        max_frame = track_portion["Spot frame"].max()
        X = extract_distinct_features(df=track_portion, feature_list=feature_list)
        probs = clf.predict_proba(X)
        true_p[max_frame] = probs[0][1]
        # print(f"track portion: [{i}:{i + window}]")
        # print(clf.classes_)
        # print(probs)
    print(true_p)
    return true_p


def evaluate(clf, X_test, y_test):
    predicted = cross_val_predict(clf, X_test, y_test, cv=5)
    report = classification_report(y_test, predicted)

    pred = clf.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print(report)
    print(auc)
    return report, auc


def get_unique_indexes(y):
    idxs = y.index.unique()
    lst = [y[idx] if isinstance(y[idx], np.bool_) else y[idx].iloc[0] for idx in idxs]
    y_new = pd.Series(lst, index=idxs).sort_index()
    return y_new


def drop_columns_intensity(df):
    return df[['min', 'max', 'mean', 'sum', 'Spot track ID', 'Spot frame', 'target']]


def drop_columns_motility(df):  # , motility, visual
    to_keep = ['Spot frame', 'Spot track ID', 'target']  # , 'Track N spots'
    visual_features = ['Spot center intensity (Counts)', 'Detection quality', 'Spot intensity Mean ch1 (Counts)',
                       'Spot intensity Std ch1 (Counts)', 'Spot intensity Min ch1 (Counts)',
                       'Spot intensity Max ch1 (Counts)', 'Spot intensity Median ch1 (Counts)',
                       'Spot intensity Sum ch1 (Counts)', 'Spot quick mean (Counts)', 'Spot radius (µm)', ]
    motility_features = ['Spot position X (µm)', 'Spot position Y (µm)', ]

    # to_keep.extend(visual_features) if visual else to_keep.extend([])
    to_keep.extend(motility_features)  # if motility else to_keep.extend([])
    to_keep = [value for value in to_keep if value in df.columns]
    return df[to_keep]


def normalize_intensity(df):
    columns = [e for e in list(df.columns) if e not in ('Spot frame', 'Spot track ID', 'target')]
    scaler = StandardScaler()  # create a scaler
    df[columns] = scaler.fit_transform(df[columns])  # transform the feature
    return df


def normalize_motility(df):
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

    #########################

    # if 'delta Spot position X (µm)' in df.columns:
    #     df['Spot position X (µm)'] = df['Spot position X (µm)'].diff()
    #
    # if 'delta Spot position Y (µm)' in df.columns:
    #     df['Spot position Y (µm)'] = df['Spot position Y (µm)'].diff()
    #
    # df.fillna(0, inplace=True)

    return df


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


def open_dirs(main_dir, inner_dir):
    if not os.path.exists(main_dir):
        os.mkdir(main_dir)
    print(main_dir + "/" + inner_dir)
    if not os.path.exists(main_dir + "/" + inner_dir):
        os.mkdir(main_dir + "/" + inner_dir)


def train(X_train, y_train):
    clf = RandomForestClassifier(max_depth=8)

    # import xgboost as xgb
    # clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train, )
    return clf


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


def plot_avg_conf(path):
    def plot(df, color1, color2):
        avg_vals_diff = ([df[col].mean() for col in df.columns])
        std_vals_diff = ([df[col].std() for col in df.columns])
        p_std = np.asarray(avg_vals_diff) + np.asarray(std_vals_diff)
        m_std = np.asarray(avg_vals_diff) - np.asarray(std_vals_diff)

        plt.plot([i * 5 / 60 for i in range(len(avg_vals_diff))], avg_vals_diff, color=color1)
        plt.fill_between([i * 5 / 60 for i in range(len(avg_vals_diff))], m_std, p_std, alpha=0.5, color=color2)

    fuse_df = pickle.load(open(path + f"/df_prob_w={30}, video_num={3}", 'rb'))
    not_fuse_df = pickle.load(open(path + f"/df_prob_w={30}, video_num={2}", 'rb'))

    plot(fuse_df, "DarkOrange", "Orange")
    plot(not_fuse_df, "blue", "blue")
    plt.legend(["Erk avg", "Erk std", "Control avg", "Control std"])
    plt.xlabel("time (h)")
    plt.ylabel("avg confidence")
    plt.title("avg differentiation confidence over time (motility)")
    plt.plot([i * 5 / 60 for i in range(260)], [0.5 for i in range(260)], color="black", linestyle="--")
    plt.savefig(path + "/avg conf s3, s2.png")
    plt.show()
    plt.clf()


if __name__ == '__main__':
    cluster_path = "muscle-formation-diff"
    local_path = ".."
    path = cluster_path

    diff_window = [140, 170]
    tracks_len = 30

    con_windows = [

        [[0, 30], [40, 70], [90, 120], [140, 170], [180, 210], [220, 250]],
        [[10, 40], [40, 70], [70, 100], [100, 130], [130, 160], [160, 190], [190, 220], [220, 250]]

        # [[0, 50], [50, 100], [100, 150], [150, 200], [200, 250]],
        # [[0, 50], [60, 110], [110, 160], [160, 210], [210, 260]],
        # [[10, 60], [60, 110], [110, 160], [160, 210], [210, 260], [150, 180], [180, 210], [210, 240]],
    ]

    for w in con_windows:
        dir_path = f"20-03-2022-manual_mastodon_motility shifted tracks"
        second_dir = f"{diff_window} frames ERK, {w} frames con track len {tracks_len}"
        dir_path += "/" + second_dir
        plot_avg_conf(dir_path)

        clf, X_train, X_test, y_train, y_test = load_data(dir_path)
        pred = clf.predict(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        txt_file = open(dir_path + '/info.txt', 'a')
        txt_file.write(f"\n\n new auc: {auc}")

        txt_file.close()