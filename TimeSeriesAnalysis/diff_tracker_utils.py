import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute, impute_dataframe_zero
import numpy as np
import os
from sklearn import metrics
import more_itertools as mit
import consts
from experimentation import get_shifts
from skimage import io
import pandas as pd
import glob
import skimage
import nuc_segmentor
import sys
import cv2

pd.options.mode.chained_assignment = None  # default='warn'


def get_position(ind, df):
    x = int(df.iloc[ind]["Spot position X (µm)"] / 0.462)
    y = int(df.iloc[ind]["Spot position Y (µm)"] / 0.462)
    spot_frame = int(df.iloc[ind]["Spot frame"])
    return x, y, spot_frame


def get_centered_image(ind, df, im_actin, window_size):
    x, y, spot_frame = get_position(ind, df)
    cropped = im_actin[spot_frame][x - window_size:x + window_size, y - window_size: y + window_size]
    return cropped


def get_single_cell_intensity_measures(label, df, im_actin, window_size):
    # try:
    df_measures = pd.DataFrame(columns=["min", "max", "mean", "sum", "Spot track ID", "Spot frame", "x", "y", ])
    for i in range(len(df)):  # len(df)
        img = get_centered_image(i, df, im_actin, window_size)
        try:
            min_i, max_i, mean_i, sum_i = img.min(), img.max(), img.mean(), img.sum()
        except:
            continue
        x, y, spot_frame = get_position(i, df)
        data = {"min": min_i, "max": max_i, "mean": mean_i, "sum": sum_i, "Spot track ID": label,
                "Spot frame": spot_frame,
                "x": x, "y": y}
        df_measures = df_measures.append(data, ignore_index=True)
    # except:
    #     return pd.DataFrame()
    return df_measures


def get_intensity_measures_df(df, video_actin_path, window_size, local_density):
    im_actin = io.imread(video_actin_path)
    total_df = pd.DataFrame()
    for label, label_df in df.groupby("Spot track ID"):
        if len(label_df) >= window_size:
            df_measures = get_single_cell_intensity_measures(label=label, df=label_df, im_actin=im_actin,
                                                             window_size=window_size)
            if not df_measures.empty:
                if len(df_measures) >= window_size:
                    if local_density:
                        df_measures["local density"] = label_df["local density"]
                    df_measures["manual"] = 1
                    total_df = pd.concat([total_df, df_measures], axis=0)

    return total_df


def get_local_densities_df(df_s, tracks_s, neighboring_distance=100):
    local_densities = pd.DataFrame(columns=[i for i in range(df_s["Spot frame"].max() + 2)])
    for track in tracks_s:
        spot_frames = list(track.sort_values("Spot frame")["Spot frame"])
        track_local_density = {
            t: get_local_density(df=df_s,
                                 x=track[track["Spot frame"] == t]["Spot position X (µm)"].values[0],
                                 y=track[track["Spot frame"] == t]["Spot position Y (µm)"].values[0],
                                 t=t,
                                 neighboring_distance=neighboring_distance)
            for t in spot_frames}
        local_densities = local_densities.append(track_local_density, ignore_index=True)
    return local_densities


def load_clean_rows(csv_path):
    df = pd.read_csv(csv_path, encoding="cp1252")
    df = df.drop(labels=range(0, 2), axis=0)
    return df


def get_density(df, experiment):
    densities = pd.DataFrame()
    for t, t_df in df.groupby("Spot frame"):
        densities = densities.append({"Spot frame": t, "density": len(t_df)}, ignore_index=True)
    densities["experiment"] = experiment
    return densities


def get_local_density(df, x, y, t, neighboring_distance):
    neighbors = df[(np.sqrt(
        (df["Spot position X (µm)"] - x) ** 2 + (df["Spot position Y (µm)"] - y) ** 2) <= neighboring_distance) &
                   (df['Spot frame'] == t) &
                   (0 < np.sqrt((df["Spot position X (µm)"] - x) ** 2 + (df["Spot position Y (µm)"] - y) ** 2))]
    return len(neighbors)


def get_tracks(path, manual_tagged_list, target=1):
    df_s = load_clean_rows(path)  # .dropna()
    df_s.rename(columns={"Spot position": "Spot position X (µm)", "Spot position.1": "Spot position Y (µm)",
                         "Spot center intensity": "Spot center intensity Center ch1 (Counts)",
                         "Spot intensity": "Spot intensity Mean ch1 (Counts)",
                         "Spot intensity.1": "Spot intensity Std ch1 (Counts)",
                         "Spot intensity.2": "Spot intensity Min ch1 (Counts)",
                         "Spot intensity.3": "Spot intensity Max ch1 (Counts)",
                         "Spot intensity.4": "Spot intensity Median ch1 (Counts)",
                         "Spot intensity.5": "Spot intensity Sum ch1 (Counts)",
                         },
                inplace=True)

    df_s.rename(columns=lambda c: 'Spot position X (µm)' if c.startswith('Spot position X') else c, inplace=True)
    df_s.rename(columns=lambda c: 'Spot position Y (µm)' if c.startswith('Spot position Y') else c, inplace=True)

    df_s["Spot frame"] = df_s["Spot frame"].astype(int)
    df_s["Spot position X (µm)"] = df_s["Spot position X (µm)"].astype(float)
    df_s["Spot position Y (µm)"] = df_s["Spot position Y (µm)"].astype(float)
    df_s["Spot track ID"] = df_s["Spot track ID"].astype(float)
    # print(manual_tagged_list)
    data = df_s[df_s["manual"] == 1] if manual_tagged_list else df_s
    tracks_s = get_tracks_list(data, target=target)
    return df_s, tracks_s


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


def drop_columns_nuc_intensity(track, added_features=[]):
    to_keep = ['x',
               'y',
               'max_nuc',
               'mean_nuc',
               'min_nuc',
               'nuc_size',
               'std_nuc',
               'sum_nuc',
               'Spot track ID',
               'Spot frame', 'target']
    to_keep.extend(added_features)
    return track[to_keep]


def normalize_nuc_intensity(track, added_features=[]):
    columns = [e for e in list(track.columns) if e not in ('Spot frame', 'Spot track ID', 'target')]
    columns = [e for e in columns if e not in added_features]
    scaler = StandardScaler()  # create a scaler
    track[columns] = scaler.fit_transform(track[columns])  # transform the feature
    return track


def normalize_track(track, motility, visual, nuc_intensity, shift_tracks, vid_path=None):
    # normalize track:
    if motility:
        track = drop_columns_motility(track)
        track = normalize_motility(track)

    elif visual:
        track = drop_columns_intensity(track)
        track = normalize_intensity(track)

    elif nuc_intensity:
        track = drop_columns_nuc_intensity(track)
        track = normalize_nuc_intensity(track)

    track.dropna(inplace=True)

    if shift_tracks:
        track = correct_shifts(track, vid_path)
    return track


def add_features(track, df_s, local_density=True, neighboring_distance=50):
    if local_density:
        spot_frames = list(track.sort_values("Spot frame")["Spot frame"])
        track_local_density = [
            get_local_density(df=df_s,
                              x=track[track["Spot frame"] == t]["Spot position X (µm)"].values[0],
                              y=track[track["Spot frame"] == t]["Spot position Y (µm)"].values[0],
                              t=t,
                              neighboring_distance=neighboring_distance)
            for t in spot_frames]
        track["local density"] = track_local_density
    return track


def add_features_df(df, df_s, local_density=True):
    if local_density:
        new_df = pd.DataFrame()
        for label, track in df.groupby("Spot track ID"):
            track = track.sort_values("Spot frame")
            track = add_features(track, local_density=local_density, df_s=df_s)
            new_df = new_df.append(track, ignore_index=True)
        return new_df
    else:
        return df


def transform_tsfresh_single_cell(track, motility, visual, nuc_intensity, window, shift_tracks=False, vid_path=None):
    # track = track.sort_values("Spot frame")
    track = normalize_track(track, motility, visual, nuc_intensity, shift_tracks, vid_path)

    track_transformed = pd.DataFrame()
    target = track["target"].iloc[0]
    track = track.drop(columns="target")

    list_of_track_portions = [track.iloc[i:i + window, :] for i in range(0, len(track), 1) if
                              i < len(track) - window + 1]
    for track_portion in list_of_track_portions:
        portion_transformed = extract_features(track_portion,
                                               column_id="Spot track ID",
                                               column_sort="Spot frame",
                                               show_warnings=False,
                                               disable_progressbar=True,
                                               n_jobs=8)  # 12
        portion_transformed["Spot frame"] = track_portion["Spot frame"].max()
        track_transformed = track_transformed.append(portion_transformed, ignore_index=True)
    track_transformed["Spot track ID"] = track["Spot track ID"].max()
    track_transformed["target"] = target
    return track_transformed


def split_track_to_portions(track, window):
    track_portions_lst = [track.iloc[i:i + window, :] for i in range(0, len(track), 1) if i < len(track) - window + 1]
    return pd.concat(track_portions_lst, axis=0, ignore_index=True)


def split_data_to_time_portions(data, window):
    data = data.drop_duplicates(subset=["Spot track ID", "Spot frame"])  # remove duplicates
    time_windows = data.sort_values("Spot frame")['Spot frame'].unique()
    time_windows_strides = list(mit.windowed(time_windows, n=window, step=1))
    t_portion_lst = [data[data["Spot frame"].isin(time_windows_strides[i])] for i in range(len(time_windows_strides))]

    return t_portion_lst


def remove_short_tracks(df_to_transform, len_threshold):
    counts = df_to_transform.groupby("Spot track ID")["Spot track ID"].transform(len)
    mask = (counts >= len_threshold)
    return df_to_transform[mask]


def trandform_tsfresh_df_time(df_to_transform, target, motility, nuc_intensity, visual, window, shift_tracks=False,
                              vid_path=None, feature_list=None):
    df_to_transform["target"] = target
    df_to_transform = remove_short_tracks(df_to_transform, window)
    df_time_window_split_list = split_data_to_time_portions(df_to_transform, window)

    transformed_data = pd.DataFrame()
    for time_portion in tqdm(df_time_window_split_list):
        time_portion = remove_short_tracks(time_portion, window)
        if not time_portion.empty:
            time_portion = normalize_track(time_portion, motility, visual, nuc_intensity, shift_tracks, vid_path)
            portion_transformed = extract_features(time_portion,
                                                   column_id="Spot track ID",
                                                   column_sort="Spot frame",
                                                   show_warnings=False,
                                                   disable_progressbar=True,
                                                   n_jobs=8)  # 12
            impute(portion_transformed)
            # impute_dataframe_zero(portion_transformed)
            portion_transformed["Spot track ID"] = portion_transformed.index
            portion_transformed["Spot frame"] = time_portion["Spot frame"].max()
            portion_transformed["target"] = target
            transformed_data = transformed_data.append(portion_transformed, ignore_index=True)

    return transformed_data


# todo: this is the former version of my code.
def trandform_tsfresh_df_track(df_to_transform, target, motility, nuc_intensity, visual, window, shift_tracks=False,
                               vid_path=None, feature_list=None):
    transformed_tracks = pd.DataFrame()
    if len(df_to_transform) == 0:
        return transformed_tracks

    tracks = get_tracks_list(df_to_transform, target=target)

    for track in tqdm(tracks):
        if len(track) >= window:
            track_transformed = transform_tsfresh_single_cell(track=track, motility=motility, visual=visual,
                                                              window=window, nuc_intensity=nuc_intensity,
                                                              shift_tracks=shift_tracks, vid_path=vid_path)
            impute(track_transformed)
            transformed_tracks = transformed_tracks.append(track_transformed, ignore_index=True)
            print()

    return transformed_tracks


def calc_prob(transformed_tracks_df, clf, n_frames=260):
    df_score = pd.DataFrame(columns=[i for i in range(n_frames)])
    for track_id, track in transformed_tracks_df.groupby("Spot track ID"):
        spot_frames = list(track.sort_values("Spot frame")["Spot frame"])
        diff_score = {"Spot track ID": track_id}
        for t in spot_frames:
            probs = clf.predict_proba(track[track["Spot frame"] == t].drop(["Spot track ID", "Spot frame"], axis=1))
            diff_score[t] = probs[0][1]

        df_score = df_score.append(diff_score, ignore_index=True, sort=False)
    return df_score


def calc_prob_delta(window, tracks, clf, X_test, motility, visual, wt_cols, moving_window=False,
                    aggregate_windows=False, calc_delta=False, add_time=False, shift_tracks=False, vid_path=None):
    wt_cols = [i for i in range(260)] if moving_window else wt_cols
    df_prob = pd.DataFrame(columns=wt_cols)

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

        track = normalize_track(track, motility, visual, shift_tracks, vid_path)

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
    # Todo I might have a bug in here. Each record holds a transformation of the former 30 frames,
    #  so in this way i take too many time frames, with duplicates
    # diff_df = diff_df[(diff_df["Spot frame"] >= diff_start) & (diff_df["Spot frame"] < diff_end)]
    # todo: the correction:
    diff_df = diff_df[diff_df["Spot frame"] == diff_end]

    for label, label_df in diff_df.groupby('Spot track ID'):
        # if len(
        # label_df) == window_size:  # todo: i removed that row, since I already know that each record holds the transformation of the last 30 frames
        new_diff_df = new_diff_df.append(label_df)

    # control video
    # Cut the needed time window
    control_df = pd.DataFrame()
    new_label = max(con_df['Spot track ID'].unique()) + 1
    for start, end in con_t_windows:
        # Todo I might have a bug in here. Each record holds a transformation of the former 30 frames,
        #  so in this way i take too many time frames, with duplicates
        # tmp_df = con_df[(con_df["Spot frame"] >= start) & (con_df["Spot frame"] < end)]
        # todo: the correction:
        tmp_df = con_df[con_df["Spot frame"] == end]
        for label, label_df in tmp_df.groupby('Spot track ID'):
            # if len(
            #         label_df) == window_size:  # todo: i removed that row, since I already know that each record holds the transformation of the last 30 frames
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

    print(true_p)
    return true_p


def evaluate(clf, X_test, y_test):
    # predicted = cross_val_predict(clf, X_test, y_test, cv=5)

    pred = clf.predict(X_test)
    report = classification_report(y_test, pred)

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


def drop_columns_intensity(df, added_features=[]):
    to_keep = ['min', 'max', 'mean', 'sum', 'Spot track ID', 'Spot frame', 'target']
    to_keep.extend(added_features)
    return df[to_keep]


def drop_columns_motility(df, added_features=[]):  # , motility, visual
    to_keep = ['Spot frame', 'Spot track ID', 'target']  # , 'Track N spots'
    motility_features = ['Spot position X (µm)', 'Spot position Y (µm)', ]
    to_keep.extend(added_features)
    to_keep.extend(motility_features)  # if motility else to_keep.extend([])
    to_keep = [value for value in to_keep if value in df.columns]
    return df[to_keep]


def normalize_intensity(df, added_features=[]):
    columns = [e for e in list(df.columns) if e not in ('Spot frame', 'Spot track ID', 'target')]
    columns = [e for e in columns if e not in added_features]
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
        X_train.to_csv(dir_name + "/" + "X_train")
        # pickle.dump(X_train, open(dir_name + "/" + "X_train", 'wb'))
    if X_test is not None:
        X_test.to_csv(dir_name + "/" + "X_test")
        # pickle.dump(X_test, open(dir_name + "/" + "X_test", 'wb'))
    if y_test is not None:
        y_test.to_csv(dir_name + "/" + "y_test")
        # pickle.dump(y_test, open(dir_name + "/" + "y_test", 'wb'))
    if y_train is not None:
        y_train.to_csv(dir_name + "/" + "y_train")
        # pickle.dump(y_train, open(dir_name + "/" + "y_train", 'wb'))
    if clf is not None:
        joblib.dump(clf, dir_name + "/" + "clf.joblib")


def load_data(dir_name):
    # load the model & train set & test set
    clf = joblib.load(dir_name + "/clf.joblib")
    # X_train = pickle.load(open(dir_name + "/" + "X_train", 'rb'))
    # X_test = pickle.load(open(dir_name + "/" + "X_test", 'rb'))
    # y_train = pickle.load(open(dir_name + "/" + "y_train", 'rb'))
    # y_test = pickle.load(open(dir_name + "/" + "y_test", 'rb'))
    X_train = pd.read_csv(dir_name + "/" + "X_train", encoding="cp1252")
    X_test = pd.read_csv(dir_name + "/" + "X_test", encoding="cp1252")
    y_train = pd.read_csv(dir_name + "/" + "y_train", encoding="cp1252")
    y_test = pd.read_csv(dir_name + "/" + "y_test", encoding="cp1252")

    return clf, X_train, X_test, y_train, y_test


def open_dirs(main_dir, inner_dir):
    if not os.path.exists(main_dir):
        os.mkdir(main_dir)
    print(main_dir + "/" + inner_dir)
    if not os.path.exists(main_dir + "/" + inner_dir):
        os.mkdir(main_dir + "/" + inner_dir)


def train(X_train, y_train):
    clf = RandomForestClassifier(max_depth=8)
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
    plt.close()
    plt.clf()


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
    plt.close()
    plt.clf()


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
    plt.close()
    plt.clf()


def plot_avg_conf(path):
    def plot(df, color1, color2):
        avg_vals_diff = ([df[col].mean() for col in df.columns])
        std_vals_diff = ([df[col].std() for col in df.columns])
        p_std = np.asarray(avg_vals_diff) + np.asarray(std_vals_diff)
        m_std = np.asarray(avg_vals_diff) - np.asarray(std_vals_diff)

        plt.plot([i * 5 / 60 for i in range(len(avg_vals_diff))], avg_vals_diff, color=color1)
        plt.fill_between([i * 5 / 60 for i in range(len(avg_vals_diff))], m_std, p_std, alpha=0.5, color=color2)

    fuse_df = pickle.load(open(path + f"/df_prob_w={30}, video_num={5}", 'rb'))
    not_fuse_df = pickle.load(open(path + f"/df_prob_w={30}, video_num={1}", 'rb'))

    plot(fuse_df, "DarkOrange", "Orange")
    plot(not_fuse_df, "blue", "blue")
    plt.legend(["Erk avg", "Erk std", "Control avg", "Control std"])
    plt.xlabel("time (h)")
    plt.ylabel("avg confidence")
    plt.title("avg differentiation confidence over time (motility)")
    plt.plot([i * 5 / 60 for i in range(260)], [0.5 for i in range(260)], color="black", linestyle="--")
    plt.savefig(path + "/avg conf s5, s1.png")
    plt.show()
    plt.close()
    plt.clf()


def get_tracks_list(int_df, target):
    if target:
        int_df['target'] = target
    tracks = list()
    for label, labeld_df in int_df.groupby('Spot track ID'):
        tracks.append(labeld_df)
    return tracks


def single_nuc_crop(x, y, win_size, spot_frame, video):
    """
    The method returns a cropped image of the given nuclei's information
    :param x: nuclei's center x coordination
    :param y: nuclei's center y coordination
    :param win_size: diameter of the cropped image (image size is win_size X win_size)
    :param spot_frame: frame number to crop an image from
    :param video: nuclei channel video
    :return: single_nuc_crop: np.array of shape (win_size, win_size, 1) of the cropped image
    """
    # Crop the needed nuclear
    single_nuc_crop = video[spot_frame, int(y - (win_size / 2)):int(y + (win_size / 2)),
                      int(x - (win_size / 2)):int(x + (win_size / 2))]
    return single_nuc_crop


def get_nuclei_measures(segmentor, label, df, vid_nuc, window_size):
    df_measures = pd.DataFrame(columns=["min", "max", "mean", "sum", "Spot track ID", "Spot frame", "x", "y", ])
    missed_segmentation_counter = 0
    for i in range(len(df)):  # len(df)
        x = df.iloc[i]["Spot position X (µm)"] / 0.462
        y = df.iloc[i]["Spot position Y (µm)"] / 0.462
        t = df.iloc[i]["Spot frame"]
        crop = single_nuc_crop(x, y, window_size, t, vid_nuc)
        try:
            _, threshold = cv2.threshold(crop, 0, np.max(crop), cv2.THRESH_TRIANGLE)
            pre_seg = segmentor.preprocess_image(threshold)
            segmented_crop = segmentor.segment_image(pre_seg)
            segmented_crop = segmentor.segment_postprocess(segmented_crop)
            nuc_size = nuc_segmentor.NucleiFeatureCalculator.size(segmented_crop)

            min_nuc, max_nuc, mean_nuc, std_nuc, sum_nuc = nuc_segmentor.NucleiFeatureCalculator.intensity(
                segmented_crop, crop)
        except:  # raised if image crop is empty.
            print(crop.shape)
            missed_segmentation_counter += 1
            continue
        data = {"nuc_size": nuc_size, "min_nuc": min_nuc, "max_nuc": max_nuc, "mean_nuc": mean_nuc, "sum_nuc": sum_nuc,
                "std_nuc": std_nuc, "Spot track ID": label, "Spot frame": t, "x": x, "y": y}
        df_measures = df_measures.append(data, ignore_index=True)
    print(f"cell #{label}, missed spots: {missed_segmentation_counter}/{len(df)}")
    return df_measures


def get_nuc_intensity_measures_df(df, nuclei_vid_path, window_size, local_density):
    vid_nuc = io.imread(nuclei_vid_path)
    total_df = pd.DataFrame()
    segmentor = nuc_segmentor.NucSegmentor()
    window_size = 2 * window_size
    for label, label_df in df.groupby("Spot track ID"):
        if len(label_df) >= window_size:
            df_measures = get_nuclei_measures(segmentor, label, label_df, vid_nuc, window_size)
            if not df_measures.empty:
                if len(df_measures) >= window_size:
                    if local_density:
                        df_measures["local density"] = label_df["local density"]
                    df_measures["manual"] = 1
                    total_df = pd.concat([total_df, df_measures], axis=0)
    return total_df


def run_tsfresh_preprocess(df, s_run, motility, intensity, nuc_intensity, tracks_len, local_density, winsize, split_by):
    if intensity:
        print(df.shape)
        df = get_intensity_measures_df(df=df,
                                       video_actin_path=path + s_run["actin_path"],
                                       window_size=winsize, local_density=local_density)
        print(df.shape)
    if nuc_intensity:
        df = get_nuc_intensity_measures_df(df=df, nuclei_vid_path=path + s_run["nuc_path"], window_size=winsize,
                                           local_density=local_density)

    if split_by == "time":
        trans_df = trandform_tsfresh_df_time(df_to_transform=df, target=s_run["target"], motility=motility,
                                             visual=intensity, nuc_intensity=nuc_intensity,
                                             window=tracks_len,
                                             shift_tracks=False, vid_path=None)
    elif split_by == "track_id":

        trans_df = trandform_tsfresh_df_track(df_to_transform=df, target=s_run["target"], motility=motility,
                                              visual=intensity, nuc_intensity=nuc_intensity,
                                              window=tracks_len,
                                              shift_tracks=False, vid_path=None)
    return trans_df


def split_data_by_tracks(data, n_tasks):
    ids_list = data["Spot track ID"].unique()
    n = len(ids_list) // n_tasks
    ids_chunks = [ids_list[i:i + n] for i in range(0, len(ids_list), n)]
    return ids_chunks


def set_params(to_run_p, registration_p):
    motility = True if to_run_p == "motility" else False
    intensity = True if to_run_p == "intensity" else False
    nuc_intensity = True if to_run_p == "nuc_intensity" else False
    csv_path = s_run["pkl_all_reg_path"] if registration_p else s_run["csv_all_path"]

    return motility, intensity, nuc_intensity, csv_path


if __name__ == '__main__':
    path = consts.cluster_path
    to_run = sys.argv[1]  # "motility"  # "intensity" "nuc_intensity" motility
    # s_run = consts.s_runs[os.getenv('SLURM_ARRAY_TASK_ID')[0]]
    s_run = consts.s_runs[sys.argv[3]]
    registration = sys.argv[2] == 'True'
    # jobid = int(os.getenv('SLURM_ARRAY_TASK_ID'))

    # path = consts.local_path
    jobid = 100
    # to_run="nuc_intensity"
    # s_run = consts.s_runs["1"]
    # registration = False

    motility, intensity, nuc_intensity, csv_path = set_params(to_run, registration)

    split_by = "track_id"
    n_tasks = 15

    srun_files_dir_path = path + f"/data/mastodon/ts_transformed_new/{to_run}/{s_run['name']}"
    os.makedirs(srun_files_dir_path, exist_ok=True)

    file_save_path = srun_files_dir_path + f"/{s_run['name']}_imputed reg={registration}, " \
                                           f"local_den={consts.local_density}" + \
                     (f"win size {consts.window_size}" if intensity else "")

    print(f"running: to_run={to_run}, "
          f"video={s_run['name']}, "
          f"local density={consts.local_density}, "
          f"reg={registration}")

    df_all, _ = get_tracks(path + csv_path, manual_tagged_list=True)
    df_tagged = df_all[df_all["manual"] == 1]
    df_tagged = add_features_df(df_tagged, df_all, local_density=consts.local_density)
    del df_all

    # ======================================================================

    if split_by == "time":
        df_time_window_split_list = split_data_to_time_portions(df_tagged, consts.tracks_len)
        run_splits = [list(c) for c in mit.divide(n_tasks, df_time_window_split_list)]
        curr_split = run_splits[jobid]
        print(f"split #{jobid}")
        print(f"total chunks in the current split:{len(curr_split)}")

        for i, df_chunk in enumerate(curr_split):
            print(f"chunk #{i}")
            trans_df = run_tsfresh_preprocess(df_chunk, s_run, motility, intensity, nuc_intensity, consts.tracks_len,
                                              consts.local_density, consts.window_size, split_by)
            if not trans_df.empty:
                trans_df.to_csv(file_save_path + f"{jobid}_{i}")

                csv = pd.read_csv(file_save_path + f"{jobid}_{i}", encoding="cp1252")
                print(csv.head())

    if split_by == "track_id":
        df_all_trans = pd.DataFrame()
        ids_chunks = split_data_by_tracks(df_tagged, n_tasks)
        print(f"split #{jobid}")
        print(f"total chunks in the current split:{len(ids_chunks)}")
        for chunk_id, chunk in enumerate(ids_chunks):
            print(f"chunk #{chunk_id}")
            df_chunk = df_tagged[df_tagged["Spot track ID"].isin(chunk)]
            trans_df = run_tsfresh_preprocess(df_chunk, s_run, motility, intensity, nuc_intensity, consts.tracks_len,
                                              consts.local_density, consts.window_size, split_by)
            if not trans_df.empty:
                df_all_trans = df_all_trans.append(trans_df)
                print(df_all_trans.head())
        df_all_trans.to_csv(file_save_path + f"{jobid}")

    # ======================================================================

    df_all_chunks = pd.DataFrame()
    for file in os.listdir(srun_files_dir_path):
        if f"reg={registration}, local_den={consts.local_density}" in file:
            if intensity:
                if f"win size {consts.window_size}" not in file:
                    continue
            try:
                chunk_df = pd.read_csv(os.path.join(srun_files_dir_path, file), encoding="cp1252")
                print(chunk_df.shape)
                if chunk_df.shape[0] > 0:
                    df_all_chunks = pd.concat([df_all_chunks, chunk_df], ignore_index=True)
            except:
                continue
    df_all_chunks.to_csv(path + f"/data/mastodon/ts_transformed_new/{to_run}/"
                                f"{s_run['name']}_imputed reg={registration}, local_den={consts.local_density}, "
                                f"win size {consts.window_size}")

    print(df_all_chunks.shape)
    print("saved")

    # delete files
    for file in os.listdir(srun_files_dir_path):
        if f"reg={registration}, local_den={consts.local_density}" in file:
            if intensity:
                if f"win size {consts.window_size}" not in file:
                    continue

            os.remove(os.path.join(srun_files_dir_path, file))
