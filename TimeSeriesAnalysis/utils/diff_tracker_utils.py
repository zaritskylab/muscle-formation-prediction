import os
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve
from sklearn import metrics

sys.path.append('/sise/home/shakarch/muscle-formation-diff')
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
import numpy as np
import more_itertools as mit
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'


def evaluate(clf, X_test, y_test):
    pred = clf.predict(X_test)
    report = classification_report(y_test, pred)

    fpr, tpr, thresholds = roc_curve(y_test, pred, pos_label=1)
    auc_score = metrics.auc(fpr, tpr)
    print(report)
    print(auc_score)
    return report, auc_score


def train_model(X_train, y_train):
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    # clf = LinearDiscriminantAnalysis()
    clf = RandomForestClassifier(max_depth=8)
    clf.fit(X_train, y_train)
    return clf


def get_position(ind, df):
    x = int(df.iloc[ind]["Spot position X"] / 0.462)
    y = int(df.iloc[ind]["Spot position Y"] / 0.462)
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
    return df_measures


def get_local_densities_df(df_s, tracks_s, neighboring_distance=100):
    local_densities = pd.DataFrame(columns=[i for i in range(df_s["Spot frame"].max() + 2)])
    for track in tracks_s:
        spot_frames = list(track.sort_values("Spot frame")["Spot frame"])
        track_local_density = {
            t: get_local_density(df=df_s,
                                 x=track[track["Spot frame"] == t]["Spot position X"].values[0],
                                 y=track[track["Spot frame"] == t]["Spot position Y"].values[0],
                                 t=t,
                                 neighboring_distance=neighboring_distance)
            for t in spot_frames}
        local_densities = local_densities.append(track_local_density, ignore_index=True)
    return local_densities


def get_density(df, experiment):
    densities = pd.DataFrame()
    for t, t_df in df.groupby("Spot frame"):
        densities = densities.append({"Spot frame": t, "density": len(t_df)}, ignore_index=True)
    densities["experiment"] = experiment
    return densities


def get_local_density(df, x, y, t, neighboring_distance):
    neighbors = df[(np.sqrt(
        (df["Spot position X"] - x) ** 2 + (df["Spot position Y"] - y) ** 2) <= neighboring_distance) &
                   (df['Spot frame'] == t) &
                   (0 < np.sqrt((df["Spot position X"] - x) ** 2 + (df["Spot position Y"] - y) ** 2))]
    return len(neighbors)


def extract_distinct_features(df, feature_list, column_id="Spot track ID", column_sort="Spot frame"):
    df = extract_features(df, column_id=column_id, column_sort=column_sort)  # , show_warnings=False
    impute(df)
    return df[feature_list]


def add_features(track, df_s, local_density=True, neighboring_distance=50):
    if local_density:
        spot_frames = list(track.sort_values("Spot frame")["Spot frame"])
        track_local_density = [
            get_local_density(df=df_s,
                              x=track[track["Spot frame"] == t]["Spot position X"].values[0],
                              y=track[track["Spot frame"] == t]["Spot position Y"].values[0],
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


def split_data_to_time_portions(data, track_len):
    data = data.drop_duplicates(subset=["Spot track ID", "Spot frame"])  # remove duplicates
    time_windows = data.sort_values("Spot frame")['Spot frame'].unique()
    time_windows_strides = list(mit.windowed(time_windows, n=track_len, step=1))
    t_portion_lst = [data[data["Spot frame"].isin(time_windows_strides[i])] for i in range(len(time_windows_strides))]

    return t_portion_lst


def remove_short_tracks(df_to_transform, len_threshold):
    counts = df_to_transform.groupby("Spot track ID")["Spot track ID"].transform(len)
    mask = (counts >= len_threshold)
    return df_to_transform[mask]


def calc_prob(transformed_tracks_df, clf, n_frames=260):
    df_score = pd.DataFrame(columns=[i for i in range(n_frames)])
    for track_id, track in transformed_tracks_df.groupby("Spot track ID"):
        spot_frames = list(track.sort_values("Spot frame")["Spot frame"])
        diff_score = {"Spot track ID": track_id}
        try:
            for t in spot_frames:
                probs = clf.predict_proba(track[track["Spot frame"] == t].drop(["Spot track ID", "Spot frame"], axis=1))

                diff_score[t] = pd.to_numeric(probs[0][1], downcast='float')

            df_score = df_score.append(diff_score, ignore_index=True, sort=False)
        except Exception as e:
            print(e)
            print(track[track["Spot frame"] == t].drop(["Spot track ID", "Spot frame"], axis=1).size)
    print(df_score.shape)
    return df_score


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
    print("size of diff_df: ", diff_df.shape)

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
    print("size of con_df: ", con_df.shape)

    new_diff_df, max_val = set_indexes(new_diff_df, target=True, max_val=max_val)
    con_df, _ = set_indexes(con_df, target=False, max_val=max_val)
    total_df = pd.concat([new_diff_df, con_df], ignore_index=True)
    return total_df


def get_unique_indexes(y):
    idxs = y.index.unique()
    lst = [y[idx] if isinstance(y[idx], np.bool_) else y[idx].iloc[0] for idx in idxs]
    y_new = pd.Series(lst, index=idxs).sort_index()
    return y_new


def open_dirs(main_dir, inner_dir):
    if not os.path.exists(main_dir):
        os.mkdir(main_dir)
    print(main_dir + "/" + inner_dir)
    if not os.path.exists(main_dir + "/" + inner_dir):
        os.mkdir(main_dir + "/" + inner_dir)


def split_data_by_tracks(data, n_tasks):
    ids_list = data["Spot track ID"].unique()
    n = len(ids_list) // n_tasks
    ids_chunks = [ids_list[i:i + n] for i in range(0, len(ids_list), n)]
    return ids_chunks


if __name__ == '__main__':
    print("diff_tracker_utils")
