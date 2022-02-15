from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
import numpy as np
import pandas as pd

def extract_distinct_features(df, feature_list, column_id="label", column_sort="frame"):
    df = extract_features(df, column_id=column_id, column_sort=column_sort) #, show_warnings=False
    impute(df)
    return df[feature_list]

def concat_dfs_motility(lst_videos, csv_path, diff_t_window=None, con_t_windows=None):
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
            control_df = pd.DataFrame()
            new_label = max(df['Spot track ID'].unique()) + 1
            for start, end in con_t_windows:
                tmp_df = df[(df["Spot frame"] >= start) & (df["Spot frame"] < end)]
                for label, label_df in tmp_df.groupby('Spot track ID'):
                    if len(label_df) == window_size:
                        new_label += 1
                        label_df["Spot track ID"] = new_label
                        control_df = control_df.append(label_df)
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

def concat_dfs_intensity(diff_df, con_df, diff_t_window=None, con_t_windows=None):
    def set_indexes(df, target, max_val):
        df["label"] = df["label"] + max_val
        max_val = df["label"].max() + 1
        df['target'] = np.array([target for i in range(len(df))])
        return df, max_val

    max_val = 0
    diff_start, diff_end = diff_t_window
    window_size = diff_end - diff_start

    # Erk video
    # Cut the needed time window
    diff_df = diff_df[(diff_df["frame"] >= diff_start) & (diff_df["frame"] < diff_end)]

    # control video
    # Cut the needed time window
    control_df = pd.DataFrame()
    new_label = max(con_df['label'].unique()) + 1
    for start, end in con_t_windows:
        tmp_df = con_df[(con_df["frame"] >= start) & (con_df["frame"] < end)]
        for label, label_df in tmp_df.groupby('label'):
            if len(label_df) == window_size:
                new_label += 1
                label_df["label"] = new_label
                control_df = control_df.append(label_df)
    con_df = control_df.copy()

    diff_df, max_val = set_indexes(diff_df, target=True, max_val=max_val)
    con_df, _ = set_indexes(con_df, target=False, max_val=max_val)
    total_df = pd.concat([diff_df, con_df], ignore_index=True)
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
    true_prob = []

    step_size = 1 if moving_window else window
    for i in range(0, len(track), step_size):
        if i + window > len(track):
            break
        track_portion = track[i:i + window]
        X = extract_distinct_features(df=track_portion, feature_list=feature_list)
        probs = clf.predict_proba(X)
        true_prob.append(probs[0][1])
        print(f"track portion: [{i}:{i + window}]")
        print(clf.classes_)
        print(probs)

    return true_prob

def evaluate(clf, X_test, y_test):
    predicted = cross_val_predict(clf, X_test, y_test, cv=5)
    report = classification_report(y_test, predicted)
    auc_score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(report)
    print(auc_score)
    return report, auc_score

def get_unique_indexes(y):
    idxs = y.index.unique()
    lst = [y[idx] if isinstance(y[idx], np.bool_) else y[idx].iloc[0] for idx in idxs]
    y_new = pd.Series(lst, index=idxs).sort_index()
    return y_new

def drop_columns_intensity(df):
    return df[['min', 'max', 'mean', 'sum', 'label', 'frame', 'target']]

def drop_columns_motility(df, motility, visual):
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

def normalize_intensity(df):
    columns = [e for e in list(df.columns) if e not in ('frame', 'label', 'target')]
    scaler = StandardScaler()  # create a scaler
    df[columns] = scaler.fit_transform(df[columns])  # transform the feature
    return df

def normalize_motility(df, motility):
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
    return df