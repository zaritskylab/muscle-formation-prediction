import pandas as pd
import joblib


def load_clean_rows(file_path):
    df = pd.read_csv(file_path, encoding="cp1252")
    df = df.drop(labels=range(0, 2), axis=0)

    return df


def load_data(dir_name):
    clf = joblib.load(dir_name + "/clf.joblib")
    x_train = pd.read_csv(dir_name + "/" + "X_train", encoding="cp1252")
    x_test = pd.read_csv(dir_name + "/" + "X_test", encoding="cp1252")
    y_train = pd.read_csv(dir_name + "/" + "y_train", encoding="cp1252")
    y_test = pd.read_csv(dir_name + "/" + "y_test", encoding="cp1252")

    return clf, x_train, x_test, y_train, y_test


def get_tracks_list(int_df, target):
    if target:
        int_df['target'] = target
    tracks = list()
    for label, labeled_df in int_df.groupby('Spot track ID'):
        tracks.append(labeled_df)
    return tracks


def save_data(dir_name, clf=None, X_train=None, X_test=None, y_train=None, y_test=None):
    if X_train is not None:
        X_train.to_csv(dir_name + "/" + "X_train")
    if X_test is not None:
        X_test.to_csv(dir_name + "/" + "X_test")
    if y_test is not None:
        y_test.to_csv(dir_name + "/" + "y_test")
    if y_train is not None:
        y_train.to_csv(dir_name + "/" + "y_train")
    if clf is not None:
        joblib.dump(clf, dir_name + "/" + "clf.joblib")


def get_tracks(file_path, manual_tagged_list, target=1):
    df_s = load_clean_rows(file_path)
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
    data = df_s[df_s["manual"] == 1] if manual_tagged_list else df_s
    tracks_s = get_tracks_list(data, target=target)
    return df_s, tracks_s
