import sys
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

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


def train_model_compare_algorithms(X_train, y_train, X_test, y_test, dir_path):
    models = []
    models.append(('RF', RandomForestClassifier()))
    models.append(('GB', GradientBoostingClassifier()))
    models.append(('LR', LogisticRegression()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('SVM', SVC(probability=True)))

    txt_file = open(dir_path + '/clf_comparison.txt', 'a')
    for name, model in models:
        model.fit(X_train, y_train)
        report, auc_score = evaluate(model, X_test, y_test)

        # save AUC score
        txt_file.write(f"classifier: {name}, auc score: {auc_score}\n")
        print(f"classifier: {name}, auc score: {auc_score}")
    txt_file.close()


def train_model(X_train, y_train, modality):
    params_dict = {"motility": {'max_depth': 12, 'min_samples_leaf': 1, 'n_estimators': 100},
                   "actin_intensity": {'max_depth': 20, 'min_samples_leaf': 1, 'n_estimators': 200}}
    params = params_dict.get(modality) if params_dict.get(modality) is not None else {}

    clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)
    return clf


def extract_distinct_features(df, feature_list, column_id="Spot track ID", column_sort="Spot frame"):
    df = extract_features(df, column_id=column_id, column_sort=column_sort)  # , show_warnings=False
    impute(df)
    return df[feature_list]


def split_data_to_time_portions(data, track_len):
    data = data.drop_duplicates(subset=["Spot track ID", "Spot frame"])  # remove duplicates
    time_windows = data.sort_values("Spot frame")['Spot frame'].unique()
    time_windows_strides = list(mit.windowed(time_windows, n=track_len, step=1))
    t_portion_lst = [data[data["Spot frame"].isin(time_windows_strides[i])] for i in range(len(time_windows_strides))]

    return t_portion_lst


def calc_state_trajectory(transformed_tracks_df, clf, n_frames=260):
    df_score = pd.DataFrame(columns=[i for i in range(n_frames)])
    for track_id, track in transformed_tracks_df.groupby("Spot track ID"):
        spot_frames = list(track.sort_values("Spot frame")["Spot frame"])
        diff_score = {"Spot track ID": track_id}
        try:
            for t in spot_frames:
                probs = clf.predict_proba(track[track["Spot frame"] == t].drop(["Spot track ID", "Spot frame"], axis=1))
                diff_score[t] = pd.to_numeric(probs[0][1], downcast='float')

            diff_score_df = pd.DataFrame(diff_score, index=[0])
            df_score = pd.concat([df_score, diff_score_df], ignore_index=True, sort=False)
            # df_score = df_score.append(diff_score, ignore_index=True, sort=False)
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
    diff_df = diff_df[diff_df["Spot frame"] == diff_end]
    print("size of diff_df: ", diff_df.shape)

    for label, label_df in diff_df.groupby('Spot track ID'):
        # new_diff_df = new_diff_df.append(label_df)
        new_diff_df = pd.concat([new_diff_df, label_df], ignore_index=True)

    # control video
    # Cut the needed time window
    control_df = pd.DataFrame()
    new_label = max(con_df['Spot track ID'].unique()) + 1
    for start, end in con_t_windows:
        tmp_df = con_df[con_df["Spot frame"] == end]
        for label, label_df in tmp_df.groupby('Spot track ID'):
            new_label += 1
            label_df["Spot track ID"] = new_label
            # control_df = control_df.append(label_df)
            control_df = pd.concat([control_df, label_df], ignore_index=True)
    con_df = control_df.copy()
    print("size of con_df: ", con_df.shape)

    new_diff_df, max_val = set_indexes(new_diff_df, target=True, max_val=max_val)
    con_df, _ = set_indexes(con_df, target=False, max_val=max_val)
    total_df = pd.concat([new_diff_df, con_df], ignore_index=True)
    return total_df


if __name__ == '__main__':
    print("diff_tracker_utils")
