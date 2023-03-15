import os
import sys
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, brier_score_loss
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
import seaborn as sns
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
    models.append(('RF', RandomForestClassifier())) #**{'max_depth': 12, 'min_samples_leaf': 1, 'n_estimators': 100}
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
    params_dict = {"motility":
                       {'max_depth': 12, 'min_samples_leaf': 1, 'n_estimators': 100}, #'class_weight': 'balanced',
                   "actin_intensity": {'class_weight': None, 'max_depth': 20, 'min_samples_leaf': 1,
                                       'n_estimators': 200}}
    params = params_dict.get(modality) if params_dict.get(modality) is not None else {}
    clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)
    return clf


def plot_calibration_curve(y_true, not_calibrated_probs, calibrated_probs_sigmoid, calibrated_probs_isotonic, save_path,
                           clf_score, clf_sigmoid_score, clf_isotonic_score):
    plt.figure(figsize=(10, 4))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

    x, y = calibration_curve(y_true, not_calibrated_probs, n_bins=10, normalize=False)
    plt.plot(y, x, marker='.', label=f'original, brier={round(clf_score, 3)}')

    x, y = calibration_curve(y_true, calibrated_probs_sigmoid, n_bins=10, normalize=False)
    plt.plot(y, x, marker='.', label=f'sigmoid calibration, brier={round(clf_sigmoid_score, 3)}')

    x, y = calibration_curve(y_true, calibrated_probs_isotonic, n_bins=10, normalize=False)
    plt.plot(y, x, marker='.', label=f'isotonic calibration, brier={round(clf_isotonic_score, 3)}')

    plt.title("calibrated vs not calibrated curves", fontsize=20)
    plt.xlabel('Mean predicted probability for each bin', fontsize=14)
    plt.ylabel('fraction of positive classes in each bin', fontsize=14)
    plt.legend()
    plt.savefig(save_path + "/calibration_curve.eps", format="eps")
    plt.show()
    plt.clf()


def plot_predicted_vs_empirical_probs(x_data, y_data, not_calibrated_clf, calibrated_clf, save_path):
    # Plot Predicted Probabilities vs Empirical Probabilities
    plt.figure(figsize=(20, 10))

    plt.subplot(3, 1, 1)
    sns.histplot(data=not_calibrated_clf.predict_proba(x_data)[:, 1])
    plt.title(f"test predicted probabilities (not calibrated)", fontsize=20)

    plt.subplot(3, 1, 2)
    sns.histplot(data=calibrated_clf.predict_proba(x_data)[:, 1])
    plt.title(f"test predicted probabilities (calibrated)", fontsize=20)

    plt.subplot(3, 1, 3)
    sns.histplot(data=y_data.astype(int))
    plt.title(f"test true values", fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path + "/calibration predicted probabilities.eps", format="eps")
    plt.show()
    plt.clf()


def calibrate_model(x_train, y_train, x_test, y_test, clf, path, modality):
    # from imblearn.under_sampling import RandomUnderSampler
    # undersample = RandomUnderSampler(random_state=42)
    # print(y_train.value_counts(), y_test.value_counts())
    # x_train, y_train = undersample.fit_resample(x_train, y_train)  # todo remove
    # x_test, y_test = undersample.fit_resample(x_test, y_test)  # todo remove
    #
    # print(y_train.value_counts(),y_test.value_counts() )
    # params_dict = {"motility":
    #                    {'bootstrap': False, 'class_weight': 'balanced', 'max_depth': 24, 'max_features': 'auto',
    #                     'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 200},
    #                "actin_intensity": {'bootstrap': False, 'class_weight': 'balanced', 'max_depth': 40, 'max_features': 'auto',
    #                                    'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
    #                }
    # params = params_dict.get(modality) if params_dict.get(modality) is not None else {}
    # base_clf = RandomForestClassifier(**params)
    # base_clf.fit(x_train, y_train)
    #
    # not_calibrated_probs = base_clf.predict_proba(x_test)[:, 1]
    #
    # calibrated_model_sigmoid = CalibratedClassifierCV(base_clf, cv=5, method="sigmoid")
    # calibrated_model_sigmoid.fit(x_train, y_train)
    # calibrated_probs_sigmoid = calibrated_model_sigmoid.predict_proba(x_test)[:, 1]
    #
    # calibrated_model_isotonic = CalibratedClassifierCV(base_clf, cv=5, method="isotonic")
    # calibrated_model_isotonic.fit(x_train, y_train)
    # calibrated_probs_isotonic = calibrated_model_isotonic.predict_proba(x_test)[:, 1]
    #
    # # calculate brier score:
    # clf_score = brier_score_loss(y_test, not_calibrated_probs)
    # print("With no calibration: %1.3f" % clf_score, flush=True)
    # clf_sigmoid_score = brier_score_loss(y_test, calibrated_probs_sigmoid)
    # print("With sigmoid calibration: %1.3f" % clf_sigmoid_score, flush=True)
    # clf_isotonic_score = brier_score_loss(y_test, calibrated_probs_isotonic)
    # print("With isotonic calibration: %1.3f" % clf_isotonic_score, flush=True)
    #
    # # calibrate with best calibration method
    # # calibrated_model = calibrated_model_isotonic if clf_isotonic_score <= clf_sigmoid_score else calibrated_model_sigmoid
    # calibrated_model = calibrated_model_sigmoid
    # print(calibrated_model)
    #
    # plot_calibration_curve(y_test, not_calibrated_probs, calibrated_probs_sigmoid, calibrated_probs_isotonic, path,
    #                        clf_score, clf_sigmoid_score, clf_isotonic_score)
    # plot_predicted_vs_empirical_probs(x_test, y_test, base_clf, calibrated_model, path)
    from imblearn.under_sampling import RandomUnderSampler
    undersample = RandomUnderSampler(random_state=42)
    print("train", y_train.value_counts())
    print("test", y_test.value_counts())
    x_train, y_train = undersample.fit_resample(x_train, y_train)  # todo remove
    x_test, y_test = undersample.fit_resample(x_test, y_test)  # todo remove
    print("train", y_train.value_counts())
    print("test", y_test.value_counts())

    params_dict = {"motility":
                       {'class_weight': 'balanced', 'max_depth': 12, 'min_samples_leaf': 1, 'n_estimators': 100},
                   "actin_intensity": {'class_weight': None, 'max_depth': 20, 'min_samples_leaf': 1,
                                       'n_estimators': 200}}

    params = params_dict.get(modality) if params_dict.get(modality) is not None else {}
    base_clf = RandomForestClassifier(**params)
    base_clf.fit(x_train, y_train)
    not_calibrated_probs = base_clf.predict_proba(x_test)[:, 1]

    calibrated_model_sigmoid = CalibratedClassifierCV(base_clf, cv=5, method="sigmoid")
    calibrated_model_sigmoid.fit(x_train, y_train)
    calibrated_probs_sigmoid = calibrated_model_sigmoid.predict_proba(x_test)[:, 1]
    calibrated_model_isotonic = CalibratedClassifierCV(base_clf, cv=5, method="isotonic")
    calibrated_model_isotonic.fit(x_train, y_train)
    calibrated_probs_isotonic = calibrated_model_isotonic.predict_proba(x_test)[:, 1]

    # calculate brier score:
    clf_score = brier_score_loss(y_test, not_calibrated_probs)
    print("With no calibration: %1.3f" % clf_score, flush=True)
    clf_sigmoid_score = brier_score_loss(y_test, calibrated_probs_sigmoid)
    print("With sigmoid calibration: %1.3f" % clf_sigmoid_score, flush=True)
    clf_isotonic_score = brier_score_loss(y_test, calibrated_probs_isotonic)
    print("With isotonic calibration: %1.3f" % clf_isotonic_score, flush=True)

    # calibrate with best calibration method
    calibrated_model = calibrated_model_sigmoid
    print(calibrated_model)
    plot_calibration_curve(y_test, not_calibrated_probs, calibrated_probs_sigmoid, calibrated_probs_isotonic, path,
                           clf_score, clf_sigmoid_score, clf_isotonic_score)
    plot_predicted_vs_empirical_probs(x_test, y_test, base_clf, calibrated_model, path)
    return calibrated_model


def calibrate_model2(x_train, y_train, x_test, y_test, clf, path, modality):
    from imblearn.under_sampling import RandomUnderSampler
    undersample = RandomUnderSampler(random_state=42)
    x_train, y_train = undersample.fit_resample(x_train, y_train)  # todo remove

    params_dict = {"motility":
                       {'bootstrap': False, 'max_depth': 24, 'max_features': 'auto',
                        'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 200},
                   "actin_intensity": {'bootstrap': False, 'max_depth': 40, 'max_features': 'auto',
                                       'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
                   }
    new_x_train, x_calib, new_y_train, y_calib = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

    params = params_dict.get(modality) if params_dict.get(modality) is not None else {}
    base_clf = RandomForestClassifier(**params)
    base_clf.fit(new_x_train, new_y_train)
    not_calibrated_probs = base_clf.predict_proba(x_test)[:, 1]

    # get best calibrated model:
    calibrated_model_sigmoid = CalibratedClassifierCV(base_clf, cv="prefit", method="sigmoid")
    calibrated_model_sigmoid.fit(x_calib, y_calib)
    calibrated_probs_sigmoid = calibrated_model_sigmoid.predict_proba(x_test)[:, 1]

    calibrated_model_isotonic = CalibratedClassifierCV(base_clf, cv="prefit", method="isotonic")
    calibrated_model_isotonic.fit(x_calib, y_calib)
    calibrated_probs_isotonic = calibrated_model_isotonic.predict_proba(x_test)[:, 1]

    # calculate brier score:
    clf_score = brier_score_loss(y_test, not_calibrated_probs)
    print("With no calibration: %1.3f" % clf_score, flush=True)
    clf_sigmoid_score = brier_score_loss(y_test, calibrated_probs_sigmoid)
    print("With sigmoid calibration: %1.3f" % clf_sigmoid_score, flush=True)
    clf_isotonic_score = brier_score_loss(y_test, calibrated_probs_isotonic)
    print("With isotonic calibration: %1.3f" % clf_isotonic_score, flush=True)

    # calibrate with best calibration method
    calibrated_model = calibrated_model_sigmoid
    print(calibrated_model)

    plot_calibration_curve(y_test, not_calibrated_probs, calibrated_probs_sigmoid, calibrated_probs_isotonic, path,
                           clf_score, clf_sigmoid_score, clf_isotonic_score)
    plot_predicted_vs_empirical_probs(x_test, y_test, base_clf, calibrated_model, path)

    return calibrated_model


# def calibrate_model(x_train, y_train, x_test, y_test, clf, path, modality):
#     from imblearn.under_sampling import RandomUnderSampler
#     undersample = RandomUnderSampler(random_state=42)
#     # x_test, y_test = undersample.fit_resample(x_test, y_test)
#
#     # x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.2, random_state=42)
#     # not_calibrated_probs = clf.predict_proba(x_val)[:, 1]
#
#     # get best calibrated model:
#     # calibrated_model_sigmoid = CalibratedClassifierCV(clf, cv="prefit", method="sigmoid")
#     # calibrated_model_sigmoid.fit(x_test, y_test)
#     # calibrated_probs_sigmoid = calibrated_model_sigmoid.predict_proba(x_val)[:, 1]
#     #
#     # calibrated_model_isotonic = CalibratedClassifierCV(clf, cv="prefit", method="isotonic")
#     # calibrated_model_isotonic.fit(x_test, y_test)
#     # calibrated_probs_isotonic = calibrated_model_isotonic.predict_proba(x_val)[:, 1]
#
#     params_dict = {"motility": {'bootstrap': True, 'max_depth': 8, 'max_features': 'auto', 'min_samples_leaf': 1,
#                                 'min_samples_split': 10, 'n_estimators': 100},
#                    # {'bootstrap': False, 'max_depth': 60, 'max_features': 'auto', 'min_samples_leaf': 1,
#                    #          'min_samples_split': 2,
#                    #          'n_estimators': 100},
#                    "actin_intensity": {'bootstrap': False, 'max_depth': 40, 'max_features': 'auto',
#                                        'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
#                    }
#     params = params_dict.get(modality) if params_dict.get(modality) is not None else {}
#     base_clf = RandomForestClassifier(**params)
#
#     x_test, y_test = undersample.fit_resample(x_test, y_test)  # todo remove
#
#     not_calibrated_probs = clf.predict_proba(x_test)[:, 1]
#
#     calibrated_model_sigmoid = CalibratedClassifierCV(base_clf, cv=5, method="sigmoid")
#     calibrated_model_sigmoid.fit(x_train, y_train)
#     calibrated_probs_sigmoid = calibrated_model_sigmoid.predict_proba(x_test)[:, 1]
#
#     calibrated_model_isotonic = CalibratedClassifierCV(base_clf, cv=5, method="isotonic")
#     calibrated_model_isotonic.fit(x_train, y_train)
#     calibrated_probs_isotonic = calibrated_model_isotonic.predict_proba(x_test)[:, 1]
#
#     y_val = y_test
#
#     # calculate brier score:
#     clf_score = brier_score_loss(y_val, not_calibrated_probs)
#     print("With no calibration: %1.4f" % clf_score, flush=True)
#     clf_sigmoid_score = brier_score_loss(y_val, calibrated_probs_sigmoid)
#     print("With sigmoid calibration: %1.4f" % clf_sigmoid_score, flush=True)
#     clf_isotonic_score = brier_score_loss(y_val, calibrated_probs_isotonic)
#     print("With isotonic calibration: %1.4f" % clf_isotonic_score, flush=True)
#
#     # calibrate with best calibration method
#     calibrated_model = calibrated_model_isotonic if clf_isotonic_score <= clf_sigmoid_score else calibrated_model_sigmoid
#
#     # plot calibration curve
#     plt.figure(figsize=(10, 4))
#     plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
#
#     x, y = calibration_curve(y_val, not_calibrated_probs, n_bins=10, normalize=True)
#     plt.plot(y, x, marker='.', label=f'original, brier={round(clf_score, 3)}')
#
#     x, y = calibration_curve(y_val, calibrated_probs_sigmoid, n_bins=10, normalize=True)
#     plt.plot(y, x, marker='.', label=f'sigmoid calibration, brier={round(clf_sigmoid_score, 3)}')
#
#     x, y = calibration_curve(y_val, calibrated_probs_isotonic, n_bins=10, normalize=True)
#     plt.plot(y, x, marker='.', label=f'isotonic calibration, brier={round(clf_isotonic_score, 3)}')
#
#     plt.title("calibrated vs not calibrated curves", fontsize=20)
#     plt.xlabel('Mean predicted probability for each bin', fontsize=14)
#     plt.ylabel('fraction of positive classes in each bin', fontsize=14)
#     plt.legend()
#     plt.savefig(path + "/calibration_curve.eps", format="eps")
#     plt.show()
#     plt.clf()
#
#     # Plot Predicted Probabilities vs Empirical Probabilities
#     plt.figure(figsize=(20, 10))
#
#     plt.subplot(3, 1, 1)
#     sns.histplot(data=clf.predict_proba(x_test)[:, 1])
#     plt.title(f"test predicted probabilities (not calibrated)", fontsize=20)
#
#     plt.subplot(3, 1, 2)
#     sns.histplot(data=calibrated_model.predict_proba(x_test)[:, 1])
#     plt.title(f"test predicted probabilities (calibrated)", fontsize=20)
#
#     plt.subplot(3, 1, 3)
#     sns.histplot(data=y_test.astype(int))
#     plt.title(f"test true values", fontsize=20)
#     plt.tight_layout()
#     plt.savefig(path + "/calibration predicted probabilities.eps", format="eps")
#     plt.show()
#     plt.clf()
#
#     return calibrated_model


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
        df_measures = pd.concat([df_measures, pd.DataFrame(data, index=[0])], ignore_index=True)
        # df_measures = df_measures.append(data, ignore_index=True)
    return df_measures


def get_local_densities_df(df_s, tracks_s, neighboring_distance=50):
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
        # local_densities = local_densities.append(track_local_density, ignore_index=True)
        local_densities = pd.concat([local_densities, track_local_density], ignore_index=True)

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
            # new_df = new_df.append(track, ignore_index=True)
            new_df = pd.concat([new_df, track], ignore_index=True)
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
