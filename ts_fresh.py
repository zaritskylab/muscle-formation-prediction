import itertools
from random import sample

import sklearn
from sklearn import clone
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# from xgboost import XGBClassifier
import joblib
from DataPreprocessing.load_tracks_xml import *
from tsfresh import extract_features, extract_relevant_features, select_features
import tsfresh
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, load_robot_execution_failures
from tsfresh.utilities.dataframe_functions import impute
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, train_test_split, StratifiedKFold, GridSearchCV
import pickle
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import seaborn as sns
from ts_interpretation import plot_cell_probability

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

warnings.filterwarnings("ignore", message="A value is trying to be set on a copy of a slice from a DataFrame")


def drop_columns(df, motility=True, intensity=True, basic=False):
    intensity_features = ['median_intensity', 'min_intensity', 'max_intensity', 'mean_intensity',
                          'total_intensity', 'std_intensity', 'contrast', 'snr', 'w', 'q']
    basic_features = ['t_stamp', 'z', 'spot_id', 'q']
    motility_features = ['x', 'y']
    to_remove = []
    to_remove.extend(basic_features) if not basic else to_remove.extend([])
    to_remove.extend(intensity_features) if not intensity else to_remove.extend([])
    to_remove.extend(motility_features) if not motility else to_remove.extend([])
    df = df[df.columns.drop(to_remove)]
    return df


def get_unique_indexes(y):
    idxs = y.index.unique()
    lst = [y[idx] if isinstance(y[idx], np.bool_) else y[idx].iloc[0] for idx in idxs]
    y_new = pd.Series(lst, index=idxs).sort_index()
    return y_new


def normalize_tracks(df, motility=False, intensity=False):
    if motility:
        for label in df.label:
            to_reduce_x = df[df.label == label].iloc[0].x
            to_reduce_y = df[df.label == label].iloc[0].y
            df.loc[(df.label == label).values, "x"] = df[df.label == label].x.apply(lambda num: num - to_reduce_x)
            df.loc[(df.label == label).values, "y"] = df[df.label == label].y.apply(lambda num: num - to_reduce_y)

    if intensity:
        columns = list(df.columns)
        columns.remove("t")
        columns.remove("label")
        columns.remove("target")

        # create a scaler
        scaler = StandardScaler()
        # transform the feature
        df[columns] = scaler.fit_transform(df[columns])
    return df


def concat_dfs(min_time_diff, lst_videos):
    min_time_diff = min_time_diff
    max_val = 0
    total_df = pd.DataFrame()
    for i in lst_videos:
        xml_path = r"data/tracks_xml/0104/Experiment1_w1Widefield550_s{}_all_0104.xml".format(i)
        xml_path = xml_path if os.path.exists(xml_path) else "muscle-formation-diff/" + xml_path
        _, df = load_tracks_xml(xml_path)
        df.label = df.label + max_val
        max_val = df["label"].max() + 1
        target = False
        if i in (3, 4, 5, 6, 11, 12):
            if df["t"].max() >= min_time_diff:
                target = True
        # target = True if i in (3, 4, 5, 6, 11, 12) else False
        df['target'] = np.array([target for i in range(len(df))])
        total_df = pd.concat([total_df, df], ignore_index=True)
    return total_df


def long_extract_features(df):
    y = pd.Series(df['target'])
    y.index = df["label"]
    y = get_unique_indexes(y)
    df = df[df.columns.drop(['target'])]
    extracted_features = extract_features(df, column_id="label", column_sort="t")
    impute(extracted_features)
    features_filtered = select_features(extracted_features, y)
    return features_filtered


def short_extract_features(df, y):
    # df = df[df.columns.drop(['target'])]
    features_filtered_direct = extract_relevant_features(df, y, column_id="label", column_sort='t', show_warnings=False,
                                                         n_jobs=8)
    return features_filtered_direct


def feature_importance(clf, feature_names, path):
    sorted_idx = clf.feature_importances_.argsort()
    plt.barh(clf.feature_importances_[sorted_idx])
    plt.xlabel("Random Forest Feature Importance")
    plt.title('Feature Importance Plot')
    plt.savefig(path + "/feature importance.png")
    plt.show()


def get_single_cells_diff_score_plot(tracks, clf, features_filtered_direct):
    all_probs = []
    for cell in (5, 7, 8, 9, 10):  # , 14, 16, 19, 27, 30
        true_prob = []
        n = 20
        for i in range(0, len(tracks[cell]), n):
            x = 0
            track_portion = tracks[cell][i:i + n]
            extracted_features = extract_features(track_portion, column_id="label", column_sort="t",
                                                  show_warnings=False, n_jobs=8)
            impute(extracted_features)
            X = extracted_features[features_filtered_direct.columns]
            probs = clf.predict_proba(X)
            true_prob.append(probs[0][1])
            print(f"track portion: [{i}:{i + n}]")
            print(clf.classes_)
            print(probs)
            track_len = len(tracks[cell])
            label_time = [(tracks[cell].iloc[val]['t'] / 60) / 60 for val in range(0, track_len, n)]

        plt.plot(range(0, track_len, n), true_prob)
        plt.xticks(range(0, track_len, n), labels=np.around(label_time, decimals=1))
        plt.ylim(0, 1, 0.1)
        plt.title(f"probability to differentiation over time- diff, cell #{cell}")
        plt.xlabel("time [h]")
        plt.ylabel("prob")
        plt.grid()
        plt.show()
        all_probs.append(true_prob)
        return all_probs


def train(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier()
    # clf.fit(X_train, y_train)
    # feature_importance(clf)
    predicted = cross_val_predict(clf, X_test, y_test, cv=5)
    report = classification_report(y_test, predicted)
    auc_score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(report)
    print(auc_score)
    return clf, report, auc_score


def plot_roc(clf, X_test, y_test, path):
    fig = plt.figure(figsize=(20, 6))
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


def get_x_y(min_length, max_length, min_time_diff, lst_videos, motility, intensity):
    df = concat_dfs(min_time_diff, lst_videos, motility=motility, intensity=intensity)
    df = drop_columns(df, motility=motility, intensity=intensity)
    df = normalize_tracks(df, motility=motility, intensity=intensity)

    occurrences = df.groupby(["label"]).size()
    labels = []
    for label in df["label"].unique():
        if (min_length <= occurrences[label] <= max_length):
            labels.append(label)
    df = df[df["label"].isin(labels)]

    y = pd.Series(df['target'])
    y.index = df["label"]
    y = get_unique_indexes(y)
    df = df.drop("target", axis=1)
    return df, y


def nested_cross_validation(X, y):
    '''
    :param X:
    :param y:
    :return:
    '''
    k = 5
    f1_xgb = 0
    f1_rf = 0
    xgb_best_prms = dict()
    rf_best_prms = dict()

    cv_outer = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
    for train_idx, val_idx in cv_outer.split(X, y):
        train_data, val_data = X.iloc[train_idx], X.iloc[val_idx]
        train_target, val_target = y.iloc[train_idx], y.iloc[val_idx]

        cv_inner = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)

        rf = RandomForestClassifier()
        space_RF = {'n_estimators': np.arange(10, 100, 5).tolist(), 'max_depth': [2, 4, 8, 16, 32, 64]}
        gd_search = GridSearchCV(rf, space_RF, scoring='f1', n_jobs=-1, cv=cv_inner, refit=True).fit(train_data,
                                                                                                     train_target)
        predictions = gd_search.predict(val_data)
        rf_best_prms = gd_search.best_params_ if sklearn.metrics.f1_score(val_target,
                                                                          predictions) > f1_rf else rf_best_prms

        xgb = XGBClassifier(use_label_encoder=False, verbosity=0)
        space_XGB = {'n_estimators': np.arange(10, 100, 5).tolist(), 'max_depth': [2, 4, 8, 16, 32, 64],
                     'learning_rate': [0.1, 0.05, 0.01]}
        gd_search = GridSearchCV(xgb, space_XGB, scoring='f1', n_jobs=-1, cv=cv_inner, refit=True).fit(train_data,
                                                                                                       train_target)
        predictions = gd_search.predict(val_data)
        xgb_best_prms = gd_search.best_params_ if sklearn.metrics.f1_score(val_target,
                                                                           predictions) > f1_rf else xgb_best_prms

    return rf_best_prms, xgb_best_prms


def evaluate(y_test, y_hat, scores):
    cm = confusion_matrix(y_test, y_hat)
    scores["accuracy"] = sklearn.metrics.accuracy_score(y_test, y_hat)
    scores["f1_score"] = sklearn.metrics.f1_score(y_test, y_hat)
    scores["specificity"] = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    scores["sensitivity"] = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    scores["AUROC"] = sklearn.metrics.roc_auc_score(y_test, y_hat)


def retrain_model(model, params, X_train, X_test, y_train, y_test):
    '''
    :param model:
    :param params:
    :param X:
    :param y:
    :return:
    '''
    scores = {'accuracy': [], 'f1_score': [], 'sensitivity': [], 'specificity': [], 'AUROC': []}
    avg_scores = {}
    avg_stds = {}
    print("Retraining {}".format(model))
    for i in range(10):
        model = clone(model)
        model.set_params(**params)
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        evaluate(y_test, y_hat, scores)
    for metric, lst in scores.items():
        avg_scores[metric] = str(np.mean(lst)) + "+-" + str(np.std(lst))
        avg_stds[metric] = np.std(lst)
    print(f"Average scores for the model : {avg_scores}")
    return model


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


def extract_distinct_features(df, feature_list):
    df = extract_features(df, column_id="label", column_sort="t")
    impute(df)
    return df[feature_list]


def get_path(path):
    return path if os.path.exists(path) else "muscle-formation-diff/" + path


if __name__ == '__main__':
    print("Let's go!")

    motility = False
    intensity = True
    min_length = 0
    max_length = 950
    min_time_diff = 0

    exp_1 = [1, 2, 3, 4]
    exp_2 = [5, 6, 7, 8]
    exp_3 = [9, 10, 11, 12]
    # exp_1=[1]
    # exp_2= [3]
    # exp_3= [1,3]

    train_video_lists = [list(itertools.chain(exp_1, exp_2)),
                         list(itertools.chain(exp_1, exp_3)), list(itertools.chain(exp_2, exp_3))]
    test_video_lists = [exp_3, exp_2, exp_1]

    auc_scores = []

    for (exp_train_lst, exp_test_lst) in zip(train_video_lists, test_video_lists):

        dir_name = f"train_set {exp_train_lst}_ test_set {exp_test_lst}_ motility-{motility}_intensity-{intensity}"
        print(dir_name)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        X_train, y_train = get_x_y(min_length=min_length, max_length=max_length, min_time_diff=min_time_diff,
                                   lst_videos=exp_train_lst, motility=motility, intensity=intensity)
        # extract features using ts-fresh
        X_train = short_extract_features(X_train, y_train)

        X_test, y_test = get_x_y(min_length=min_length, max_length=max_length, min_time_diff=min_time_diff,
                                 lst_videos=exp_test_lst, motility=motility, intensity=intensity)
        X_test = extract_distinct_features(df=X_test, feature_list=X_train.columns)

        clf, report, auc_score = train(X_train, X_test, y_train, y_test)
        plot_roc(clf=clf, X_test=X_test, y_test=y_test, path=dir_name)
        principal_df, pca = build_pca(3, X_test)
        plot_pca(principal_df, pca, dir_name)
        feature_importance(clf, X_train.columns, dir_name)

        auc_scores.append(auc_score)

        pickle.dump(X_train, open(dir_name + "/" + "X_train", 'wb'))
        pickle.dump(X_test, open(dir_name + "/" + "X_test", 'wb'))
        joblib.dump(clf, dir_name + "/" + "clf.joblib")

        txt_file = open(dir_name + '/info.txt', 'a')
        txt_file.write(f"classification report: {report}\n auc score: {auc_score}")
        txt_file.close()

    print(f"AVG AUC score: {np.mean(auc_scores)}")

# rf_best_prms, xgb_best_prms = nested_cross_validation(X_train, y_train)
# xgb_model = retrain_model(XGBClassifier(), xgb_best_prms, X_train, X_test, y_train, y_test)
# rf_model = retrain_model(RandomForestClassifier(), rf_best_prms, X_train, X_test, y_train, y_test)
