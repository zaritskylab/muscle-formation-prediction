import pickle
from collections import Counter
from utils.diff_tracker_utils import *
from utils.data_load_save import *
from utils.plots_functions_utils import *
from utils import data_load_save as load_save_utils
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import consts
import numpy as np
from tsfresh import select_features
import sys
import datetime
import os


def clean_redundant_columns(df):
    remove_cols = []
    cols_to_remove = ["target"]
    for col_to_remove in cols_to_remove:
        for col_name in df.columns:
            if col_name.startswith(col_to_remove):
                remove_cols.append(col_name)
    df = df.drop(columns=remove_cols)
    return df


def get_to_run(path, to_run, con_train_num=None, con_test_num=None, diff_train_num=None,
               diff_test_num=None, registration="no_reg_", local_density=False, impute_func="impute",
               impute_methodology="ImputeSingleCell"):
    path_prefix = path + f"/data/mastodon/ts_transformed_new/{to_run}/{impute_methodology}_{impute_func}/"
    end = f"_imputed reg={registration}, local_den={local_density}, win size 16"

    diff_df_train, con_df_train, con_df_test, diff_df_test = None, None, None, None

    if diff_train_num:
        diff_df_train = pd.read_csv(path_prefix + f"S{diff_train_num}" + end, encoding="cp1252", index_col=[0])
        print("diff train len", len(diff_df_train))

    if con_train_num:
        con_df_train = pd.read_csv(path_prefix + f"S{con_train_num}" + end, encoding="cp1252", index_col=[0])
        print("con_df_train len", len(con_df_train))

    if con_test_num:
        con_df_test = pd.read_csv(path_prefix + f"S{con_test_num}" + end, encoding="cp1252", index_col=[0])

    if diff_test_num:
        diff_df_test = pd.read_csv(path_prefix + f"S{diff_test_num}" + end, encoding="cp1252", index_col=[0])

    return diff_df_train, con_df_train, con_df_test, diff_df_test


def prep_data(diff_df, con_df, diff_t_window, con_t_windows):
    print("concatenating control data & ERKi data")
    df = concat_dfs(diff_df, con_df, diff_t_window,
                          con_t_windows)
    df = df.sample(frac=1).reset_index(drop=True)
    print("shape after concat_dfs", df.shape)
    print(df.isna().sum())

    df = df.dropna(axis=1)
    print("shape after dropna", df.shape)

    df.index = df['Spot track ID']
    y = pd.Series(df['target'])
    y.index = df['Spot track ID']
    df = df.drop(["target", "Spot frame", "Spot track ID"], axis=1)
    return df, y


def evaluate_clf(dir_path, clf, X_test, y_test, y_train):
    report, auc_score = evaluate(clf, X_test, y_test)

    # load the model & train set & test set
    # clf, X_train, X_test, y_train, y_test = utils.load_data(dir_path)

    # plot ROC curve
    plot_roc(clf=clf, X_test=X_test, y_test=y_test, path=dir_path)

    # perform PCA analysis
    principal_df, pca = build_pca(3, X_test)
    plot_pca(principal_df, pca, dir_path)

    # calculate feature importance
    # utils.feature_importance(clf, X_train.columns, dir_path)

    # save classification report & AUC score
    txt_file = open(dir_path + '/info.txt', 'a')
    txt_file.write(f"classification report: {report}\n auc score: {auc_score}\n train samples:{Counter(y_train)}")

    txt_file.close()

    return auc_score


def remove_target_cols(df):
    remove_cols = []
    cols_to_remove = ["target", "Spot track ID"]
    for col_name in df.columns:
        if col_name.startswith("target"):
            remove_cols.append(col_name)
    df = df.drop(columns=remove_cols)
    return df


def get_confidence_interval(con_df_test, diff_df_test, dir_path, X_train, y_train, X_test, y_test, cols, n_runs=10):
    print(f"training- {n_runs} runs")

    auc_lst = []
    for i in range(n_runs):
        clf = train_model(X_train, y_train)
        auc = evaluate_clf(dir_path, clf, X_test, y_test, y_train)
        auc_lst.append(auc)

        print("calc avg prob")
        df_score_con = calc_prob(con_df_test[cols], clf, n_frames=260)
        df_score_dif = calc_prob(diff_df_test[cols], clf, n_frames=260)

        pickle.dump(df_score_con, open(dir_path + f"/df_prob_w=30, video_num={con_test_n} run_{i}", 'wb'))
        pickle.dump(df_score_dif, open(dir_path + f"/df_prob_w=30, video_num={diff_test_n} run_{i}", 'wb'))

    print(auc_lst)
    try:
        np.save(dir_path + f'/auc_lst{modality}.npy', auc_lst, allow_pickle=True)
    except:
        pass

    sns.distplot(auc_lst)
    plt.xlabel("auc")
    plt.savefig(dir_path + f"/aucs_{modality}, {n_runs}")
    plt.show()
    plt.clf()


def plot_avg_conf(df_con, df_diff, mot_int, path=""):
    def plot(df, color1, color2):
        avg_vals_diff = ([df[col].mean() for col in df.columns])
        std_vals_diff = ([df[col].std() for col in df.columns])
        p_std = np.asarray(avg_vals_diff) + np.asarray(std_vals_diff)
        m_std = np.asarray(avg_vals_diff) - np.asarray(std_vals_diff)

        plt.plot([i * 5 / 60 for i in range(len(avg_vals_diff))], avg_vals_diff, color=color1)
        plt.fill_between([i * 5 / 60 for i in range(len(avg_vals_diff))], m_std, p_std, alpha=0.5, color=color2)

    plot(df_diff, "DarkOrange", "Orange")
    plot(df_con, "blue", "blue")
    plt.legend(["Erk avg", "Control avg", "Erk std", "Control std"])
    plt.xlabel("time (h)")
    plt.ylabel("avg confidence")
    plt.title(f"avg differentiation confidence over time ({mot_int})")
    plt.plot([i * 5 / 60 for i in range(260)], [0.5 for i in range(260)], color="black", linestyle="--")
    plt.savefig(path)
    plt.show()
    plt.clf()


if __name__ == '__main__':
    path = consts.cluster_path
    modality = sys.argv[1]
    registration_method = sys.argv[2]
    impute_func = sys.argv[3]
    impute_methodology = sys.argv[4]

    print(f"running: modality={modality}, "
          f"local density={consts.local_density}, "
          f"reg={registration_method}, "
          f"impute func= {impute_func},"
          f"impute_methodology= {impute_methodology}")

    for con_train_n, diff_train_n, con_test_n, diff_test_n in [(1, 5, 1, 3), (2, 3, 1, 5)]:
        diff_df_train, con_df_train, _, _ = get_to_run(path=path, to_run=modality,
                                                       con_train_num=con_train_n,
                                                       diff_train_num=diff_train_n,
                                                       con_test_num=None,
                                                       diff_test_num=None,
                                                       local_density=consts.local_density,
                                                       registration=registration_method,
                                                       impute_func=impute_func,
                                                       impute_methodology=impute_methodology)

        today = datetime.datetime.now()
        dir_path = f"{today.strftime('%d-%m-%Y')}-{modality} local dens-{consts.local_density}, s{con_train_n}, s{diff_train_n} train, reg {registration_method}"
        second_dir = f"{consts.diff_window} ERK, {consts.con_window} con track len {consts.tracks_len}, impute_func-{impute_methodology}_{impute_func}"
        os.makedirs(dir_path, exist_ok=True)
        os.makedirs(os.path.join(dir_path, second_dir), exist_ok=True)
        dir_path += "/" + second_dir

        print("preparing data")
        X_train, y_train = prep_data(diff_df=diff_df_train, con_df=con_df_train, diff_t_window=consts.diff_window,
                                     con_t_windows=consts.con_window)
        del diff_df_train
        del con_df_train

        print("deleted diff_df_train, con_df_train")
        X_train = select_features(X_train, y_train)

        clf = train_model(X_train, y_train)
        cols = list(X_train.columns)
        load_save_utils.save_data(dir_path, X_train=X_train)
        del X_train
        _, _, con_df_test, diff_df_test = get_to_run(path=path, to_run=modality,
                                                     con_train_num=None,
                                                     con_test_num=con_test_n,
                                                     diff_train_num=None,
                                                     diff_test_num=diff_test_n,
                                                     local_density=consts.local_density,
                                                     registration=registration_method,
                                                     impute_func=impute_func,
                                                     impute_methodology=impute_methodology)

        X_test, y_test = prep_data(diff_df=diff_df_test, con_df=con_df_test, diff_t_window=consts.diff_window,
                                   con_t_windows=consts.con_window)

        X_test = X_test[cols]

        cols.extend(["Spot track ID", "Spot frame"])

        load_save_utils.save_data(dir_path, y_train=y_train, X_test=X_test, y_test=y_test, clf=clf)
        auc = evaluate_clf(dir_path, clf, X_test, y_test, y_train)
        del X_test
        del y_test

        print("calc avg prob")
        df_score_con = calc_prob(con_df_test[cols], clf, n_frames=260)
        df_score_dif = calc_prob(diff_df_test[cols], clf, n_frames=260)

        plot_avg_conf(df_score_con.drop("Spot track ID", axis=1), df_score_dif.drop("Spot track ID", axis=1),
                      mot_int=modality,
                      path=dir_path + f"/avg conf s{con_test_n}, s{diff_test_n}.png")
