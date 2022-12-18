from collections import Counter
import os, sys

sys.path.append('/sise/home/shakarch/muscle-formation-diff')
sys.path.append(os.path.abspath('..'))

sys.path.append(os.path.abspath('../..'))
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from TimeSeriesAnalysis.params import impute_methodology, impute_func, registration_method
import TimeSeriesAnalysis.consts as consts
from TimeSeriesAnalysis.utils.diff_tracker_utils import *
from TimeSeriesAnalysis.utils.data_load_save import *
from TimeSeriesAnalysis.utils.plots_functions_utils import *
from TimeSeriesAnalysis.utils import data_load_save as load_save_utils
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tsfresh import select_features
import datetime


def clean_redundant_columns(df):
    remove_cols = []
    cols_to_remove = ["target"]
    for col_to_remove in cols_to_remove:
        for col_name in df.columns:
            if col_name.startswith(col_to_remove):
                remove_cols.append(col_name)
    df = df.drop(columns=remove_cols)
    return df


def load_tsfresh_csv(path, modality, vid_num, registration="no_reg_", local_density=False, impute_func="impute",
                     impute_methodology="ImputeAllData"):
    path_prefix = path + f"/data/mastodon/ts_transformed/{modality}/{impute_methodology}_{impute_func}/"
    end = f"_reg={registration}, local_den={local_density}, win size {params.window_size}"

    print("reading csv - load_tsfresh_csv function")
    # df = pd.read_csv(path_prefix + f"S{vid_num}" + end, encoding="cp1252", index_col=[0])
    df = pickle.load(open(path_prefix + f"S{vid_num}" + end + ".pkl", 'rb'))
    # print(df.info(memory_usage='deep'), flush=True)
    df = downcast_df(df, fillna=False)  # todo check
    df = clean_redundant_columns(df)

    # print(df.info(memory_usage='deep'), flush=True)
    return df


def get_to_run(path, modality, con_train_num=None, con_test_num=None, diff_train_num=None,
               diff_test_num=None, local_density=False):
    diff_df_train, con_df_train, con_df_test, diff_df_test = None, None, None, None

    if diff_train_num:
        diff_df_train = load_tsfresh_csv(path, modality, diff_train_num, registration_method,
                                         local_density, impute_func, impute_methodology)
        print("diff train len", diff_df_train.shape)

    if con_train_num:
        con_df_train = load_tsfresh_csv(path, modality, con_train_num, registration_method,
                                        local_density, impute_func, impute_methodology)
        print("con_df_train len", con_df_train.shape)

    if con_test_num:
        con_df_test = load_tsfresh_csv(path, modality, con_test_num, registration_method,
                                       local_density, impute_func, impute_methodology)

    if diff_test_num:
        diff_df_test = load_tsfresh_csv(path, modality, diff_test_num, registration_method,
                                        local_density, impute_func, impute_methodology)

    return diff_df_train, con_df_train, con_df_test, diff_df_test


def prep_data(diff_df, con_df, diff_t_window, con_t_windows):
    print("\n preparing data")
    print("\nconcatenating control data & ERKi data")
    df = concat_dfs(diff_df, con_df, diff_t_window, con_t_windows)
    df = df.sample(frac=1).reset_index(drop=True)
    print("\nshape after concat_dfs", df.shape)
    # print(df.isna().sum())

    df = df.replace([np.inf], np.nan)
    df = df.dropna(axis=1)
    print("\nshape after dropna", df.shape)

    # df = df.sample(frac=0.95)
    # print("\nshape after sampling", df.shape)

    df.index = df['Spot track ID']
    y = pd.Series(df['target'])
    y.index = df['Spot track ID']
    df = df.drop(["target", "Spot frame", "Spot track ID"], axis=1)
    # print(df.info(memory_usage='deep'))
    return df, y


def evaluate_clf(dir_path, clf, X_test, y_test, y_train, diff_window, con_window):
    report, auc_score = evaluate(clf, X_test, y_test)

    # load the model & train set & test set
    # clf, X_train, X_test, y_train, y_test = utils.load_data(dir_path)

    # plot ROC curve
    plot_roc(clf=clf, X_test=X_test, y_test=y_test, path=dir_path)

    # perform PCA analysis
    # principal_df, pca = build_pca(3, X_test)
    # plot_pca(principal_df, pca, dir_path)

    # calculate feature importance
    # utils.feature_importance(clf, X_train.columns, dir_path)

    # save classification report & AUC score
    txt_file = open(dir_path + '/info.txt', 'a')
    txt_file.write(f"classification report: {report}"
                   f"\n auc score: {auc_score}"
                   f"\n train samples:{Counter(y_train)}"
                   f"\n {diff_window} ERK, {con_window} con frames"
                   f"\n n features= {X_test.shape}"
                   )

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


def get_confidence_interval(con_df_test, diff_df_test, dir_path, X_train, y_train, X_test, y_test, cols, con_test_n,
                            diff_test_n, diff_window, con_window, n_runs=10):
    print(f"training- {n_runs} runs")

    auc_lst = []
    for i in range(n_runs):
        clf = train_model(X_train, y_train)
        auc = evaluate_clf(dir_path, clf, X_test, y_test, y_train, diff_window, con_window)
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


def get_to_run_both_modalities(path, local_den, con_train_n=None, diff_train_n=None, con_test_n=None, diff_test_n=None):
    diff_df_train_mot, con_df_train_mot, con_df_test_mot, diff_df_test_mot = get_to_run(path=path, modality="motility",
                                                                                        local_density=local_den,
                                                                                        con_train_num=con_train_n,
                                                                                        diff_train_num=diff_train_n,
                                                                                        con_test_num=con_test_n,
                                                                                        diff_test_num=diff_test_n)
    diff_df_train_int, con_df_train_int, con_df_test_int, diff_df_test_int = get_to_run(path=path,
                                                                                        modality="actin_intensity",
                                                                                        local_density=local_den,
                                                                                        con_train_num=con_train_n,
                                                                                        diff_train_num=diff_train_n,
                                                                                        con_test_num=con_test_n,
                                                                                        diff_test_num=diff_test_n)

    if (con_train_n is not None) and (diff_train_n is not None):
        diff_df = diff_df_train_mot.merge(diff_df_train_int, on=["Spot track ID", "Spot frame"], how="left")
        con_df = con_df_train_mot.merge(con_df_train_int, on=["Spot track ID", "Spot frame"], how="left")

    elif (con_test_n is not None) and (diff_test_n is not None):
        diff_df = diff_df_test_mot.merge(diff_df_test_int, on=["Spot track ID", "Spot frame"], how="left")
        con_df = con_df_test_mot.merge(con_df_test_int, on=["Spot track ID", "Spot frame"], how="left")

    diff_df = diff_df.dropna(axis=0)
    con_df = con_df.dropna(axis=0)
    return diff_df, con_df


def build_model_trans_tracks(path, local_density, window_size, tracks_len, con_window, diff_window, modality):
    print(f"\nrunning: build_models_on_transformed_tracks"
          f"\nmodality={modality}, local density={local_density}, reg={registration_method}, "
          f"impute func= {impute_func},impute_methodology= {impute_methodology}")

    for con_train_n, diff_train_n, con_test_n, diff_test_n in [(1, 5, 2, 3), (2, 3, 1, 5), ]:
        print(f"\n train: con_n-{con_train_n},dif_n-{diff_train_n}; test: con_n-{con_test_n},dif_n-{diff_test_n}")

        if modality == "both":
            diff_df_train, con_df_train = get_to_run_both_modalities(path, local_density, con_train_n=con_train_n,
                                                                     diff_train_n=diff_train_n, con_test_n=None,
                                                                     diff_test_n=None)
        else:
            diff_df_train, con_df_train, _, _ = get_to_run(path=path, modality=modality, local_density=local_density,
                                                           con_train_num=con_train_n, diff_train_num=diff_train_n,
                                                           con_test_num=None, diff_test_num=None)

        today = datetime.datetime.now()
        dir_path = f"{consts.storage_path}{today.strftime('%d-%m-%Y')}-{modality} local dens-{local_density}, s{con_train_n}, s{diff_train_n} train {diff_window} diff window" + (
            f" win size {window_size}" if modality != "motility" else "")
        second_dir = f"track len {tracks_len}, impute_func-{impute_methodology}_{impute_func} reg {registration_method}"
        os.makedirs(dir_path, exist_ok=True)
        os.makedirs(os.path.join(dir_path, second_dir), exist_ok=True)
        dir_path += "/" + second_dir

        X_train, y_train = prep_data(diff_df=diff_df_train, con_df=con_df_train, diff_t_window=diff_window,
                                     con_t_windows=con_window)
        X_train = downcast_df(X_train)

        del diff_df_train
        del con_df_train
        print("\ndeleted diff_df_train, con_df_train")

        X_train = select_features(X_train, y_train, n_jobs=10) #, chunksize=10
        print("\nDone feature selection", flush=True)

        filter_col = [col for col in X_train if col.startswith('mean')]
        print(filter_col)

        clf = train_model(X_train, y_train)
        load_save_utils.save_data(dir_path, X_train=X_train)
        del X_train
        if modality == "both":
            diff_df_test, con_df_test = get_to_run_both_modalities(path, local_density, con_train_n=None,
                                                                   diff_train_n=None, con_test_n=con_test_n,
                                                                   diff_test_n=diff_test_n)

        else:
            _, _, con_df_test, diff_df_test = get_to_run(path=path, modality=modality, local_density=local_density,
                                                         con_train_num=None, diff_train_num=None,
                                                         con_test_num=con_test_n, diff_test_num=diff_test_n, )

        X_test, y_test = prep_data(diff_df=diff_df_test, con_df=con_df_test, diff_t_window=diff_window,
                                   con_t_windows=con_window)

        cols = list(clf.feature_names_in_)
        X_test = X_test[cols]
        cols.extend(["Spot track ID", "Spot frame"])
        print(cols)

        load_save_utils.save_data(dir_path, y_train=y_train, X_test=X_test, y_test=y_test, clf=clf)
        evaluate_clf(dir_path, clf, X_test, y_test, y_train, diff_window, con_window)
        del X_test
        del y_test

        print("calc avg prob")
        df_score_con = calc_prob(con_df_test[cols].dropna(axis=1), clf, n_frames=260)
        df_score_dif = calc_prob(diff_df_test[cols].dropna(axis=1), clf, n_frames=260)

        plot_avg_conf(df_score_con.drop("Spot track ID", axis=1), df_score_dif.drop("Spot track ID", axis=1),
                      mot_int=modality, path=dir_path + f"/avg conf s{con_test_n}, s{diff_test_n}.png")


if __name__ == '__main__':
    modality = sys.argv[1]

    build_model_trans_tracks(consts.storage_path, params.local_density, params.window_size, params.tracks_len,
                             params.con_window, params.diff_window, modality)
