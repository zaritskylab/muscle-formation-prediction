from collections import Counter
import os
import sys

sys.path.append('/sise/home/shakarch/muscle-formation-diff')
sys.path.append(os.path.abspath('..'))

sys.path.append(os.path.abspath('../..'))
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from configuration.params import impute_methodology, impute_func, registration_method
from model_layer.utils import *
from data_layer.utils import *
from analysis.plots_functions_utils import *
from data_layer import utils as load_save_utils
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


def load_tsfresh_csv(path, modality, vid_num, feature_type, specific_feature_type, win_size, registration="no_reg_",
                     local_density=False, impute_func="impute", impute_methodology="ImputeAllData"):
    path_prefix = path + f"data/mastodon/ts_transformed/{modality}/{params.impute_methodology}_{params.impute_func}/"
    end = f"/{feature_type}/{specific_feature_type}/merged_chunks_reg={params.registration_method},local_den={local_density},win size={win_size}"
    print(f"read data from video number {vid_num}")
    df = pickle.load(open(path_prefix + f"S{vid_num}" + end + ".pkl", 'rb'))
    df = downcast_df(df, fillna=False)
    df = clean_redundant_columns(df)
    return df


def get_to_run(path, modality, feature_type, specific_feature_type, win_size, con_train_num=None, con_test_num=None,
               diff_train_num=None,
               diff_test_num=None, local_density=False):
    diff_df_train, con_df_train, con_df_test, diff_df_test = None, None, None, None

    if diff_train_num:
        diff_df_train = load_tsfresh_csv(path, modality, diff_train_num, feature_type, specific_feature_type, win_size,
                                         registration_method, local_density, impute_func, impute_methodology)
        print(f"diff train len: {diff_df_train.shape}", flush=True)

    if con_train_num:
        con_df_train = load_tsfresh_csv(path, modality, con_train_num, feature_type, specific_feature_type, win_size,
                                        registration_method, local_density, impute_func, impute_methodology)
        print(f"con_df_train len: {con_df_train.shape}", flush=True)

    if con_test_num:
        con_df_test = load_tsfresh_csv(path, modality, con_test_num, feature_type, specific_feature_type, win_size,
                                       registration_method, local_density, impute_func, impute_methodology)

    if diff_test_num:
        diff_df_test = load_tsfresh_csv(path, modality, diff_test_num, feature_type, specific_feature_type, win_size,
                                        registration_method, local_density, impute_func, impute_methodology)

    return diff_df_train, con_df_train, con_df_test, diff_df_test


def prep_data(diff_df, con_df, diff_t_window, con_t_windows):
    print("\n preparing data", flush=True)
    print("\nconcatenating control data & ERKi data")
    df = concat_dfs(diff_df, con_df, diff_t_window, con_t_windows)
    del diff_df
    del con_df
    df = df.sample(frac=1).reset_index(drop=True)
    print("\nshape after concat_dfs", df.shape)

    df = df.replace([np.inf], np.nan)
    df = df.dropna(axis=1)
    print("\nshape after dropna", df.shape)

    df.index = df['Spot track ID']
    y = pd.Series(df['target'])
    y.index = df['Spot track ID']
    df = df.drop(["target", "Spot frame", "Spot track ID"], axis=1)
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
    for col_name in df.columns:
        if col_name.startswith("target"):
            remove_cols.append(col_name)
    df = df.drop(columns=remove_cols)
    return df


def get_confidence_interval(con_df_test, diff_df_test, dir_path, X_train, y_train, X_test, y_test, cols, con_test_n,
                            diff_test_n, diff_window, con_window, modality, n_runs=10):
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


def plot_avg_conf(df_con, df_diff, type_modality, path=""):
    def plot(df, color1, color2):
        avg_vals_diff = ([df[col].mean() for col in df.columns])
        std_vals_diff = ([df[col].std() for col in df.columns])
        p_std = np.asarray(avg_vals_diff) + np.asarray(std_vals_diff)
        m_std = np.asarray(avg_vals_diff) - np.asarray(std_vals_diff)

        plt.plot([(i + int(df.columns[0])) * 5 / 60 for i in range(len(avg_vals_diff))], avg_vals_diff,
                 color=color1, label="avg")

        plt.fill_between([(i + int(df.columns[0])) * 5 / 60 for i in range(len(avg_vals_diff))], m_std, p_std,
                         alpha=0.4,
                         color=color2, label="std")

    plot(df_diff, "DarkOrange", "Orange")
    plot(df_con, "blue", "blue")
    plt.legend(["Erk avg", "Control avg", "Erk std", "Control std"])
    plt.xlabel("time (h)")
    plt.ylabel("avg score")
    plt.title(f"avg differentiation score over time ({type_modality})")
    plt.plot([i * 5 / 60 for i in range(260)], [0.5 for i in range(260)], color="black", linestyle="--")
    if path:
        plt.savefig(path + f"avg differentiation score over time ({type_modality}).eps", format="eps", dpi=300)
    plt.show()
    plt.clf()


def get_to_run_both_modalities(path, local_den, modality1, modality2, con_train_n=None, diff_train_n=None,
                               con_test_n=None, diff_test_n=None, feature_type="", specific_feature_type="",
                               window_size=params.window_size):
    diff_df_train_mot, con_df_train_mot, con_df_test_mot, diff_df_test_mot = get_to_run(path=path, modality=modality1,
                                                                                        local_density=local_den,
                                                                                        con_train_num=con_train_n,
                                                                                        diff_train_num=diff_train_n,
                                                                                        con_test_num=con_test_n,
                                                                                        diff_test_num=diff_test_n,
                                                                                        feature_type=feature_type,
                                                                                        specific_feature_type=specific_feature_type,
                                                                                        win_size=window_size)
    diff_df_train_int, con_df_train_int, con_df_test_int, diff_df_test_int = get_to_run(path=path,
                                                                                        modality=modality2,
                                                                                        local_density=local_den,
                                                                                        con_train_num=con_train_n,
                                                                                        diff_train_num=diff_train_n,
                                                                                        con_test_num=con_test_n,
                                                                                        diff_test_num=diff_test_n,
                                                                                        feature_type=feature_type,
                                                                                        specific_feature_type=specific_feature_type,
                                                                                        win_size=window_size)
    if (con_train_n is not None) and (diff_train_n is not None):
        diff_df = diff_df_train_mot.merge(diff_df_train_int, on=["Spot track ID", "Spot frame"], how="left")
        con_df = con_df_train_mot.merge(con_df_train_int, on=["Spot track ID", "Spot frame"], how="left")

    elif (con_test_n is not None) and (diff_test_n is not None):
        diff_df = diff_df_test_mot.merge(diff_df_test_int, on=["Spot track ID", "Spot frame"], how="left")
        con_df = con_df_test_mot.merge(con_df_test_int, on=["Spot track ID", "Spot frame"], how="left")

    diff_df = diff_df.dropna(axis=0)
    con_df = con_df.dropna(axis=0)

    return diff_df, con_df


def get_data(modality, path, local_density, con_train_n, diff_train_n, con_test_n, diff_test_n, feature_type,
             specific_feature_type, window_size):
    if len(modality.split("-")) == 2:
        first_modality, second_modality = modality.split("-")[0], modality.split("-")[1]
        diff_df, con_df = get_to_run_both_modalities(path, local_density, first_modality,
                                                     second_modality,
                                                     con_train_n=con_train_n,
                                                     diff_train_n=diff_train_n,
                                                     con_test_n=con_test_n,
                                                     diff_test_n=diff_test_n,
                                                     feature_type=feature_type,
                                                     specific_feature_type=specific_feature_type,
                                                     window_size=window_size)
    else:
        diff_df_train, con_df_train, con_df_test, diff_df_test = get_to_run(path=path, modality=modality,
                                                                            local_density=local_density,
                                                                            feature_type=feature_type,
                                                                            specific_feature_type=specific_feature_type,
                                                                            win_size=window_size,
                                                                            con_train_num=con_train_n,
                                                                            diff_train_num=diff_train_n,
                                                                            con_test_num=con_test_n,
                                                                            diff_test_num=diff_test_n)
        if (con_train_n is not None) and (diff_train_n is not None):
            diff_df = diff_df_train
            con_df = con_df_train

        elif (con_test_n is not None) and (diff_test_n is not None):
            diff_df = diff_df_test
            con_df = con_df_test
    print("ended loading", flush=True)
    return diff_df, con_df


def build_model_trans_tracks(path, local_density, window_size, tracks_len, con_window, diff_window, modality,
                             feature_type="", specific_feature_type=""):
    print(f"\nrunning: build_models_on_transformed_tracks; modality={modality}")

    for con_train_n, diff_train_n, con_test_n, diff_test_n in [(1, 5, 2, 3), (2, 3, 1, 5), ]:
        print(f"\n train: con_n-{con_train_n},dif_n-{diff_train_n}; test: con_n-{con_test_n},dif_n-{diff_test_n}")

        path_by_feature = f"{consts.storage_path}{feature_type}/{specific_feature_type}"
        today = datetime.datetime.now()
        dir_path = f"{path_by_feature}/{today.strftime('%d-%m-%Y')}-{modality} local dens-{local_density}, s{con_train_n}, s{diff_train_n} train" + (
            f" win size {window_size}" if modality != "motility" else "")
        second_dir = f"track len {tracks_len}, impute_func-{impute_methodology}_{impute_func} reg {registration_method}"
        os.makedirs(dir_path, exist_ok=True)
        os.makedirs(os.path.join(dir_path, second_dir), exist_ok=True)
        dir_path += "/" + second_dir + "/"

        diff_df_train, con_df_train = get_data(modality, path, local_density, con_train_n, diff_train_n,
                                               None, None, feature_type, specific_feature_type, window_size)
        print("loaded data, now start prep data", flush=True)
        X_train, y_train = prep_data(diff_df=diff_df_train, con_df=con_df_train, diff_t_window=diff_window,
                                     con_t_windows=con_window)

        X_train = downcast_df(X_train)

        del diff_df_train
        del con_df_train
        print("\ndeleted diff_df_train, con_df_train", flush=True)

        X_train = select_features(X_train, y_train, n_jobs=10)  # , chunksize=10
        print("\nDone feature selection", flush=True)

        filter_col = [col for col in X_train if col.startswith('mean')]
        print(filter_col)

        clf = train_model(X_train, y_train, modality)
        load_save_utils.save_data(dir_path, X_train=X_train)
        # del X_train

        diff_df_test, con_df_test = get_data(modality, path, local_density, None, None, con_test_n, diff_test_n,
                                             feature_type, specific_feature_type, window_size)

        print("X_test, y_test")
        X_test, y_test = prep_data(diff_df=diff_df_test, con_df=con_df_test, diff_t_window=diff_window,
                                   con_t_windows=con_window)

        cols = list(clf.feature_names_in_)
        X_test = X_test[cols]
        cols.extend(["Spot track ID", "Spot frame"])
        print(cols)

        evaluate_clf(dir_path, clf, X_test, y_test, y_train, diff_window, con_window)

        train_model_compare_algorithms(X_train, y_train, X_test, y_test, dir_path)

        load_save_utils.save_data(dir_path, y_train=y_train, X_test=X_test, y_test=y_test, clf=clf)
        del X_test
        del y_test

        print("calc avg prob")
        df_score_con = calc_prob(con_df_test[cols].dropna(axis=1), clf, n_frames=260)
        df_score_dif = calc_prob(diff_df_test[cols].dropna(axis=1), clf, n_frames=260)

        pickle.dump(df_score_con, open(dir_path + f"df_score_vid_num_S{con_test_n}.pkl", 'wb'))
        pickle.dump(df_score_dif, open(dir_path + f"df_score_vid_num_S{diff_test_n}.pkl", 'wb'))

        plot_avg_conf(df_score_con.drop("Spot track ID", axis=1),
                      df_score_dif.drop("Spot track ID", axis=1), modality, dir_path)


if __name__ == '__main__':
    modality = sys.argv[1]

    build_model_trans_tracks(path=consts.storage_path, local_density=params.local_density,
                             window_size=params.window_size,
                             tracks_len=params.tracks_len, con_window=params.con_window,
                             diff_window=params.diff_window, modality=modality)
