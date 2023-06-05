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
from analysis.utils import *
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


def load_tsfresh_csv(transfromed_pkl_path, modality, vid_num):
    print(f"read data from video number {vid_num}")
    df = pickle.load(open(transfromed_pkl_path % (modality, vid_num), 'rb'))
    df = downcast_df(df, fillna=False)
    df = clean_redundant_columns(df)
    return df


def get_to_run(transformed_data_path, modality, con_train_num=None, con_test_num=None, diff_train_num=None, diff_test_num=None):
    diff_df_train, con_df_train, con_df_test, diff_df_test = None, None, None, None

    if diff_train_num:
        diff_df_train = load_tsfresh_csv(transformed_data_path, modality, diff_train_num)
        print(f"diff train len: {diff_df_train.shape}", flush=True)

    if con_train_num:
        con_df_train = load_tsfresh_csv(transformed_data_path, modality, con_train_num)
        print(f"con_df_train len: {con_df_train.shape}", flush=True)

    if con_test_num:
        con_df_test = load_tsfresh_csv(transformed_data_path, modality, con_test_num)

    if diff_test_num:
        diff_df_test = load_tsfresh_csv(transformed_data_path, modality, diff_test_num)

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

    # plot ROC curve
    plot_roc(clf=clf, X_test=X_test, y_test=y_test, path=dir_path)

    # save classification report & AUC score
    txt_file = open(dir_path + '/info.txt', 'a')
    txt_file.write(f"classification report: {report}"
                   f"\n auc score: {auc_score}"
                   f"\n train samples:{Counter(y_train)}"
                   f"\n {diff_window} ERK, {con_window} con frames"
                   f"\n n features= {X_test.shape}")

    txt_file.close()

    return auc_score


def plot_avg_score(df_con, df_diff, type_modality, path=""):
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


def get_to_run_both_modalities(path, modality1, modality2, con_train_n=None, diff_train_n=None,
                               con_test_n=None, diff_test_n=None):
    diff_df_train_mot, con_df_train_mot, con_df_test_mot, diff_df_test_mot = get_to_run(transformed_data_path=path, modality=modality1,
                                                                                        con_train_num=con_train_n,
                                                                                        diff_train_num=diff_train_n,
                                                                                        con_test_num=con_test_n,
                                                                                        diff_test_num=diff_test_n)
    diff_df_train_int, con_df_train_int, con_df_test_int, diff_df_test_int = get_to_run(transformed_data_path=path,
                                                                                        modality=modality2,
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


def get_data(modality, path, local_density, con_train_n, diff_train_n, con_test_n, diff_test_n):
    if len(modality.split("-")) == 2:
        first_modality, second_modality = modality.split("-")[0], modality.split("-")[1]
        diff_df, con_df = get_to_run_both_modalities(path, local_density, first_modality,
                                                     second_modality,
                                                     diff_train_n=diff_train_n,
                                                     con_test_n=con_test_n,
                                                     diff_test_n=diff_test_n)
    else:
        diff_df_train, con_df_train, con_df_test, diff_df_test = get_to_run(transformed_data_path=path, modality=modality,
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


def build_state_prediction_model(save_data_dir_path, diff_df_train, con_df_train, diff_df_test, con_df_test,
                                 diff_window, con_window):
    X_train, y_train = prep_data(diff_df=diff_df_train, con_df=con_df_train,
                                 diff_t_window=diff_window, con_t_windows=con_window)
    X_train = downcast_df(X_train)

    del diff_df_train
    del con_df_train

    print("Start feature selection", flush=True)
    X_train = select_features(X_train, y_train, n_jobs=10)
    clf = train_model(X_train, y_train, modality)

    X_test, y_test = prep_data(diff_df=diff_df_test, con_df=con_df_test,
                               diff_t_window=diff_window, con_t_windows=con_window)
    X_test = X_test[clf.feature_names_in_]

    evaluate_clf(save_data_dir_path, clf, X_test, y_test, y_train, diff_window, con_window)
    load_save_utils.save_data(save_data_dir_path, X_train=X_train, y_train=y_train, X_test=X_test,
                              y_test=y_test, clf=clf)

    return clf


def build_state_prediction_model_light(save_dir_path, transformed_data_path, local_density, con_window, diff_window, modality):
    diff_df_train, con_df_train = get_data(modality, transformed_data_path, local_density, con_train_n, diff_train_n,
                                           None, None)
    print("loaded data, now start prep data", flush=True)
    X_train, y_train = prep_data(diff_df=diff_df_train, con_df=con_df_train, diff_t_window=diff_window,
                                 con_t_windows=con_window)
    X_train = downcast_df(X_train)

    del diff_df_train
    del con_df_train
    print("deleted diff_df_train, con_df_train", flush=True)

    X_train = select_features(X_train, y_train, n_jobs=10)  # , chunksize=10
    print("Done feature selection", flush=True)

    clf = train_model(X_train, y_train, modality)
    load_save_utils.save_data(save_dir_path, X_train=X_train)

    diff_df_test, con_df_test = get_data(modality, transformed_data_path, local_density, None, None, con_test_n, diff_test_n)

    X_test, y_test = prep_data(diff_df=diff_df_test, con_df=con_df_test, diff_t_window=diff_window,
                               con_t_windows=con_window)
    X_test = X_test[list(clf.feature_names_in_)]

    evaluate_clf(save_dir_path, clf, X_test, y_test, y_train, diff_window, con_window)
    train_model_compare_algorithms(X_train, y_train, X_test, y_test, save_dir_path)
    load_save_utils.save_data(save_dir_path, y_train=y_train, X_test=X_test, y_test=y_test, clf=clf)

    return clf


if __name__ == '__main__':
    modality = sys.argv[1]
    for con_train_n, diff_train_n, con_test_n, diff_test_n in [(1, 5, 2, 3), (2, 3, 1, 5), ]:
        print(f"\n train: con_n-{con_train_n},dif_n-{diff_train_n}; test: con_n-{con_test_n},dif_n-{diff_test_n}")

        today = datetime.datetime.now().strftime('%d-%m-%Y')
        dir_path = f"{consts.storage_path}/{today}-{modality} local dens-False, s{con_train_n}, s{diff_train_n} train" \
                   + (f" win size {params.window_size}" if modality != "motility" else "")

        second_dir = f"track len {params.tracks_len}, impute_func-{impute_methodology}_{impute_func} reg {registration_method}"
        os.makedirs(dir_path, exist_ok=True)
        os.makedirs(os.path.join(dir_path, second_dir), exist_ok=True)
        save_dir_path = dir_path + "/" + second_dir + "/"

        clf = build_state_prediction_model_light(save_dir_path=save_dir_path, transformed_data_path=consts.transformed_data_path,
                                                 local_density=params.local_density,
                                                 con_window=params.con_window,
                                                 diff_window=params.diff_window, modality=modality)

        cols = list(clf.feature_names_in_)
        cols.extend(["Spot track ID", "Spot frame"])

        print("calc avg prob")
        diff_df_test, con_df_test = get_data(modality, consts.transformed_data_path, False, None, None,
                                             con_test_n, diff_test_n)

        df_score_con = calc_state_trajectory(con_df_test[cols].dropna(axis=1), clf, n_frames=260)
        df_score_dif = calc_state_trajectory(diff_df_test[cols].dropna(axis=1), clf, n_frames=260)

        pickle.dump(df_score_con, open(save_dir_path + f"df_score_vid_num_S{con_test_n}.pkl", 'wb'))
        pickle.dump(df_score_dif, open(save_dir_path + f"df_score_vid_num_S{diff_test_n}.pkl", 'wb'))

        plot_avg_score(df_score_con.drop("Spot track ID", axis=1),
                       df_score_dif.drop("Spot track ID", axis=1), modality, save_dir_path)

