import pickle
from collections import Counter
import diff_tracker_utils as utils
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from diff_tracker import DiffTracker
from imblearn.over_sampling import RandomOverSampler
import consts
import numpy as np
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute


def get_intensity_df(path, tagged_csv, csv_all, vid_path_actin, winsize, local_density):
    df_all_tracks, tracks_s_train = utils.get_tracks(path + csv_all, target=1)

    df_tagged = pd.read_csv(path + tagged_csv, encoding="ISO-8859-1")
    df_tagged = df_tagged[df_tagged["manual"] == 1]
    df_tagged = df_all_tracks[df_all_tracks["manual"] == 1]

    df_tagged = utils.add_features_df(df_tagged, df_all_tracks, local_density=local_density)
    df_tagged = utils.get_intensity_measures_df(df=df_tagged,
                                                video_actin_path=path + vid_path_actin,
                                                window_size=winsize, local_density=local_density)
    return df_tagged


def clean_redundant_columns(df, save_path):
    remove_cols = []
    cols_to_remove = ["target", "Unnamed", "sum_nuc", "std_nuc", "min_nuc", "max_nuc", "x", "y"]
    for col_to_remove in cols_to_remove:
        for col_name in df.columns:
            if col_name.startswith(col_to_remove):
                remove_cols.append(col_name)
    df = df.drop(columns=remove_cols)
    # df.to_csv(save_path)
    return df


def get_to_run(path, to_run, con_train_num=None, con_test_num=None, diff_train_num=None, diff_test_num=None,
               registration=False,
               local_density=False):
    diff_df_train, con_df_train, con_df_test, diff_df_test = None, None, None, None
    path_prefix = path + f"/data/mastodon/ts_transformed_new/{to_run}/"
    end = f"_imputed reg={registration}, local_den={local_density}, win size 16"
    if diff_train_num:
        diff_df_train = pd.read_csv(path_prefix + f"S{diff_train_num}" + end, encoding="cp1252", nrows=8000)
        diff_df_train = clean_redundant_columns(diff_df_train, path_prefix + f"S{diff_train_num}" + end)
        print("diff train len", len(diff_df_train))
        # todo: remove
        # diff_df_train = diff_df_train.sample(frac=0.3)

    if con_train_num:
        con_df_train = pd.read_csv(path_prefix + f"S{con_train_num}" + end, encoding="cp1252", nrows=5000)
        con_df_train = clean_redundant_columns(con_df_train, path_prefix + f"S{con_train_num}" + end)
        print("con_df_train len", len(con_df_train))
        # con_df_train = con_df_train.sample(frac=0.3)


    if con_test_num:
        con_df_test = pd.read_csv(path_prefix + f"S{con_test_num}" + end, encoding="cp1252", nrows=5000)
        con_df_test = clean_redundant_columns(con_df_test, path_prefix + f"S{con_test_num}" + end)
        # con_df_test = con_df_test.sample(frac=0.3)

    if diff_test_num:
        diff_df_test = pd.read_csv(path_prefix + f"S{diff_test_num}" + end, encoding="cp1252", nrows=8000) #, nrows=8000
        diff_df_test = clean_redundant_columns(diff_df_test, path_prefix + f"S{diff_test_num}" + end)
        # diff_df_test = diff_df_test.sample(frac=0.3)

    # str = " win size 15" if to_run == "intensity" else ""
    # str += ", reg True" if registration else ""
    # diff_df_train = pickle.load(
    #     open(
    #         path + f"/data/mastodon/ts_transformed_new/{to_run}/" + f"05_01_SS{diff_train_num}_transformed_local_den_{local_density}{str}",
    #         'rb'))
    # con_df_train = pickle.load(
    #     open(
    #         path + f"/data/mastodon/ts_transformed_new/{to_run}/" + f"05_01_SS{con_train_num}_transformed_local_den_{local_density}{str}",
    #         'rb'))
    # con_df_test = pickle.load(
    #     open(
    #         path + f"/data/mastodon/ts_transformed_new/{to_run}/" + f"05_01_SS{con_test_num}_transformed_local_den_{local_density}{str}",
    #         'rb'))
    # diff_df_test = pickle.load(
    #     open(
    #         path + f"/data/mastodon/ts_transformed_new/{to_run}/" + f"05_01_SS{diff_test_num}_transformed_local_den_{local_density}{str}",
    #         'rb'))

    return diff_df_train, con_df_train, con_df_test, diff_df_test


def prep_data(diff_df, con_df, diff_t_window, con_t_windows):
    print("concatenating control data & ERKi data")
    df = utils.concat_dfs(diff_df, con_df, diff_t_window,
                          con_t_windows)
    df = df.sample(frac=1).reset_index(drop=True)
    print("shape after concat_dfs", df.shape)
    print(df.isna().sum())
    # todo: remove
    # df = df.sample(frac=0.8)

    # df = df.dropna()
    # print("shape after dropna", df.shape)

    df.index = df['Spot track ID']
    y = pd.Series(df['target'])
    y.index = df['Spot track ID']
    df = df.drop(["target", "Spot frame", "Spot track ID"], axis=1)
    return df, y


def evaluate_clf(dir_path, clf, X_test, y_test, y_train):
    report, auc_score = utils.evaluate(clf, X_test, y_test)

    # load the model & train set & test set
    # clf, X_train, X_test, y_train, y_test = utils.load_data(dir_path)

    # plot ROC curve
    utils.plot_roc(clf=clf, X_test=X_test, y_test=y_test, path=dir_path)

    # perform PCA analysis
    principal_df, pca = utils.build_pca(3, X_test)
    utils.plot_pca(principal_df, pca, dir_path)

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
        clf = utils.train(X_train, y_train)
        auc = evaluate_clf(dir_path, clf, X_test, y_test)
        auc_lst.append(auc)

        print("calc avg prob")
        df_score_con = utils.calc_prob(con_df_test[cols], clf, n_frames=260)
        df_score_dif = utils.calc_prob(diff_df_test[cols], clf, n_frames=260)

        pickle.dump(df_score_con, open(dir_path + f"/df_prob_w=30, video_num={con_test_n} run_{i}", 'wb'))
        pickle.dump(df_score_dif, open(dir_path + f"/df_prob_w=30, video_num={diff_test_n} run_{i}", 'wb'))

    print(auc_lst)
    try:
        np.save(dir_path + f'/auc_lst{to_run}.npy', auc_lst, allow_pickle=True)
    except:
        pass

    sns.distplot(auc_lst)
    plt.xlabel("auc")
    plt.savefig(dir_path + f"/aucs_{to_run}, {n_runs}")
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
    diff_window = [140, 170]
    tracks_len = 30
    con_windows = [[0, 30], [40, 70], [90, 120], [140, 170], [180, 210], [220, 250]]

    path = consts.cluster_path
    to_run = "nuc_intensity"
    local_density = False
    registration = False

    for con_train_n, diff_train_n, con_test_n, diff_test_n in [(1, 5, 1, 3), (2, 3, 1, 5), ]:
        diff_df_train, con_df_train, _, _ = get_to_run(path=path, to_run=to_run,
                                                       con_train_num=con_train_n,
                                                       con_test_num=None,
                                                       diff_train_num=diff_train_n,
                                                       diff_test_num=None,
                                                       local_density=local_density,
                                                       registration=registration)

        dir_path = f"19-06-2022-manual_mastodon_{to_run} local density-{local_density}, s{con_train_n}, s{diff_train_n} are train, reg {registration}"
        second_dir = f"{diff_window} frames ERK, {con_windows} frames con track len {tracks_len}"
        utils.open_dirs(dir_path, second_dir)
        dir_path += "/" + second_dir

        print("preparing data")
        X_train, y_train = prep_data(diff_df=diff_df_train, con_df=con_df_train, diff_t_window=diff_window,
                                     con_t_windows=con_windows)
        del diff_df_train
        del con_df_train

        print("deleted diff_df_train, con_df_train")
        # impute(X_train)
        X_train = select_features(X_train, y_train)

        clf = utils.train(X_train, y_train)
        cols = list(X_train.columns)
        utils.save_data(dir_path, X_train=X_train)
        del X_train
        _, _, con_df_test, diff_df_test = get_to_run(path=path, to_run=to_run,
                                                     con_train_num=None,
                                                     con_test_num=con_test_n,
                                                     diff_train_num=None,
                                                     diff_test_num=diff_test_n,
                                                     local_density=local_density,
                                                     registration=registration)

        X_test, y_test = prep_data(diff_df=diff_df_test, con_df=con_df_test, diff_t_window=diff_window,
                                   con_t_windows=con_windows)


        # impute(X_test)
        X_test = X_test[cols]



        cols.extend(["Spot track ID", "Spot frame"])


        utils.save_data(dir_path, y_train=y_train, X_test=X_test, y_test=y_test, clf=clf)
        auc = evaluate_clf(dir_path, clf, X_test, y_test, y_train)
        del X_test
        del y_test

        print("calc avg prob")
        df_score_con = utils.calc_prob(con_df_test[cols], clf, n_frames=260)
        df_score_dif = utils.calc_prob(diff_df_test[cols], clf, n_frames=260)

        plot_avg_conf(df_score_con.drop("Spot track ID", axis=1), df_score_dif.drop("Spot track ID", axis=1),
                      mot_int=to_run,
                      path=dir_path + f"/avg conf s{con_test_n}, s{diff_test_n}.png")
