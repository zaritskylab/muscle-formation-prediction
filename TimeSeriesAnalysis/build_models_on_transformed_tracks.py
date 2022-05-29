import pickle
from collections import Counter
import diff_tracker_utils as utils
import pandas as pd

from diff_tracker import DiffTracker
from imblearn.over_sampling import RandomOverSampler
import consts
from tsfresh import select_features


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


def get_to_run(path, to_run, con_train_num, con_test_num, diff_train_num, diff_test_num, local_density=False):
    str = " win size 15" if to_run == "intensity" else ""
    diff_df_train = pickle.load(
        open(
            path + f"/data/mastodon/ts_transformed_new/{to_run}/" + f"05_01_SS{diff_train_num}_transformed_local_den_{local_density}{str}",
            # win size 15
            'rb'))

    con_df_train = pickle.load(
        open(
            path + f"/data/mastodon/ts_transformed_new/{to_run}/" + f"05_01_SS{con_train_num}_transformed_local_den_{local_density}{str}",
            # win size 15

            'rb'))
    con_df_test = pickle.load(
        open(
            path + f"/data/mastodon/ts_transformed_new/{to_run}/" + f"05_01_SS{con_test_num}_transformed_local_den_{local_density}{str}",
            # win size 15
            'rb'))
    diff_df_test = pickle.load(
        open(
            path + f"/data/mastodon/ts_transformed_new/{to_run}/" + f"05_01_SS{diff_test_num}_transformed_local_den_{local_density}{str}",
            # win size 15
            'rb'))

    return diff_df_train, con_df_train, con_df_test, diff_df_test


def prep_data(diff_df, con_df, diff_t_window, con_t_windows):
    print("concatenating control data & ERKi data")
    df = utils.concat_dfs(diff_df, con_df, diff_t_window,
                          con_t_windows)
    df = df.sample(frac=1).reset_index(drop=True)

    df = df.dropna()
    # df.reset_index()
    df.index = df['Spot track ID']
    y = pd.Series(df['target'])
    y.index = df['Spot track ID']
    df = df.drop(["target", "Spot frame", "Spot track ID"], axis=1)
    return df, y


def evaluate_clf(dir_path, clf, X_test, y_test):
    report, auc_score = utils.evaluate(clf, X_test, y_test)

    # load the model & train set & test set
    clf, X_train, X_test, y_train, y_test = utils.load_data(dir_path)

    # plot ROC curve
    utils.plot_roc(clf=clf, X_test=X_test, y_test=y_test, path=dir_path)

    # perform PCA analysis
    principal_df, pca = utils.build_pca(3, X_test)
    utils.plot_pca(principal_df, pca, dir_path)

    # calculate feature importance
    utils.feature_importance(clf, X_train.columns, dir_path)

    # save classification report & AUC score
    txt_file = open(dir_path + '/info.txt', 'a')
    txt_file.write(f"classification report: {report}\n auc score: {auc_score}\n train samples:{Counter(y_train)}")

    txt_file.close()


def remove_target_cols(df):
    remove_cols = []
    for col_name in df.columns:
        if col_name.startswith("target"):
            remove_cols.append(col_name)
    df = df.drop(columns=remove_cols)
    return df


if __name__ == '__main__':
    diff_window = [140, 170]
    tracks_len = 30
    con_windows = [[0, 30], [40, 70], [90, 120], [140, 170], [180, 210], [220, 250]]

    path = consts.cluster_path
    to_run = "nuc_intensity"
    motility = False
    intensity = False
    nuc_intensity = True
    local_density = False

    for con_train_n, diff_train_n, con_test_n, diff_test_n in [(1, 5, 2, 3), (2, 3, 1, 5)]:
        diff_df_train, con_df_train, con_df_test, diff_df_test = get_to_run(path=path, to_run=to_run,
                                                                            con_train_num=con_train_n,
                                                                            con_test_num=con_test_n,
                                                                            diff_train_num=diff_train_n,
                                                                            diff_test_num=diff_test_n,
                                                                            local_density=local_density)

        # diff_df_train2, con_df_train2, con_df_test2, diff_df_test2 = get_to_run(path=path, to_run="motility",
        #                                                                         con_train_num=con_train_n,
        #                                                                         con_test_num=con_test_n,
        #                                                                         diff_train_num=diff_train_n,
        #                                                                         diff_test_num=diff_test_n,
        #                                                                         local_density=local_density)
        #
        # diff_df_train = pd.merge(diff_df_train, diff_df_train2, on=['Spot track ID', 'Spot frame', 'target'])
        # con_df_train = pd.merge(con_df_train, con_df_train2, on=['Spot track ID', 'Spot frame', 'target'])
        # con_df_test = pd.merge(con_df_test, con_df_test2, on=['Spot track ID', 'Spot frame', 'target'])
        # diff_df_test = pd.merge(diff_df_test, diff_df_test2, on=['Spot track ID', 'Spot frame', 'target'])
        # del (diff_df_train2, con_df_train2, con_df_test2, diff_df_test2)

        dir_path = f"30-03-2022-manual_mastodon_{to_run} local density-{local_density}, s{con_train_n}, s{diff_train_n} are train"
        second_dir = f"{diff_window} frames ERK, {con_windows} frames con track len {tracks_len}"
        utils.open_dirs(dir_path, second_dir)
        dir_path += "/" + second_dir

        print("preparing data")
        X_train, y_train = prep_data(diff_df=diff_df_train, con_df=con_df_train, diff_t_window=diff_window,
                                     con_t_windows=con_windows)
        X_train = select_features(X_train, y_train)

        X_test, y_test = prep_data(diff_df=diff_df_test, con_df=con_df_test, diff_t_window=diff_window,
                                   con_t_windows=con_windows)
        X_test = X_test[X_train.columns]

        print("training")
        clf = utils.train(X_train, y_train)
        utils.save_data(dir_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, clf=clf)
        evaluate_clf(dir_path, clf, X_test, y_test)
