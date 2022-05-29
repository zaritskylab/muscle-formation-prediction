import pickle
import consts
import diff_tracker_utils as utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import pandas as pd
from build_models_on_transformed_tracks import get_to_run


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
    plt.legend(["Erk avg", "Erk std", "Control avg", "Control std"])
    plt.xlabel("time (h)")
    plt.ylabel("avg confidence")
    plt.title(f"avg differentiation confidence over time ({mot_int})")
    plt.plot([i * 5 / 60 for i in range(260)], [0.5 for i in range(260)], color="black", linestyle="--")
    plt.savefig(path)
    plt.show()
    plt.clf()


def auc_over_time(df_con, df_diff, clf):
    def get_t_pred(df):
        df.dropna(inplace=True)
        t_pred = []
        for track_id, track in df.groupby("Spot track ID"):
            track_to_predict = track[track["Spot frame"] == t].drop(["Spot track ID", "Spot frame"], axis=1)
            if len(track_to_predict) > 0:
                pred = clf.predict(track_to_predict)
                t_pred.append(pred[0])
        return t_pred

    time_points = list(df_con.sort_values("Spot frame")["Spot frame"].unique())
    aucs = {}
    for t in time_points:
        if t > 126:
            x = 8
        t_pred_con = get_t_pred(df_con)
        t_pred_diff = get_t_pred(df_diff)
        true_pred_con = [0 for i in range(len(t_pred_con))]
        true_pred_diff = [1 for i in range(len(t_pred_diff))]

        # print(t)
        fpr, tpr, thresholds = metrics.roc_curve(true_pred_con + true_pred_diff, t_pred_con + t_pred_diff,
                                                 pos_label=1)
        aucs[t * 5 / 60] = metrics.auc(fpr, tpr)
    return aucs


def plot_auc_over_time(aucs, path):
    auc_scores = aucs.items()
    x, y = zip(*auc_scores)
    plt.plot(x, y)

    plt.xlabel("time (h)")
    plt.ylabel("auc")
    plt.title("auc over time")
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

    for con_train_n, diff_train_n, con_test_n, diff_test_n in [
        (1, 5, 2, 3), (2, 3, 1, 5),
        (2, 3, 2, 3), (1, 5, 1, 5),
    ]:
        diff_df_train, con_df_train, con_df_test, diff_df_test = get_to_run(path=path, to_run=to_run,
                                                                            con_train_num=con_train_n,
                                                                            con_test_num=con_test_n,
                                                                            diff_train_num=diff_train_n,
                                                                            diff_test_num=diff_test_n,
                                                                            local_density=local_density)

        dir_path = f"30-03-2022-manual_mastodon_{to_run} local density-{local_density}, s{con_train_n}, s{diff_train_n} are train"
        second_dir = f"{diff_window} frames ERK, {con_windows} frames con track len {tracks_len}"
        utils.open_dirs(dir_path, second_dir)
        dir_path += "/" + second_dir
        clf, X_train, X_test, y_train, y_test = utils.load_data(dir_path)

        cols = list(X_train.columns)
        cols.extend(["Spot track ID", "Spot frame"])

        con_df_test = con_df_test[cols]
        diff_df_test = diff_df_test[cols]

        print("calc AUC")
        aucs = auc_over_time(con_df_test, diff_df_test, clf)
        plot_auc_over_time(aucs, dir_path + f"/auc over time s{diff_test_n}, s{con_test_n}.png")

        print("calc avg prob")
        df_score_con = utils.calc_prob(con_df_test, clf, n_frames=260)
        df_score_dif = utils.calc_prob(diff_df_test, clf, n_frames=260)

        pickle.dump(df_score_con, open(dir_path + f"/df_prob_w=30, video_num={con_test_n}", 'wb'))
        pickle.dump(df_score_dif, open(dir_path + f"/df_prob_w=30, video_num={diff_test_n}", 'wb'))
        plot_avg_conf(df_score_con.drop("Spot track ID", axis=1), df_score_dif.drop("Spot track ID", axis=1),
                      mot_int=to_run,
                      path=dir_path + f"/avg conf s{con_test_n}, s{diff_test_n}.png")

        print()
