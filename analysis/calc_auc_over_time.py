import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics


def plot_auc_over_time(aucs_lst, path=None, time=(0, 25)):
    """
    Plots AUC (Area Under the Curve) values over time, based on the input list of AUC scores.
    :param aucs_lst: (list) List of tuples containing AUC scores and corresponding labels.
    The AUC scores are expected to be in the form of a dictionary, with time as keys and AUC values as values.
    :param path: (str, optional) The file path to save the plot as an EPS file. Defaults to None.
    :param time: (tuple, optional) The time range (in hours) to consider for plotting. Defaults to (0, 25).
    :return: None
    """
    for aucs, label in aucs_lst:
        auc_scores = pd.DataFrame({"time": aucs.keys(), "auc": aucs.values()})
        auc_scores = auc_scores[(auc_scores["time"] >= time[0]) & (auc_scores["time"] <= time[1])]
        plt.plot(auc_scores["time"], auc_scores["auc"], label=label)

    plt.axhline(0.5, color='gray', linestyle='dashed')
    plt.xlabel("time (h)")
    plt.ylabel("auc")
    plt.title("auc over time")
    plt.ylim((0.2, 1))
    plt.legend()
    if path:
        plt.savefig(path, format="eps")
    plt.show()
    plt.clf()


def auc_over_time(df_con, df_diff, clf):
    """
    Calculates AUC (Area Under the Curve) values over time based on predicted results. The AUC values are calculated
    using the true labels (0 for control, 1 for differential) and the predicted scores. The time points are derived from
    the "Spot frame" column of the control dataframe.
    :param df_con: (pandas.DataFrame) The dataframe containing control data (DMSO).
    :param df_diff: (pandas.DataFrame) The dataframe containing differentiated cells data (ERKi).
    :param clf: The classifier used for prediction.
    :return: (dict) A dictionary where the keys are time points and the values are the corresponding AUC values.
    """

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
    auc_scores = {}
    for t in time_points:
        t_pred_con = get_t_pred(df_con)
        t_pred_diff = get_t_pred(df_diff)
        true_pred_con = [0 for i in range(len(t_pred_con))]
        true_pred_diff = [1 for i in range(len(t_pred_diff))]

        fpr, tpr, thresholds = metrics.roc_curve(true_pred_con + true_pred_diff, t_pred_con + t_pred_diff,
                                                 pos_label=1)
        auc_scores[t * 5 / 60] = metrics.auc(fpr, tpr)
    return auc_scores
