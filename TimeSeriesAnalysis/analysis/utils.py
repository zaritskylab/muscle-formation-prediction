import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import roc_curve
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import matplotlib as mpl

from TimeSeriesAnalysis.analysis.calc_auc_over_time import auc_over_time, plot_auc_over_time
from TimeSeriesAnalysis.data_layer.utils import load_tsfresh_transformed_df




def plot_roc(clf, X_test, y_test, path=None):
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
    if path:
        plt.savefig(path + "/" + 'ROC.eps', format="eps")
    plt.show()
    plt.close()
    plt.clf()


def plot_avg_conf(conf_data, modality, path="", plot_std=True, frames=(0, 260), xlim=(2.5, 24)):
    """
    :param conf_data: [(df_score_dif.drop("Spot track ID", axis=1), "ERK", "DarkOrange","Orange"),(df_score_con.drop("Spot track ID", axis=1), "Control", "blue", "blue")]
    :param mot_int: modality name
    :param path:
    :param plot_std:
    :return:
    """
    fig = plt.figure(figsize=(6, 4))

    def plot(df, modality, avg_color, std_color, label, plot_std, frames):
        time = np.arange(0, df["Spot frame"].max() + 1) * 1 / 12
        avg_vals_diff = df.groupby("time")[f"score_{modality}"].mean()
        std_vals_diff = df.groupby("time")[f"score_{modality}"].std()

        plt.plot(avg_vals_diff, color=avg_color, label=label)

        if plot_std:
            plt.fill_between(time, avg_vals_diff - std_vals_diff, avg_vals_diff + std_vals_diff, alpha=0.4,
                             color=std_color)

    plt.grid(False)
    plt.xlabel("time (h)")
    plt.ylabel("avg score")
    plt.title(f"avg differentiation score over time ({modality})")
    plt.axhline(0.5, color='gray', linestyle='dashed')
    plt.axvspan(frames[0] * 5 / 60, frames[1] * 5 / 60, alpha=0.6, color='lightgray')
    plt.axvline(frames[0] * 5 / 60, color='gray', linestyle='dashed')
    plt.axvline(frames[1] * 5 / 60, color='gray', linestyle='dashed')

    for (df, label, avg_color, std_color) in conf_data:
        plot(df, modality, avg_color, std_color, label, plot_std, frames)

    plt.ylim((-0.1, 1.1))
    plt.xlim(xlim)
    plt.savefig(path + '.eps', format='eps')
    plt.show()
    plt.clf()


def get_correlation_coefficients(data, time, x_prop, y_prop, rolling_w, corr_metric):
    data = data[(data["time"] >= time[0]) & (data["time"] <= time[1])]
    data = data.astype('float64')
    corr = np.array(data.groupby('Spot track ID').apply(
        lambda df: df[x_prop].rolling(rolling_w).mean().corr(df[y_prop].rolling(rolling_w).mean(), method=corr_metric)))
    return corr


def plot_diff_trajectories_single_cells(conf_data, modality, path=None, rolling_w=1):
    def plot(df, modality):
        for track_id, track in df.groupby("Spot track ID"):
            track = track[(track["Spot frame"] <= track["fusion_frame"])].sort_values("time")
            track[f"score_{modality}"] = track[f"score_{modality}"].rolling(rolling_w).mean()
            monotonicity = get_correlation_coefficients(track, (6, 13), f"score_{modality}", "time", rolling_w=6,
                                                        corr_metric="spearman")[0]

            fig = plt.figure(figsize=(6, 3))
            cmap = mpl.cm.get_cmap('RdYlGn_r')
            norm = mpl.colors.Normalize(vmin=-1, vmax=1)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.plot(track["time"], track[f"score_{modality}"], linewidth=1, color=cmap(monotonicity), label=track_id)
            plt.axhline(0.5, color='black', linestyle='dashed')
            plt.legend()
            plt.xlabel("time (h)")
            plt.ylabel("avg score")
            plt.title(f"avg differentiation score over time ({modality})")
            plt.ylim((0, 1))
            plt.xlim((4, 22))
            plt.colorbar(sm, ticks=np.linspace(-1, 1, 5))
            if path:
                plt.savefig(
                    path + "/" + f'{np.round(monotonicity, 4)} monotonicity diff_over_t ({modality}) id={track_id}.eps',
                    format="eps")
            plt.show()
            plt.clf()

    for (df, label) in conf_data:
        df = df.astype("float")
        plot(df, modality)


def evaluate_model(clf, x_test, y_test, modality, con_test_n, diff_test_n):
    y_pred = clf.predict(x_test[clf.feature_names_in_])
    y_test = y_test["target"]

    accuracy = round(metrics.accuracy_score(y_test, y_pred), 3)
    sensitivity = round(metrics.recall_score(y_test, y_pred), 3)
    precision = round(metrics.precision_score(y_test, y_pred), 3)

    # calculate specificity
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
    specificity = round(tn / (tn + fp), 3)

    # calculate AUC score
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    auc = round(metrics.auc(fpr, tpr), 3)

    print(f"Evaluation metrics for {modality} based model, tested on videos (s{con_test_n}, s{diff_test_n}):")
    print(f"accuracy: {accuracy}")
    print(f"specificity: {specificity}")
    print(f"sensitivity (recall): {sensitivity}")
    print(f"precision: {precision}")
    print(f"AUC: {auc}")

    # plot ROC curve
    plot_roc(clf=clf, X_test=x_test.drop(columns=['Spot track ID']), y_test=y_test, path="")

    # calculate AUC over time
    cols = list(clf.feature_names_in_) + ["Spot track ID", "Spot frame"]
    df_con = load_tsfresh_transformed_df(modality, con_test_n, cols)
    df_diff = load_tsfresh_transformed_df(modality, diff_test_n, cols)
    aucs = auc_over_time(df_con[cols], df_diff[cols], clf)
    plot_auc_over_time([(aucs, modality)],
                                      path=consts.storage_path + f"eps_figs/auc_over_time s{con_test_n}, s{diff_test_n} {modality}.eps")

