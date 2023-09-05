import numpy as np
from scipy.stats import normaltest
from sklearn.metrics import roc_curve
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd

from analysis.calc_auc_over_time import auc_over_time, plot_auc_over_time
from analysis.calc_single_cell_properties import get_monotonicity
from data_layer.utils import load_tsfresh_transformed_df, convert_score_df
from configuration import consts


def plot_violin_distributions(data, feature, modalities, plot_window=False, xlim=None, fig_path=None):
    """
    Plots violin distributions and swarmplot for the given data and feature.

    :param data: (pd.DataFrame) The data to plot.
    :param feature: (str) The name of the feature to plot.
    :param modalities: (list) The list of modalities to include in the plot, fore example: ["motility", "intensity"]
    :param plot_window: (bool) Whether to plot a window on the x-axis.
    :param xlim: (tuple) The x-axis limits.
    :param fig_path: (str) The path to save the figure (optional).
    """
    print(f"number of cells in the analysis: {data['Spot track ID'].nunique()}")
    df = pd.DataFrame()
    medians = []
    for modality in modalities:
        col = f"{feature}_{modality}".rstrip('_')
        p_value = normaltest(data[col], nan_policy='omit')[1] if data['Spot track ID'].nunique() >= 8 else None

        mean = round(data[col].mean(), 3)
        std = round(data[col].std(), 3)
        median = round(data[col].median(), 3)
        medians.append(median)
        percentage_zero_above = round(
            100 * (len(data[data[col] > 0]) / len(data)), 3)

        print(
            f"{modality}: normality test p-val={p_value}, mean: {mean}, std: {std}, median: {median}, percentage >=0: {percentage_zero_above} %")
        df = df.append(pd.DataFrame({feature: data[col], "data": modality}),
                       ignore_index=True)

    sns.catplot(data=df, x=feature, y="data", kind="violin", color=".9", inner="quartile")
    sns.swarmplot(data=df, x=feature, y="data", size=3)

    if xlim:
        plt.xlim(xlim)
    if plot_window:
        plt.axvspan(6, 13, alpha=0.6, color='lightgray')
        plt.axvline(6, color=".8", linestyle='dashed')
        plt.axvline(13, color=".8", linestyle='dashed')

    fig_path = consts.storage_path + f"eps_figs/dist_{feature}_{str(modalities)}.eps" if fig_path is None else fig_path
    plt.savefig(fig_path, format="eps")

    plt.show()


def plot_roc(clf, X_test, y_test, path=None):
    """
    Plots the ROC curve for a classifier.

    :param clf: The classifier model.
    :param X_test: The test data features.
    :param y_test: The test data labels.
    :param path: (str) The path to save the figure (optional).
    """
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


def plot_avg_conf(conf_data, modality, path="", plot_std=True, time=(0, 24), xlim=(2.5, 24), ylim=(-0.1, 1.1),
                  axhline_val=0.5):
    """
    Plots the average differentiation score over time.

    :param conf_data: (list) A list of tuples containing the dataframes, labels, average colors, and std colors. for
    example: [(df_score_dif.drop("Spot track ID", axis=1), "ERK", "DarkOrange","Orange"),
            (df_score_con.drop("Spot track ID", axis=1), "Control", "blue", "blue")]
    :param modality: (str) The modality name. Can be "motility" or "intensity".
    :param path: (str) The path to save the figure (optional).
    :param plot_std: (bool) Whether to plot the standard deviation.
    :param time: (tuple) The time range to plot (start, end).
    :param xlim: (tuple) The x-axis limits.
    :param ylim: (tuple) The y-axis limits.
    :param axhline_val: (float) The value for the horizontal line.
    """
    plt.figure(figsize=(6, 4))

    def plot(df, modality, avg_color, std_color, label, plot_std):
        avg_vals_diff = df.groupby("time")[f"score_{modality}"].mean()
        std_vals_diff = df.groupby("time")[f"score_{modality}"].std()
        time = np.arange(df["Spot frame"].min(), df["Spot frame"].max() + 1) * 1 / 12
        # time = np.arange(0, len(std_vals_diff) + 1) * 1 / 12

        plt.plot(avg_vals_diff, color=avg_color, label=label)

        if plot_std:
            plt.fill_between(time, avg_vals_diff - std_vals_diff, avg_vals_diff + std_vals_diff, alpha=0.4,
                             color=std_color)

    plt.grid(False)
    plt.xlabel("time (h)")
    plt.ylabel("avg score")
    plt.title(f"avg differentiation score over time ({modality})")
    plt.axhline(axhline_val, color='gray', linestyle='dashed')
    plt.axvspan(time[0], time[1], alpha=0.6, color='lightgray')
    plt.axvline(time[0], color='gray', linestyle='dashed')
    plt.axvline(time[1], color='gray', linestyle='dashed')

    for (df, label, avg_color, std_color) in conf_data:
        plot(df, modality, avg_color, std_color, label, plot_std)

    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.savefig(path + '.eps', format='eps')
    plt.show()
    plt.clf()


def get_correlation_coefficients(data, time, x_prop, y_prop, rolling_w, corr_metric):
    """
        Calculates the correlation coefficients between two properties over a specified time range.

        :param data: (pd.DataFrame) The data containing the properties.
        :param time: (tuple) The time range to consider (start, end).
        :param x_prop: (str) The name of the first property.
        :param y_prop: (str) The name of the second property.
        :param rolling_w: (int) The rolling window size for calculating the correlation.
        :param corr_metric: (str) The correlation metric to use.
        :return: (np.array) The correlation coefficients.
        """
    data = data[(data["time"] >= time[0]) & (data["time"] <= time[1])]
    data = data.astype('float64')
    corr = data.groupby('Spot track ID').apply(
        lambda df: df[x_prop].rolling(rolling_w).mean().corr(df[y_prop].rolling(rolling_w).mean(), method=corr_metric))
    corr = np.array(corr)
    return corr


def plot_diff_trajectories_single_cells(conf_data, modality, path=None, rolling_w=1, x_label="time"):
    """
    Plots the differentiation score trajectories for individual cells.

    :param conf_data: (list) A list of tuples containing the dataframes and labels.
    :param modality: (str) The modality name.
    :param path: (str) The path to save the figures (optional).
    :param rolling_w: (int) The rolling window size for smoothing the trajectories.
    :param x_label: (str) The label for the x-axis.
    """

    def plot(df, modality, x_label):
        for track_id, track in df.groupby("Spot track ID"):
            track = track[(track["Spot frame"] <= track["fusion_frame"])].sort_values("time")
            track[f"score_{modality}"] = track[f"score_{modality}"].rolling(rolling_w).mean()
            monotonicity = \
                get_correlation_coefficients(track, (6, 13), f"score_{modality}", "time", rolling_w=rolling_w,
                                             corr_metric="spearman")[0]

            plt.figure(figsize=(6, 3))
            cmap = mpl.cm.get_cmap('RdYlGn_r')
            norm = mpl.colors.Normalize(vmin=-1, vmax=1)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.plot(track[x_label], track[f"score_{modality}"], linewidth=1, color=cmap(monotonicity), label=track_id)
            plt.axhline(0.5, color='black', linestyle='dashed')
            plt.legend()
            plt.xlabel(x_label)
            plt.ylabel(f"score_{modality}")
            plt.title(f"differentiation score over time ({modality})")
            plt.ylim((0, 1))
            # plt.xlim((4, 22))
            plt.colorbar(sm, ticks=np.linspace(-1, 1, 5))
            if path:
                plt.savefig(
                    path + "/" + f'{np.round(monotonicity, 4)} monotonicity diff_over_t ({modality}) id={track_id}.eps',
                    format="eps")
            plt.show()
            plt.clf()

    for (df, label) in conf_data:
        df = df.astype("float")
        plot(df, modality, x_label)


def evaluate_model(clf, x_test, y_test, modality, con_test_n, diff_test_n, plot_auc_over_t=True):
    """
    Calculates the evaluation metrics and AUC score over time for a classifier model.

    :param clf: (object) The classifier model used for prediction.
    :param x_test: (pd.DataFrame) The feature matrix for testing.
    :param y_test: (pd.DataFrame) The target variable for testing.
    :param modality: (str) Modality for which model is evaluated, Choose from "motility" or "intensity".
    :param con_test_n: (int) The number of control test videos.
    :param diff_test_n: (int) The number of differentiation test videos.
    :param plot_auc_over_t: (bool, optional) Whether to plot the AUC over time. Default is True.

    :return: (float) The AUC (Area Under the Curve) score.
    """
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

    plot_roc(clf=clf, X_test=x_test.drop(columns=['Spot track ID'], errors='ignore'), y_test=y_test, path="")
    if plot_auc_over_t:
        # calculate AUC over time
        cols = list(clf.feature_names_in_) + ["Spot track ID", "Spot frame"]
        df_con = load_tsfresh_transformed_df(modality, con_test_n, cols)
        df_diff = load_tsfresh_transformed_df(modality, diff_test_n, cols)
        aucs = auc_over_time(df_con[cols], df_diff[cols], clf)
        plot_auc_over_time([(aucs, modality)],
                           path=consts.storage_path + f"eps_figs/auc_over_time s{con_test_n}, s{diff_test_n} {modality}.eps")
    return auc


def get_mean_properties_in_range(track, feature, my_range):
    """
        Calculates the mean values of selected properties within a specified range for a given track.

        :param track: (pd.DataFrame) The track data.
        :param feature: (str) The feature/column to calculate mean value for.
        :param my_range: (tuple) The range of values to consider (start, end).
        :return: (pd.DataFrame) Mean values of selected properties within the specified range.
        """
    track = track[(track[feature] >= my_range[0]) & (track[feature] <= my_range[1])]
    track = track[[f"score_motility", "score_intensity", "speed", "mean", "persistence", "local density", "time"]]
    mean_values_df = pd.DataFrame(track.mean()).T
    mean_values_df["range"] = str(my_range)
    return mean_values_df


def plot_props_by_range(track, feature_name, ranges):
    """
        Plots the mean values of selected properties within specified ranges for a given track.

        :param track: (pd.DataFrame) The track data.
        :param feature_name: (str) The name of the feature/column to calculate mean value for.
        :param ranges: (list) List of ranges to consider for plotting (each range as a tuple).
        :return: None
    """
    track = track[track["Spot frame"] <= track["fusion_frame"]]

    df = pd.DataFrame()
    for my_range in ranges:
        data = get_mean_properties_in_range(track, feature_name, my_range)
        df = df.append(data, ignore_index=True)

    for subplot_n, col in zip(np.arange(711, 717),
                              ["score_motility", "score_intensity", "speed", "mean", "persistence", "local density"]):
        ax = plt.subplot(subplot_n)
        ax.plot(df["range"], df[col], linestyle='--', marker='o', label=col)
        plt.legend(loc='lower right')
        ax.set_yticks((round(df[col].min()), round(df[col].max())))

    plt.show()


def plot_violin_spearman_corr(data_list, modalities, color, corr_metric, ylim=(-1.5, 1.5), time=(6, 13), fig_name=None):
    """
        Plots violin plots showing the distribution of correlation coefficients between differentioation score
        and time for different modalities.

        :param data_list: (list) List of dataframes containing the score data for each modality.
        :param modalities: (list) List of modality names. For example: ["motility", "intensity"]
        :param color: (str) Color for the violin plots.
        :param corr_metric: (str) The correlation metric to use.
        :param ylim: (tuple) The y-axis limits for the plot.
        :param time: (tuple) The time range to consider for correlation calculation.
        :param fig_name: (str) Name for the figure file (optional).
        :return: None
        """
    sns.set_style("white")
    corrs = []
    for df, modality in zip(data_list, modalities):
        df = convert_score_df(df, modality)
        corr = get_correlation_coefficients(df, time, f"score_{modality}", "time", rolling_w=5,
                                            corr_metric="spearman")
        corrs.append(corr[~np.isnan(corr)])
        print(modality, "- median: ", round(np.nanmedian(corr), 2), "size of data: ", len(corr))

    plt.axhline(0, color='gray', linestyle='dashed')
    df = pd.DataFrame()
    for i, col in enumerate(modalities):
        tmp_df = pd.DataFrame({col: corrs[i]})
        df = df.append(tmp_df.dropna(), ignore_index=True)
    sns.violinplot(data=df, color=color, cut=0, inner='box')

    plt.xticks(rotation=30, ha='right', fontsize='large')
    plt.ylabel("corr")
    plt.title(f"single cell corr between score models & time corr metric-{corr_metric}")
    plt.ylim(ylim)
    fig_name = f"single cell correlation time & score cls" if fig_name is None else fig_name
    plt.savefig(consts.storage_path + f"eps_figs/" + fig_name + f" compare metric={corr_metric}.eps", format="eps")
    plt.show()


def remove_speed_outlaiers(scores_df, speed_threshold):
    # cells with missing timepont in the track (causes a bias in speed)
    miss_step_tracks = set()
    for track_id, track in scores_df.groupby(['Spot track ID']):
        if track["Spot frame"].diff().max() > 1:
            miss_step_tracks.add(track_id)

    # cells with speed higher then 20
    speed_outliars = set(scores_df[scores_df["speed"] >= speed_threshold]['Spot track ID'].unique())

    # remove bad tracks for calculation
    track_ids = set(scores_df['Spot track ID'].unique()) - speed_outliars - miss_step_tracks

    fraction = round((len(speed_outliars) + len(miss_step_tracks)) / scores_df['Spot track ID'].nunique(), 3)

    return track_ids

def plot_monotonicity_dist(df, vid_name, time, rolling_w, modality, color, pval_thresh):
    """
    Calculates monotonicity rates for single cells and plots the distribution
    :pararm df: (pd.DataFrame) single cell's track dataframe, with differntiation scores by motility & actin intensity models
    :pararm vid_name: (Str) name of video the data eas extracted from.
    :pararm time: (tuple) time range to calculate correlationin (start_time, end_time).
    :pararm rolling_w: (int) size of window for rolling average smoothing.
    :pararm modality: (Str) "motility"/"intensity".
    """
    sns.set_style("white")
    mono_values = []
    mono_p_values = []
    for track_id, track_df in df.groupby("Spot track ID"):
        if len(track_df) > rolling_w:
            mono = get_monotonicity(track_df, modality, time, rolling_w, return_p_val=False)[
                f"monotonicity_{modality}"].values[0]
            mono_p_val = get_monotonicity(track_df, modality, time, rolling_w, return_p_val=True)[
                f"monotonicity_{modality}"].values[0]
            mono_values.append(mono)
            mono_p_values.append(mono_p_val)

    mean = round(np.nanmean(mono_values), 3)
    median = round(np.nanmedian(mono_values), 3)

    sns.histplot(mono_values, stat="percent", bins=10, color=color)
    plt.axvline(median, linestyle='dashed', color="black")
    plt.yticks(np.arange(0, 45, 15))
    plt.savefig(consts.storage_path + f"eps_figs/monotonicity distribution vid {vid_name} {modality}.eps", format="eps")
    plt.show()

    print(modality, "mean", mean, "median", median)
    print("Number of cells in this analysis: ", len(mono_values))

    # print the % of cells with p-value smaller then pval_thresh
    num_pval_lower_then_thresh = sum(x < pval_thresh for x in np.array(mono_p_values))


    print(
        f"% of cells with p-value smaller then pval_thresh: {round((num_pval_lower_then_thresh / len(mono_p_values)) * 100, 3)}")