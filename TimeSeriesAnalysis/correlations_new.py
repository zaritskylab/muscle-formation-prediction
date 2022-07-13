import pickle
import consts
import diff_tracker_utils as utils
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from build_models_on_transformed_tracks import get_to_run
from scipy.spatial import distance
import scipy
import matplotlib as mpl


def load_correlations_data(s_run, dir_path_score):
    # get coordination df
    df_coord = pickle.load(
        (open(
            path + f"/Coordination/coordination_outputs/coordination_dfs/manual_tracking - only tagged tracks/coord_mastodon_s{s_run['name'][1]}.pkl",
            'rb')))
    df_coord["Spot track ID"] = df_coord["Track #"].apply(lambda x: x)
    df_coord["Spot frame"] = df_coord["cos_theta"].apply(lambda x: x[0])
    df_coord["cos_theta_values"] = df_coord["cos_theta"].apply(lambda x: x[1])

    # get single cells score df
    df_score = pickle.load(open(dir_path_score + f"/df_prob_w=30, video_num={s_run['name'][1]}", 'rb'))
    print(df_score.shape)

    # calculate local density
    df_all_tracks, _ = utils.get_tracks(path + s_run["csv_all_path"], manual_tagged_list=False)
    # df = pd.read_csv(path + s_run["csv_tagged_path"], encoding="ISO-8859-1")
    df_tagged = df_all_tracks[df_all_tracks["manual"] == 1]  # todo change

    # local_density_df = utils.add_features_df(df_tagged, df_all_tracks, local_density=local_density)

    df_score['cos_theta'] = -1
    df_score['t0'] = -1
    df_score['cos_theta'] = df_score['cos_theta'].astype('object')
    for index, row in df_score.iterrows():
        track_id = row["Spot track ID"]
        cos_theta_list = df_coord[df_coord["Spot track ID"] == track_id]["cos_theta_values"].to_list()
        df_score['cos_theta'].iloc[index] = [cos_theta_list]
        df_score['t0'].iloc[index] = df_coord[df_coord["Spot track ID"] == track_id]["t0"].max()

    return df_score, df_tagged, df_all_tracks


def get_dir_path(to_run):
    dir_path_score = f"30-03-2022-manual_mastodon_{to_run} local density-{local_density}, s{con_train_n}, s{diff_train_n} are train"
    second_dir = f"{diff_window} frames ERK, {con_windows} frames con track len {tracks_len}"
    utils.open_dirs(dir_path_score, second_dir)
    dir_path_score += "/" + second_dir
    return dir_path_score


def bin_score_values(df):
    cols = df.drop(columns=["Spot track ID", "cos_theta", "t0"]).columns
    for col in cols:
        # df_[col] = df_[col].astype(float)
        df[col] = pd.cut(df[col],
                         [-1, -.9, -.8, -.7, -.6, -.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5,
                          .6, .7, .8, .9, 1],
                         labels=[-.9, -.8, -.7, -.6, -.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5,
                                 .6, .7, .8, .9, 1])
    return df


def pearson_lag_models_correlation(df_merge_col_mot, df_merge_col_int, s_run, save_dir, pct_change, binned_score):
    lags = np.arange(-65, 65, 5)
    lag_pearson_df = pd.DataFrame()
    if binned_score:
        df_merge_col_mot = bin_score_values(df_merge_col_mot)
        df_merge_col_int = bin_score_values(df_merge_col_int)

    for lag in lags:
        # lag = 0
        print(lag)
        cols = {x: x + lag for x in range(259)}
        df_merge_col_mot_copy = df_merge_col_mot.copy().rename(columns=cols)
        lag_pearsons = get_pearsons_r(df_merge_col_mot_copy, df_merge_col_int, pct_change)
        lag_pearson_df[lag] = pd.Series(lag_pearsons)

    pallete = "Oranges" if s_run["name"] in ["S3", "S5"] else "Blues"
    sns.set_theme(style="whitegrid")
    sns.boxplot(data=lag_pearson_df.values, palette=pallete)
    sns.despine(offset=10, trim=True)
    plt.xlabel("lag")
    plt.ylabel("pearson r")
    plt.xticks(range(len(lag_pearson_df.columns)), np.arange(-65, 65, step=5))
    plt.title(f"correlation between motility & intensity scores, {s_run['name']}")
    plt.savefig(save_dir + f"correlations_motility score and intensity score {s_run['name']}.png")
    plt.show()
    plt.clf()
    plt.close()


def get_single_cell_data(track_id, df_merge_col_mot, df_merge_col_int, pct_change=False):
    score_df_mot = df_merge_col_mot[df_merge_col_mot["Spot track ID"] == track_id]
    score_df_int = df_merge_col_int[df_merge_col_int["Spot track ID"] == track_id]
    if len(score_df_int) > 0 and len(score_df_mot) > 0:
        score_df_int = score_df_int.drop(columns=["Spot track ID", "cos_theta", "t0"])
        score_df_mot = score_df_mot.drop(columns=["Spot track ID", "cos_theta", "t0"])

        score_int = score_df_int.iloc[:1, : 259].T.dropna()[score_df_int.index]
        score_mot = score_df_mot.iloc[:1, : 259].T.dropna()[score_df_mot.index]
        if pct_change:
            score_int = score_int.astype(float).pct_change()
            score_mot = score_int.astype(float).pct_change()
        return score_int, score_mot
    else:
        return [], []


def calc_pearson_r(score_int, score_mot):
    try:
        score_int.reset_index(inplace=True, drop=False)
        score_mot.reset_index(inplace=True, drop=False)
        df = pd.merge(score_int, score_mot, on=['index'])
        df = df.dropna()
        pearson_r = scipy.stats.pearsonr(df.iloc[:, 1], df.iloc[:, 2])
        return pearson_r
    except:
        return None


def get_pearsons_r(df_merge_col_mot, df_merge_col_int, pct_change):
    pearson_rs = []
    for ind, row in df_merge_col_mot.iterrows():
        track_id_mot = row["Spot track ID"]
        score_int, score_mot = get_single_cell_data(track_id_mot, df_merge_col_mot, df_merge_col_int, pct_change)
        pearson_r = calc_pearson_r(score_int, score_mot)
        if pearson_r:
            pearson_rs.append(pearson_r[0])
    return pearson_rs


def plot_distribution(color, data, title, save_dir, xlabel):
    sns.displot(data=data, kde=True, color=color)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.savefig(save_dir + title + f".png")
    plt.show()
    plt.clf()
    plt.close()


def plot_correlation_of_population_time_division(save_dir, title, data_tuple_list, pallete, color, xlabel, ylabel,
                                                 time_slots,
                                                 population_mean):
    fig, axs = plt.subplots(1, len(data_tuple_list), figsize=(16, 4), facecolor='w', edgecolor='k')
    for i, ((t_start, t_end), (x_data, y_data)) in enumerate(zip(time_slots, data_tuple_list)):
        if population_mean:
            scatter = axs[i].scatter(x_data, y_data, cmap=pallete, c=x_data.index * 5 / 60,
                                     norm=mpl.colors.Normalize(vmin=0, vmax=22))
        else:  # plot every single value
            new_df = pd.DataFrame()
            for col in x_data.columns:
                tmp_df = pd.DataFrame({"x_data": x_data[col], "y_data": y_data[col], "Spot frame": x_data.index})
                new_df = new_df.append(tmp_df)
            new_df = new_df.astype(float)
            new_df = new_df.reset_index()
            new_df = new_df.dropna()
            axs[i].hist2d(new_df["x_data"], new_df["y_data"], cmap=pallete)

        axs[i].set_title(f"time window: [{round(t_start * 5 / 60)}, {round(t_end * 5 / 60)}] (hours)")
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)
        axs[i].set_ylim(0, 0.9)
        axs[i].set_xlim(0, 0.9)
    # plt.colorbar(scatter)
    plt.suptitle(title)
    plt.savefig(save_dir + title + f".png")
    plt.show()
    plt.clf()
    plt.close()


def plot_pearson_dist_time_division(save_dir, title, pearson_data_list, color, xlabel, time_slots):
    fig, axs = plt.subplots(1, len(pearson_data_list), figsize=(18, 4), facecolor='w', edgecolor='k')

    for i, ((t_start, t_end), x_data) in enumerate(zip(time_slots, pearson_data_list)):
        # sns.displot(ax=axs[i], data=x_data, kde=True, color=color)
        axs[i].hist(x_data, color=color, alpha=0.7, rwidth=0.85)
        axs[i].set_title(f"time window: [{round(t_start * 5 / 60)}, {round(t_end * 5 / 60)}] (hours)")
        axs[i].set_xlabel(xlabel)
    plt.suptitle(title)
    plt.savefig(save_dir + title + ".png")
    plt.show()
    plt.clf()
    plt.close()


def global_correlation_mean_plots(save_dir, df_merge_mot_con, df_merge_int_con, df_merge_mot_diff,
                                  df_merge_int_diff, srun_con_n, srun_diff_n, pct_change, binned_score,
                                  time_division=False):
    time_slots = [(0, 72), (72, 170), (170, 260)]

    if binned_score:
        df_merge_mot_con = bin_score_values(df_merge_mot_con)
        df_merge_int_con = bin_score_values(df_merge_int_con)

        df_merge_mot_diff = bin_score_values(df_merge_mot_diff)
        df_merge_int_diff = bin_score_values(df_merge_int_diff)

    df_merge_mot_con = df_merge_mot_con[
        df_merge_mot_con["Spot track ID"].isin(df_merge_int_con["Spot track ID"].unique())]

    df_merge_mot_diff = df_merge_mot_diff[
        df_merge_mot_diff["Spot track ID"].isin(df_merge_int_diff["Spot track ID"].unique())]

    df_merge_mot_con.index = df_merge_mot_con["Spot track ID"]
    df_merge_int_con.index = df_merge_int_con["Spot track ID"]

    df_merge_mot_diff.index = df_merge_mot_diff["Spot track ID"]
    df_merge_int_diff.index = df_merge_int_diff["Spot track ID"]

    df_mot_con = df_merge_mot_con.drop(columns=["cos_theta", "t0", "Spot track ID"])
    df_int_con = df_merge_int_con.drop(columns=["cos_theta", "t0", "Spot track ID"])

    df_mot_diff = df_merge_mot_diff.drop(columns=["cos_theta", "t0", "Spot track ID"])
    df_int_diff = df_merge_int_diff.drop(columns=["cos_theta", "t0", "Spot track ID"])

    fig = plt.Figure(figsize=(16, 4), facecolor='w', edgecolor='k')
    scatter1 = plt.scatter(df_mot_con.mean(), df_int_con.mean(), cmap="Blues", c=df_mot_con.T.index * 5 / 60,
                           norm=mpl.colors.Normalize(vmin=0, vmax=22), label="control")
    scatter2 = plt.scatter(df_mot_diff.mean(), df_int_diff.mean(), cmap="Oranges", c=df_mot_diff.T.index * 5 / 60,
                           norm=mpl.colors.Normalize(vmin=0, vmax=22), label="ERK")
    title = f"global correlation- mean differentiation score per time point, {srun_con_n}, {srun_diff_n}"
    plt.title(title)
    plt.xlabel("avg motility differentiation score")
    plt.ylabel("avg intensity differentiation score")
    plt.ylim(0, 0.9)
    plt.xlim(0, 0.9)
    plt.colorbar(scatter1)
    plt.colorbar(scatter2)
    plt.legend()
    plt.savefig(save_dir + title + f".png")
    plt.show()
    plt.clf()
    plt.close()


def global_correlation(df_merge_col_mot, df_merge_col_int, s_run, save_dir, pct_change, binned_score,
                       time_division=False, pearson=False):
    color = "Orange" if s_run["name"] in ["S3", "S5"] else "Blue"
    pallete = "Oranges" if s_run["name"] in ["S3", "S5"] else "Blues"

    # time_slots = [(0, 72), (72, 148), (148, 200), (200, 260)]
    time_slots = [(0, 72), (72, 170), (170, 260)]

    if binned_score:
        df_merge_col_mot = bin_score_values(df_merge_col_mot)
        df_merge_col_int = bin_score_values(df_merge_col_int)

    if pearson:  # plot correlation using pearson correlation of single cell
        if time_division:
            pearsons_t_list = []
            for i, (t_start, t_end) in enumerate(time_slots):
                df_mot_t = df_merge_col_mot.iloc[:, np.r_[t_start:t_end, -3:-0]]
                df_int_t = df_merge_col_int.iloc[:, np.r_[t_start:t_end, -3:-0]]
                pearsons_t = get_pearsons_r(df_mot_t, df_int_t, pct_change)
                pearsons_t_list.append(pearsons_t)
            plot_pearson_dist_time_division(save_dir,
                                            f"pearson correlation distribution of actin intensity  and motility models, by time slots, vid {s_run['name']}",
                                            pearsons_t_list, color, "pearson value", time_slots)

        else:
            pearsons = get_pearsons_r(df_merge_col_mot, df_merge_col_int, pct_change)
            plot_distribution(color, pd.Series(pearsons),
                              f"distribution of single cell's correlation between \nactin intensity model & motility model, vid {s_run['name']}",
                              save_dir, "pearson r")
    else:  # plot simple correlation - motility over intensity
        df_merge_col_mot = df_merge_col_mot[
            df_merge_col_mot["Spot track ID"].isin(df_merge_col_int["Spot track ID"].unique())]

        df_merge_col_mot.index = df_merge_col_mot["Spot track ID"]
        df_merge_col_int.index = df_merge_col_int["Spot track ID"]

        df_mot = df_merge_col_mot.drop(columns=["cos_theta", "t0", "Spot track ID"]).T
        df_int = df_merge_col_int.drop(columns=["cos_theta", "t0", "Spot track ID"]).T

        if time_division:
            df_time_list_pop_mean = []
            df_time_list = []
            for i, (t_start, t_end) in enumerate(time_slots):
                df_mot_t = df_mot.iloc[t_start:t_end]
                df_int_t = df_int.iloc[t_start:t_end]
                df_time_list.append((df_mot_t, df_int_t))
                df_time_list_pop_mean.append((df_mot_t.T.mean(), df_int_t.T.mean()))
            # plot mean population
            plot_correlation_of_population_time_division(save_dir,
                                                         f"global correlation- mean differentiation score per time point, {s_run['name']}",
                                                         df_time_list_pop_mean, pallete, color,
                                                         "avg motility differentiation score",
                                                         "avg intensity differentiation score", time_slots,
                                                         population_mean=True)

            # plot cell-wise correlation
            plot_correlation_of_population_time_division(save_dir,
                                                         f"global correlation- differentiation score per time point, {s_run['name']}",
                                                         df_time_list, pallete, color, "motility differentiation score",
                                                         "intensity differentiation score", time_slots,
                                                         population_mean=False)


if __name__ == '__main__':
    diff_window = [140, 170]
    tracks_len = 30
    con_windows = [[0, 30], [40, 70], [90, 120], [140, 170], [180, 210], [220, 250]]
    sruns_dict = {1: consts.s1, 2: consts.s2, 3: consts.s3, 5: consts.s5}
    s_runs = [consts.s3, consts.s2, consts.s5, consts.s1]
    path = consts.local_path

    local_density = False
    to_run = "intensity"
    for con_train_n, diff_train_n, con_test_n, diff_test_n in [(1, 5, 2, 3), (2, 3, 1, 5), ]:  # (1, 5, 2, 3),
        # dir_path_score = get_dir_path(to_run)
        # for s_run in s_runs:
        #     print(s_run)
        dir_path_score = get_dir_path("motility")
        df_mot_con, _, _ = load_correlations_data(sruns_dict[con_test_n], dir_path_score)
        df_mot_diff, _, _ = load_correlations_data(sruns_dict[diff_test_n], dir_path_score)

        dir_path_score = get_dir_path("intensity")
        df_merge_col_int_con, _, _ = load_correlations_data(sruns_dict[con_test_n], dir_path_score)
        df_merge_col_int_diff, _, _ = load_correlations_data(sruns_dict[diff_test_n], dir_path_score)

        global_correlation_mean_plots(
            f"mot_int_correlations/train- {con_train_n},{diff_train_n}, test- {con_test_n}, {diff_test_n} ",
            df_mot_con, df_merge_col_int_con, df_mot_diff,
            df_merge_col_int_diff, con_test_n, diff_test_n, pct_change=False, binned_score=False,
            time_division=False)

        # global_correlation(df_merge_col_mot, df_merge_col_int, s_run,
        #                    f"mot_int_correlations/train- {con_train_n},{diff_train_n} ", False, False,
        #                    time_division=True, pearson=False)
