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


def calc_correlation_score(df, tagged_df, all_tracks, dir_path, s_name):
    second_dir = s_name + "total correlations"
    utils.open_dirs(dir_path, second_dir)
    save_dir = dir_path + "/" + second_dir

    for label, track in df.groupby("Spot track ID"):
        plt.figure(figsize=(14, 9))
        plt.title(label)
        ax1 = plt.subplot(2, 3, 1)
        ax2 = plt.subplot(2, 3, 2)
        ax3 = plt.subplot(2, 3, 3)
        ax4 = plt.subplot(2, 1, 2)
        axes = [ax1, ax2, ax3, ax4]

        # conf = track.drop(columns=["Spot track ID", "t0", "cos_theta", "Track #"]).values.tolist()
        conf = track.copy().iloc[0, :].drop(columns=["Spot track ID", "Track #", "t0", "cos_theta"])
        coord = list(track["cos_theta"].values)
        local_den_df = utils.add_features(tagged_df[tagged_df["Spot track ID"] == label], local_density=True,
                                          df_s=all_tracks)
        local_den_df = local_den_df.sort_values("Spot frame")
        local_density = local_den_df["local density"].values.tolist()

        length = min([len(conf), len(list(coord[0])), len(local_density)])
        tmp_df = pd.DataFrame({
            'score': conf[:length],
            'coordination': coord[0][:length],
            'local density': local_density[:length]
        })
        tmp_df["time"] = tmp_df.index * 5 / 60
        tmp_df = tmp_df.dropna()

        tmp_df["score:coordination"] = tmp_df["score"] / tmp_df["coordination"]
        tmp_df["score:local density"] = tmp_df["score"] / tmp_df["local density"]
        tmp_df["coordination:local density"] = tmp_df["coordination"] / tmp_df["local density"]

        # score over time & coordination
        sns.lineplot(ax=ax1, x="score:coordination", y="time", palette="crest", data=tmp_df, legend="brief")
        ax1.set_title("score:coordination")

        # coordination over density + time
        sns.scatterplot(ax=ax2, x="time", y="coordination", palette="crest", hue="local density",
                        data=tmp_df, legend="brief")
        ax2.set_title("Coordination rate over time & local density")

        # score over density + time
        sns.scatterplot(ax=ax3, x="local density", y="score", palette="crest", hue="time", data=tmp_df,
                        legend="brief")
        ax3.set_title("Differentiation score over local density & time")

        ax4.set_title("Differentiation score over time")
        ax4.plot(tmp_df["time"], tmp_df["score"])
        ax4.set_xlabel("time (h)")
        ax4.set_ylabel("score")
        ax4.grid()

        plt.suptitle(f"track #{label}")
        plt.savefig(save_dir + f"/correlations_{label}.png")
        plt.show()


def get_correlations_df(track, tagged_df, label, all_tracks):
    conf = track.copy().iloc[0, :].drop(columns=["Spot track ID", "Track #", "t0", "cos_theta"])

    try:
        coord = [np.nan for i in range(int(track["t0"].max()))]
        coord.extend(list(track["cos_theta"].values)[0][0])
    except:
        return pd.DataFrame()

    local_den_df = utils.add_features(tagged_df[tagged_df["Spot track ID"] == label], local_density=True,
                                      df_s=all_tracks)
    local_den_df = local_den_df.sort_values("Spot frame")
    local_density = [np.nan for i in range(int(track["t0"].max()))]
    local_density.extend(local_den_df["local density"].values.tolist())
    local_density = [0 for i in range(len(coord))]  # TODO remove

    length = min([len(conf), len(coord), len(local_density)])
    tmp_df = pd.DataFrame({
        'score': conf[:length],
        'coordination': coord[:length],
        'local density': local_density[:length],
        'Spot frame': [i for i in range(length)]
    })

    # calc speed
    track_data = tagged_df[tagged_df["Spot track ID"] == label][
        ["Spot frame", "Spot position X (µm)", "Spot position Y (µm)"]]
    track_data = track_data.sort_values("Spot frame")
    track_data["Speed X"] = track_data["Spot position X (µm)"].diff()
    track_data["Speed Y"] = track_data["Spot position Y (µm)"].diff()
    track_data["Speed"] = np.sqrt(np.square(track_data["Speed X"]) + np.square(track_data["Speed Y"]))
    tmp_df = pd.merge(tmp_df, track_data[["Speed", "Spot frame"]], on='Spot frame')

    tmp_df["rolling_coordination"] = tmp_df["coordination"].rolling(window=30).mean()
    tmp_df["rolling_local_density"] = tmp_df["local density"].rolling(window=30).mean()
    tmp_df["rolling_speed"] = tmp_df["Speed"].rolling(window=30).mean()

    tmp_df["time"] = tmp_df["Spot frame"] * 5 / 60
    tmp_df["Spot track ID"] = track["Spot track ID"].values[0]
    tmp_df = tmp_df.dropna()

    tmp_df["score_coordination"] = tmp_df["score"] / tmp_df["rolling_coordination"]
    tmp_df["score_local density"] = tmp_df["score"] / tmp_df["rolling_local_density"]
    tmp_df["coordination_local density"] = tmp_df["rolling_coordination"] / tmp_df[
        "rolling_local_density"]
    tmp_df["coordination_speed"] = tmp_df["coordination"] / tmp_df["rolling_speed"]
    tmp_df["score_speed"] = tmp_df["score"] / tmp_df["rolling_speed"]

    r_window_size = 30
    df_interpolated = tmp_df.interpolate()
    df_interpolated.index = df_interpolated["time"]
    # Compute rolling window synchrony
    tmp_df["p_coordination_speed"] = df_interpolated['rolling_coordination'].rolling(window=r_window_size,
                                                                                     center=True).corr(
        df_interpolated['rolling_speed']).values
    tmp_df["p_score_speed"] = df_interpolated['score'].rolling(window=r_window_size, center=True).corr(
        df_interpolated['rolling_speed']).values
    tmp_df["p_score_coordination"] = df_interpolated['score'].rolling(window=r_window_size, center=True).corr(
        df_interpolated['coordination']).values
    tmp_df["p_score_local_density"] = df_interpolated['score'].rolling(window=r_window_size, center=True).corr(
        df_interpolated['rolling_local_density']).values
    tmp_df["p_coordination_local_density"] = df_interpolated['rolling_coordination'].rolling(window=r_window_size,
                                                                                             center=True).corr(
        df_interpolated['rolling_local_density']).values

    return tmp_df


def correlations_percentage(df, tagged_df, all_tracks, dir_path, s_name):
    second_dir = s_name + "_percentage"
    save_dir = dir_path + "/" + second_dir
    all_correlations_df = pd.DataFrame()
    for label, track in df.groupby("Spot track ID"):
        if len(track.iloc[0]) < 60:
            continue
        correlations_df = get_correlations_df(track, tagged_df, label, all_tracks)
        all_correlations_df = all_correlations_df.append(correlations_df, ignore_index=True)
    pickle.dump(all_correlations_df,
                open(save_dir, 'wb'))


def correlations_single_cell(df, tagged_df, all_tracks, dir_path, s_name):
    second_dir = s_name
    utils.open_dirs(dir_path, second_dir)
    save_dir = dir_path + "/" + second_dir
    for label, track in df.groupby("Spot track ID"):
        if len(track.iloc[0]) < 60:
            continue

        correlations_df = get_correlations_df(track, tagged_df, label, all_tracks)
        sns.scatterplot(x="time", y="score", palette="crest", color="darkorange",
                        data=correlations_df, s=25, label="ERK inhibition treated single cell",
                        legend="brief").set(ylim=(0, 1))
        plt.plot(correlations_df["time"], correlations_df["score"], "--", linewidth=2, color="orange")
        plt.xlim(2, 22)
        plt.ylim(0, 1)
        plt.title("Single Cell's Differentiation Score Over time")
        plt.grid()
        plt.show()

        # plt.figure(figsize=(14, 9))
        # plt.title(label)
        # ax1 = plt.subplot(2, 3, 1)
        # ax2 = plt.subplot(2, 3, 2)
        # ax3 = plt.subplot(2, 3, 3)
        # ax4 = plt.subplot(2, 2, 3)
        # ax5 = plt.subplot(2, 2, 4)

        #
        # # score over time & coordination
        # sns.scatterplot(ax=ax1, x="rolling_speed", y="score", palette="crest", hue="rolling_coordination",
        #                 data=correlations_df,
        #                 legend="brief").set(ylim=(0, 1), xlim=(0, 15))
        # ax1.set_title("Differentiation score over speed & coordination")
        # sns.scatterplot(ax=ax2, x="time", y="rolling_coordination", palette="flare", hue="rolling_local_density",
        #                 hue_norm=(10, 90),
        #                 data=correlations_df, legend="brief").set(ylim=(0, 1))
        # ax2.set_xlim(2, 22)
        # ax2.set_title("Coordination rate over time & local density")
        # # score over density + time
        # sns.scatterplot(ax=ax3, x="rolling_local_density", y="score", palette="coolwarm", hue="time", hue_norm=(0, 22),
        #                 data=correlations_df,
        #                 legend="brief").set(ylim=(0, 1))
        # ax3.set_xlim(10, 90)
        # ax3.set_title("Differentiation score over local density & time")
        # # score over time & coordination
        # # sns.regplot(ax=ax4, x="time", y="score", data=correlations_df.astype(float), ci=None)
        # sns.scatterplot(ax=ax4, x="time", y="score", palette="crest", hue="rolling_coordination", hue_norm=(0, 1),
        #                 data=correlations_df,
        #                 legend="brief").set(ylim=(0, 1))
        # ax4.plot(correlations_df["time"], correlations_df["score"], "--", linewidth=1)
        # ax4.set_xlim(2, 22)
        # ax4.set_ylim(0, 1)
        # ax4.set_title("Differentiation score over time & coordination")
        # ax4.grid()
        # r_window_size = 30
        # # Interpolate missing data.
        # df_interpolated = correlations_df.interpolate()
        # df_interpolated.index = df_interpolated["time"]
        # # Compute rolling window synchrony
        # rolling_r_coord_speed = df_interpolated['rolling_coordination'].rolling(window=r_window_size, center=True).corr(
        #     df_interpolated['rolling_speed'])
        # rolling_r_score_speed = df_interpolated['score'].rolling(window=r_window_size, center=True).corr(
        #     df_interpolated['rolling_speed'])
        # rolling_r_score_coord = df_interpolated['score'].rolling(window=r_window_size, center=True).corr(
        #     df_interpolated['rolling_coordination'])
        # rolling_r_score_local_den = df_interpolated['score'].rolling(window=r_window_size, center=True).corr(
        #     df_interpolated['rolling_local_density'])
        # rolling_r_coord_local_den = df_interpolated['rolling_coordination'].rolling(window=r_window_size, center=True).corr(
        #     df_interpolated['rolling_local_density'])
        # rolling_r_coord_speed.plot(ax=ax5, label="coordination:speed")
        # rolling_r_score_speed.plot(ax=ax5, label="score:speed")
        # rolling_r_score_coord.plot(ax=ax5, label="score:coordination")
        # rolling_r_score_local_den.plot(ax=ax5, label="score:local density")
        # rolling_r_coord_local_den.plot(ax=ax5, label="coordination:local density")
        # ax5.set(xlabel='time (h)', ylabel='Pearson r', ylim=(-1, 1))
        # ax5.set(title=f"rolling pearson correlation (window size- {r_window_size}) ")
        # ax5.legend()
        # ax5.grid()
        #
        # plt.suptitle(f"track #{label}")
        # plt.savefig(save_dir + f"/correlations_{label}.png")
        # plt.show()

        # sns.heatmap(track[["score", "coordination", "Speed", "local density"]].pct_change().corr(), annot=True)
        # plt.show()


def get_tracks_longer_than_n(df, n):
    new_df = pd.DataFrame()
    for label, label_df in df.groupby("Spot track ID"):
        if len(label_df) > n:
            new_df = new_df.append(label_df)
    return new_df


def plot_ridgeplot(df, correlation_name, s_num, pal):
    df["time_binned"] = df["time"] // 1
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Initialize the FacetGrid object
    g = sns.FacetGrid(df, row="time_binned", hue="time_binned", aspect=15, height=.5, palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, correlation_name, bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, correlation_name, clip_on=False, color="w", lw=2, bw_adjust=.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, correlation_name)
    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    plt.savefig(dir_path_score + "/correlations_percentage/" + f"S{s_num} ridgline plot {correlation_name}.png")
    plt.show()
    plt.clf()
    plt.close()


def plot_time_heat_map(df, correlation_name, s_num, pal):
    # TODO implement here
    plt.savefig(dir_path_score + "/correlations_percentage/" + f"S{s_num} ridgline plot {correlation_name}.png")
    plt.show()


def plot_coor_coeff(df, s_num, pal, split, hue):
    corr_score_coord = []
    corr_score_speed = []
    corr_score_local_den = []
    corr_coord_speed = []
    corr_coord_local_den = []
    targets = []

    for track_id, track in df.groupby("Spot track ID"):
        # track = track.sort_values("Spot frame")

        score_ret = track["score"].pct_change()
        coordination_ret = track["rolling_coord"].pct_change()
        speed_ret = track["rolling_speed"].pct_change()
        local_density_ret = track["rolling_local_density"].pct_change()

        corr_score_coord.append(score_ret.corr(coordination_ret))
        corr_score_speed.append(score_ret.corr(speed_ret))
        corr_score_local_den.append(score_ret.corr(local_density_ret))
        corr_coord_speed.append(coordination_ret.corr(speed_ret))
        corr_coord_local_den.append(coordination_ret.corr(local_density_ret))
        targets.append(track["target"].max())

    sns.set_theme(style="whitegrid")
    iris = pd.concat([pd.DataFrame({"value": corr_score_coord, "measurement": "score_coord", "target": targets}),
                      # pd.DataFrame({"value": corr_score_speed, "measurement": "score_speed", "target": targets}),
                      # pd.DataFrame(
                      #     {"value": corr_score_local_den, "measurement": "score_local_den", "target": targets}),
                      # pd.DataFrame({"value": corr_coord_speed, "measurement": "coord_speed", "target": targets}),
                      # pd.DataFrame({"value": corr_coord_local_den, "measurement": "coord_local_den", "target": targets})
                      ])

    sns.displot(data=iris, x="value", hue=hue, kind="kde")

    plt.savefig(dir_path_score + "/correlations_percentage/" + f"S{s_num} correlation violin.png")
    plt.show()
    plt.clf()
    plt.close()

    sns.violinplot(data=iris, x="measurement", y="value", palette=pal, inner="quartile", orient="v", hue=hue,
                   split=split) \
        .set(ylim=(-1, 1))
    plt.savefig(dir_path_score + "/correlations_percentage/" + f"S{s_num} correlation violin.png")
    plt.show()
    plt.clf()
    plt.close()

    print()


def plot_mean_correlation(correlation_p_df_diff, correlation_p_df_con, correlation_name, s_num2, s_num1,
                          dir_path_score):
    def get_std_mean(df):
        # df.groupby("time")[correlation_name].apply(np.mean).reset_index()

        _df = pd.DataFrame()
        _df["mean"] = df.groupby("time")[correlation_name].apply(np.mean)
        _df["std"] = df.groupby("time")[correlation_name].apply(np.std)

        return _df

    df_con = get_std_mean(correlation_p_df_con)
    df_diff = get_std_mean(correlation_p_df_diff)

    plt.plot(df_con.index, df_con["mean"], label="Control")
    plt.fill_between(df_con.index, df_con["mean"] + df_con["std"], df_con["mean"] - df_con["std"], alpha=0.5)
    plt.plot(df_diff.index, df_diff["mean"], label="Erk")
    plt.fill_between(df_diff.index, df_diff["mean"] + df_diff["std"], df_diff["mean"] - df_diff["std"], alpha=0.5)
    plt.title(f"mean & std of {correlation_name}, s{s_num2}, s{s_num1}")
    plt.xlabel("correlation value")
    plt.ylabel("time (h)")
    plt.legend()
    plt.savefig(
        dir_path_score + "/correlations_percentage/" + f"mean & std of {correlation_name}, s{s_num2}, s{s_num1}.png")

    plt.show()
    plt.clf()
    plt.close()


def plot_correlations_percentage(dir_path_score, s_num1, s_num2):
    print("plotting plot_correlations_percentage")

    # df_merge_diff_mot = pickle.load(open(get_dir_path("motility") + f"/S{s_num2}" + "_percentage", 'rb'))
    # df_merge_diff_int = pickle.load(open(get_dir_path("intensity") + f"/S{s_num2}" + "_percentage", 'rb'))

    correlation_p_df_con = pickle.load(open(dir_path_score + f"/S{s_num1}" + "_percentage", 'rb'))
    correlation_p_df_diff = pickle.load(open(dir_path_score + f"/S{s_num2}" + "_percentage", 'rb'))
    correlation_p_df_diff["target"] = True
    correlation_p_df_con["target"] = False

    utils.open_dirs(dir_path_score, "correlations_percentage")

    con_pal = sns.cubehelix_palette(22, rot=-.25, light=.7)
    diff_pal = sns.color_palette("Oranges", n_colors=22, )

    # plot_coor_coeff(correlation_p_df_diff, s_num2, 'light:orange', hue="measurement", split=False)
    # plot_coor_coeff(correlation_p_df_con, s_num1, 'light:b', hue="measurement", split=False)
    correlation_p_df_con["target"] = correlation_p_df_con["target"].apply(lambda x: "ERK" if x == True else "Control")
    correlation_p_df_diff["target"] = correlation_p_df_diff["target"].apply(lambda x: "ERK" if x == True else "Control")
    plot_coor_coeff(pd.concat([correlation_p_df_con, correlation_p_df_diff], ignore_index=True),
                    str(s_num1) + str(s_num2), 'muted', hue="target", split=True)

    correlations = ["score_coordination", "score_local density", "coordination_local density",
                    "coordination_speed", "score_speed", "p_coordination_local_density", "p_score_local_density",
                    "p_score_speed", "p_coordination_speed", "p_score_coordination"]

    for correlation_name in correlations:
        # plot_mean_correlation(correlation_p_df_diff, correlation_p_df_con, correlation_name, s_num2, s_num1,
        #                       dir_path_score)

        # plot_ridgeplot(correlation_p_df_diff, correlation_name, s_num2, diff_pal)
        # plot_ridgeplot(correlation_p_df_con, correlation_name, s_num1, con_pal)
        pass


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
    df = pd.read_csv(path + s_run["csv_tagged_path"], encoding="ISO-8859-1")
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


def global_correlation(df_merge_col_mot, df_merge_col_int, s_run, save_dir, pct_change, binned_score,
                       time_devision=False, pearson=False):
    if binned_score:
        df_merge_col_mot = bin_score_values(df_merge_col_mot)
        df_merge_col_int = bin_score_values(df_merge_col_int)
    if pearson:
        pearsons = get_pearsons_r(df_merge_col_mot, df_merge_col_int, pct_change)

        color = "Orange" if s_run["name"] in ["S3", "S5"] else "Blue"
        sns.displot(data=pd.Series(pearsons), kde=True, color=color)
        plt.xlabel("pearson r")
        plt.title(
            f"distribution of single cell's correlation between \nactin intensity model & motility model, vid {s_run['name']}")
        plt.show()
    else:

        df_mot = df_merge_col_mot.drop(columns=["cos_theta", "t0", "Spot track ID"]).T
        df_int = df_merge_col_int.drop(columns=["cos_theta", "t0", "Spot track ID"]).T

        time_slots = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 250)]
        fig, axs = plt.subplots(1, len(time_slots), figsize=(18, 4), facecolor='w', edgecolor='k')

        for i, (t_start, t_end) in enumerate(time_slots):
            df_mot_t = df_mot.iloc[t_start:t_end].T.mean()
            df_int_t = df_int.iloc[t_start:t_end].T.mean()

            color = "Orange" if s_run["name"] in ["S3", "S5"] else "Blue"
            axs[i].scatter(df_mot_t, df_int_t, color=color)
            axs[i].set_title(f"time window: [{round(t_start * 5 / 60)}, {round(t_end * 5 / 60)}] (hours)")
            axs[i].set_xlabel("avg motility differentiation score")
            axs[i].set_ylabel("avg intensity differentiation score")
        plt.suptitle(f"global correlation- mean differentiation score per time point, {s_run['name']}")
        plt.show()



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


def plot_models_correlations(df_merge_col_mot, df_merge_col_int, s_run, pct_change, binned_score, save_dir):
    if binned_score:
        df_merge_col_mot = bin_score_values(df_merge_col_mot)
        df_merge_col_int = bin_score_values(df_merge_col_int)

    for ind, row in df_merge_col_mot.iterrows():
        track_id_mot = row["Spot track ID"]
        if track_id_mot == 3224:
            print()
        score_int, score_mot = get_single_cell_data(track_id_mot, df_merge_col_mot, df_merge_col_int, pct_change)
        if len(score_int) > 0:
            pearson_r = calc_pearson_r(score_int, score_mot)
            rolling_pearson_corr = score_int.iloc[:, 1].rolling(30).corr(score_mot.iloc[:, 1])

            plt.plot(range(len(rolling_pearson_corr)) + score_int['index'][0], rolling_pearson_corr,
                     label="rolling pearson", color="orange")
            plt.scatter(score_int['index'], score_int.iloc[:, 1], label="intensity score", color="wheat")
            plt.scatter(score_mot['index'], score_mot.iloc[:, 1], label="motility score", color="tab:orange")
            plt.xticks(np.arange(0, 260, step=30), np.arange(0, 260, step=30) * 5 / 60)
            plt.ylim((-1, 1))
            plt.title(f"cell #{track_id_mot}, pearson r={pearson_r[0]}")
            plt.xlabel("time (h)")
            plt.ylabel("differentiation score/ pearson r value")
            plt.legend()
            plt.grid()
            plt.savefig(save_dir + f"correlations_motility score and intensity score {s_run['name']}.png")
            plt.show()
            plt.clf()
            plt.close()


if __name__ == '__main__':
    diff_window = [140, 170]
    tracks_len = 30
    con_windows = [[0, 30], [40, 70], [90, 120], [140, 170], [180, 210], [220, 250]]

    s_runs = [consts.s5, consts.s1, consts.s3, consts.s2]
    path = consts.local_path

    local_density = False
    to_run = "intensity"
    for con_train_n, diff_train_n, con_test_n, diff_test_n in [(1, 5, 2, 3),(2, 3, 1, 5), ]:  # (1, 5, 2, 3),
        dir_path_score = get_dir_path(to_run)

        for s_run in s_runs:
            print(s_run)
            dir_path_score = get_dir_path("motility")
            df_merge_col_mot, df_tagged_mot, df_all_tracks_mot = load_correlations_data(s_run, dir_path_score)
            dir_path_score = get_dir_path("intensity")
            df_merge_col_int, df_tagged_int, df_all_tracks_int = load_correlations_data(s_run, dir_path_score)
            # plot_models_correlations(df_merge_col_mot, df_merge_col_int, s_run, pct_change=False, binned_score=False,
            #                          save_dir="mot_int_correlations/single cell correlations/train- {con_train_n},{diff_train_n} ")
            # global_correlation(df_merge_col_mot, df_merge_col_int, s_run, f"mot_int_correlations/binned_score/train- {con_train_n},{diff_train_n} ", False, False,
            #                    time_devision=False, pearson=False)
            # pearson_lag_models_correlation(df_merge_col_mot, df_merge_col_int, s_run,
            #                                f"mot_int_correlations/binned_score/train- {con_train_n},{diff_train_n} ",
            #                                pct_change=False, binned_score=False)

            # df_merge_col, df_tagged, df_all_tracks = load_correlations_data(s_run, dir_path_score)
            # correlations_single_cell(df_merge_col, df_tagged, df_all_tracks, dir_path_score, s_run["name"])
            # correlations_percentage(df_merge_col, df_tagged, df_all_tracks, dir_path_score, s_run["name"])

        plot_correlations_percentage(dir_path_score, con_test_n, diff_test_n)
        plot_correlations_percentage(dir_path_score, con_train_n, diff_train_n)
