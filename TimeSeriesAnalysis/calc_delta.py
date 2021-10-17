import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from DataPreprocessing.load_tracks_xml import load_tracks_xml
from TimeSeriesAnalysis.ts_fresh import drop_columns, normalize_tracks, get_prob_over_track, get_path, load_data, \
    plot_sampled_cells
from multiprocessing import Process

pd.options.mode.chained_assignment = None


def calc_prob_delta(window, tracks, clf, X_test, motility, intensity, wt_cols, moving_window=False,
                    aggregate_windows=False, calc_delta=True):
    df_prob = pd.DataFrame(columns=wt_cols)

    for ind, t in enumerate(tracks):
        track = tracks[ind]
        if window == 926:
            _window = len(track)
        elif len(track) < window:
            continue
        else:
            _window = window

        step_size = 1 if moving_window or aggregate_windows else window
        time_windows = [(track.iloc[val]['t']) for val in range(0 + window, len(track), step_size)]

        # track = track[:10]

        # normalize track:
        track = drop_columns(track, motility=motility, intensity=intensity)
        track = normalize_tracks(track, motility=motility, intensity=intensity)

        # calculate list of probabilities per window
        true_prob = get_prob_over_track(clf, track, _window, X_test, moving_window, aggregate_windows)
        if calc_delta:
            # calculate the difference in probability
            prob = [(true_prob[i] - true_prob[i - 1]) for i in range(1, len(true_prob))]
            time_windows = time_windows[1:]
        else:
            prob = true_prob

        dic = {}
        for (wt, prob) in zip(time_windows, prob):
            dic[int(wt)] = prob
        data = [dic]
        df_prob = df_prob.append(data, ignore_index=True, sort=False)

    return df_prob


def plot_avg_diff_prob(diff_video_num, con_video_num, end_of_file_name, dir_name, title):
    # windows = [40, 80, 120, 160]
    # windows = [80, 90, 100]
    windows = [30, 30, 30]
    df = pd.DataFrame()
    avg_vals_diff = []
    std_vals_diff = []
    avg_vals_con = []
    std_vals_con = []
    window_times = []
    for w in windows:
        wt_c = [wt * 1 for wt in range(0, 350, w)]
        window_times.append(np.array(wt_c) / 3600 * 300)
        df_delta_prob_diff = pickle.load(
            open(dir_name + "/" + f"df_prob_w={w}, video_num={diff_video_num}" + end_of_file_name, 'rb'))
        df_delta_prob_con = pickle.load(
            open(dir_name + "/" + f"df_prob_w={w}, video_num={con_video_num}" + end_of_file_name, 'rb'))

        avg_vals_diff.append([df_delta_prob_diff[col].mean() for col in wt_c])
        std_vals_diff.append([df_delta_prob_diff[col].std() for col in wt_c])
        avg_vals_con.append([df_delta_prob_con[col].mean() for col in wt_c])
        std_vals_con.append([df_delta_prob_con[col].std() for col in wt_c])

    # plot it!
    fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(8, 4), dpi=140)

    def plot_window(ax, window_times, avg_vals_con, avg_vals_diff, ax_0, ax_1, std_vals_con, std_vals_diff, w):
        ax[ax_0, ax_1].plot(window_times, avg_vals_con)
        ax[ax_0, ax_1].plot(window_times, avg_vals_diff)
        ax[ax_0, ax_1].set_title(f"window size={windows[w]}")

        p_std = np.asarray(avg_vals_con) + np.asarray(std_vals_con)
        m_std = np.asarray(avg_vals_con) - np.asarray(std_vals_con)
        ax[ax_0, ax_1].fill_between(window_times, m_std, p_std, alpha=0.5)

        p_std = np.asarray(avg_vals_diff) + np.asarray(std_vals_diff)
        m_std = np.asarray(avg_vals_diff) - np.asarray(std_vals_diff)
        ax[ax_0, ax_1].fill_between(window_times, m_std, p_std, alpha=0.5)
        ax[ax_0, ax_1].grid()

    # plot_window(ax, window_times[0], avg_vals_con[0], avg_vals_diff[0], 0, 0, std_vals_con[0], std_vals_diff[0], 0)
    # plot_window(ax, window_times[1], avg_vals_con[1], avg_vals_diff[1], 0, 1, std_vals_con[1], std_vals_diff[1], 1)
    # plot_window(ax, window_times[2], avg_vals_con[2], avg_vals_diff[2], 1, 0, std_vals_con[2], std_vals_diff[2], 2)
    #
    # plt.suptitle(title, wrap=True)
    # fig.legend(['control', 'ERK'], loc="lower left")
    # for ax in fig.get_axes():
    #     ax.set_xlabel('Time [h]')
    #     ax.set_ylabel(' Avg p delta')
    #     ax.label_outer()
    # plt.show()
    # plt.savefig(
    #     dir_name + "/" + f"{title}.png")
    # plt.close(fig)

    fig, ax = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(8, 4), dpi=140)
    ax.plot(window_times[1], avg_vals_con[1])
    ax.plot(window_times[1], avg_vals_diff[1])
    ax.set_title(f"window size={windows[1]}")

    p_std = np.asarray(avg_vals_con[1]) + np.asarray(std_vals_con[1])
    m_std = np.asarray(avg_vals_con[1]) - np.asarray(std_vals_con[1])
    ax.fill_between(window_times[1], m_std, p_std, alpha=0.5)

    p_std = np.asarray(avg_vals_diff[1]) + np.asarray(std_vals_diff[1])
    m_std = np.asarray(avg_vals_diff[1]) - np.asarray(std_vals_diff[1])
    ax.fill_between(window_times[1], m_std, p_std, alpha=0.5)
    ax.grid()

    plt.suptitle(title, wrap=True)
    fig.legend(['control', 'ERK'], loc="lower left")

    for ax in fig.get_axes():
        ax.set_xlabel('Time [h]')
        ax.set_ylabel(' Avg p delta')
        ax.label_outer()
    plt.show()
    plt.savefig(
        dir_name + "/" + f"{title} 30.png")
    plt.close(fig)


def get_df_delta_sums(diff_video_num, con_video_num, end_of_file_name, dir_name, sum):
    # windows = [40, 80, 120, 160]
    # windows = [80, 90, 100]
    windows = [20, 30, 40]

    df_delta_sums = pd.DataFrame()

    for w in windows:
        df_delta_prob_diff = pickle.load(
            open(dir_name + "/" + f"df_prob_w={w}, video_num={diff_video_num}" + end_of_file_name, 'rb'))
        df_delta_prob_con = pickle.load(
            open(dir_name + "/" + f"df_prob_w={w}, video_num={con_video_num}" + end_of_file_name, 'rb'))

        if sum:
            con = df_delta_prob_con.sum(axis=1)
            diff = df_delta_prob_diff.sum(axis=1)
        else:
            con = df_delta_prob_con.mean(axis=1)
            diff = df_delta_prob_diff.mean(axis=1)
        df_delta_sums = pd.concat([df_delta_sums, pd.DataFrame(
            {"delta_sum": con, "window_size": w, "diff_con": "control"})])
        df_delta_sums = pd.concat([df_delta_sums, pd.DataFrame(
            {"delta_sum": diff, "window_size": w, "diff_con": "ERK"})])
    return df_delta_sums


def plot_distribution(x_str, y_str, hue_str, data, save_fig_path, title):
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.violinplot(x=x_str, y=y_str, split=True, hue=hue_str, data=data, inner="quartile",
                   palette="bright")
    plt.title(title)
    plt.ylabel("sum of delta")
    plt.xlabel("time window size")
    plt.legend()
    plt.show()
    plt.savefig(save_fig_path)


def plot_delta_sum_distribution(diff_video_num, con_video_num, end_of_file_name, dir_name):
    df = get_df_delta_sums(diff_video_num, con_video_num, end_of_file_name, dir_name, True)
    save_fig_path = dir_name + "/" + f"distribution of sum of delta of the probability to be differentiated, video #{con_video_num},{diff_video_num}, {end_of_file_name}.png"
    title = "Distribution of sum of the change in the probability to be differentiated"
    plot_distribution(x_str="window_size", y_str="delta_sum", hue_str="diff_con", data=df, save_fig_path=save_fig_path,
                      title=title)


def plot_delta_mean_distribution(diff_video_num, con_video_num, end_of_file_name, dir_name):
    df = get_df_delta_sums(diff_video_num, con_video_num, end_of_file_name, dir_name, False)
    save_fig_path = dir_name + "/" + f"distribution of mean of delta of the probability to be differentiated, video #{con_video_num},{diff_video_num}, {end_of_file_name}.png"
    title = "Distribution of mean of the change in the probability to be differentiated"
    plot_distribution(x_str="window_size", y_str="delta_sum", hue_str="diff_con", data=df,
                      save_fig_path=save_fig_path,
                      title=title)


def run_calc(dir_name, tracks, video_num, wt_cols, motility, intensity):
    print(dir_name)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    clf, X_train, X_test, y_train, y_test = load_data(dir_name)

    if not os.path.exists(dir_name + "/" + f"df_prob_w={window}, video_num={video_num}"):
        df_prob = calc_prob_delta(window, tracks, clf, X_test, motility, intensity, wt_cols, calc_delta=False)
        pickle.dump(df_prob,
                    open(dir_name + "/" + f"df_prob_w={window}, video_num={video_num}", 'wb'))

        if not os.path.exists(dir_name + "/" + f"df_prob_w={window}, video_num={video_num} delta"):
            df_delta = calc_prob_delta(window, tracks, clf, X_test, motility, intensity, wt_cols, calc_delta=True)
            pickle.dump(df_delta,
                        open(dir_name + "/" + f"df_prob_w={window}, video_num={video_num} delta", 'wb'))


def run_plot_delta(dir_name, intensity, motility, diff_vid_num, con_vid_num):
    title = f"Averaged diff probability (intensity= {intensity}, motility={motility}, video #{con_vid_num},{diff_vid_num})"
    plot_avg_diff_prob(diff_video_num=diff_vid_num, con_video_num=con_vid_num, end_of_file_name="",
                       dir_name=dir_name, title=title)

    title = f"avg change of diff probability (intensity= {intensity}, motility={motility}, video #{con_vid_num},{diff_vid_num})"
    plot_avg_diff_prob(diff_video_num=diff_vid_num, con_video_num=con_vid_num, end_of_file_name=" delta",
                       dir_name=dir_name, title=title)

    plot_delta_mean_distribution(diff_video_num=diff_vid_num, con_video_num=con_vid_num, end_of_file_name=" delta",
                                 dir_name=dir_name)
    plot_delta_sum_distribution(diff_video_num=diff_vid_num, con_video_num=con_vid_num, end_of_file_name=" delta",
                                dir_name=dir_name)


def run_delta(diff_t_windows, video_num, window, motility, intensity, wt_cols, video_diff_num, video_con_num):
    print(f"processing window #{window}")
    # load tracks and dataframe
    # xml_path = get_path(
    #     fr"../data/tracks_xml/260721/S{video_num}_Nuclei.xml")
    # tracks, df = load_tracks_xml(xml_path)

    for diff_t_w in diff_t_windows:
        time_frame = f"{diff_t_w[0]},{diff_t_w[1]} frames ERK, {0},{30} frames con"

        # open a new directory to save the outputs in
        dir_name = f"outputs/_210726_ motility-{motility}_intensity-{intensity}/{time_frame}"

        # p = Process(target=run_calc, args=(dir_name, tracks, video_num, wt_cols, motility, intensity))
        # p.start()
        # run_calc(dir_name, tracks, video_num, wt_cols, motility, intensity)

        # p = Process(target=run_plot_delta, args=(dir_name, intensity, motility))
        # p.start()
        run_plot_delta(dir_name, intensity, motility, video_diff_num, video_con_num)


if __name__ == '__main__':

    print(
        "Let's go! In this script, we will calculate the average delta of the probability of being differentiated over time")

    # params

    motility = False
    intensity = True
    video_diff_num = 8
    video_con_num = 3

    diff_t_windows = [[140, 170]]
    for window in [20]:
        wt_cols = [wt * 300 for wt in range(0, 350, window)]

        run_delta(diff_t_windows, video_con_num, window, motility, intensity, wt_cols, video_diff_num, video_con_num)
        # run_delta(diff_t_windows, video_diff_num, window, motility, intensity, wt_cols, video_diff_num, video_con_num)

