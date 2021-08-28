import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from load_tracks_xml import load_tracks_xml
from ts_fresh import drop_columns, normalize_tracks, get_prob_over_track, get_path, load_data


def calc_prob_delta(window, tracks, clf, X_test, motility, intensity, wt_cols, moving_window=False,
                    aggregate_windows=False, calc_delta=True):
    df_prob = pd.DataFrame(columns=wt_cols)

    for ind, t in enumerate(tracks):
        track = tracks[ind]
        if len(track) < window:
            continue

        # track = track[:10]

        # normalize track:
        track = drop_columns(track, motility=motility, intensity=intensity)
        track = normalize_tracks(track, motility=motility, intensity=intensity)

        # calculate list of probabilities per window
        true_prob = get_prob_over_track(clf, track, window, X_test, moving_window, aggregate_windows)
        if calc_delta:
            # calculate the difference in probability
            prob = [(true_prob[i] - true_prob[i - 1]) for i in range(1, len(true_prob))]
        else:
            prob = true_prob
        step_size = 1 if moving_window or aggregate_windows else window
        time_windows = [(track.iloc[val]['t']) for val in range(0 + window, len(track), step_size)]
        dic = {}
        for (wt, prob) in zip(time_windows, prob):
            dic[int(wt)] = prob
        data = [dic]
        df_prob = df_prob.append(data, ignore_index=True, sort=False)

    return df_prob


def plot_avg_change_diff_prob(diff_video_num, con_video_num, end_of_file_name, dir_name):
    windows = [40, 80, 120, 160]
    avg_vals_diff = []
    avg_vals_con = []
    window_times = []
    for w in windows:
        wt_c = [wt * 90 for wt in range(0, 950, w)]
        window_times.append(np.array(wt_c) / 3600)
        df_delta_prob_diff = pickle.load(
            open(dir_name + "/" + f"df_prob_w={w}, video_num={diff_video_num}" + end_of_file_name, 'rb'))
        df_delta_prob_con = pickle.load(
            open(dir_name + "/" + f"df_prob_w={w}, video_num={con_video_num}" + end_of_file_name, 'rb'))

        avg_vals_diff.append([df_delta_prob_diff[col].mean() for col in wt_c])
        avg_vals_con.append([df_delta_prob_con[col].mean() for col in wt_c])

    fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(8, 4), dpi=140)
    title = f"The AVG change in diff probability (intensity= {intensity}, motility={motility}, video #{con_video_num},{diff_video_num})"
    plt.suptitle(title, wrap=True)

    ax[0, 0].plot(window_times[0], avg_vals_con[0])
    ax[0, 0].plot(window_times[0], avg_vals_diff[0])
    ax[0, 0].set_title(f"window size={windows[0]}")

    ax[0, 1].plot(window_times[1], avg_vals_con[1])
    ax[0, 1].plot(window_times[1], avg_vals_diff[1])
    ax[0, 1].set_title(f"window size={windows[1]}")

    ax[1, 0].plot(window_times[2], avg_vals_con[2])
    ax[1, 0].plot(window_times[2], avg_vals_diff[2])
    ax[1, 0].set_title(f"window size={windows[2]}")

    ax[1, 1].plot(window_times[3], avg_vals_con[3])
    ax[1, 1].plot(window_times[3], avg_vals_diff[3])
    ax[1, 1].set_title(f"window size={windows[3]}")

    fig.legend(['control', 'ERK'], loc="lower left")
    for ax in fig.get_axes():
        ax.set_xlabel('Time [h]')
        ax.set_ylabel(' Avg p delta')
        ax.label_outer()
    plt.show()
    plt.savefig(
        dir_name + "/" + f"avg change of diff probability, video #{con_video_num},{diff_video_num}, {end_of_file_name}.png")
    plt.close(fig)


def get_df_delta_sums(diff_video_num, con_video_num, end_of_file_name):
    windows = [40, 80, 120, 160]

    df_delta_sums = pd.DataFrame()

    for w in windows:
        df_delta_prob_diff = pickle.load(
            open(dir_name + "/" + f"df_prob_w={w}, video_num={diff_video_num}" + end_of_file_name, 'rb'))
        df_delta_prob_con = pickle.load(
            open(dir_name + "/" + f"df_prob_w={w}, video_num={con_video_num}" + end_of_file_name, 'rb'))

        df_delta_sums = pd.concat([df_delta_sums, pd.DataFrame(
            {"delta_sum": df_delta_prob_con.sum(axis=1), "window_size": w, "diff_con": "control"})])
        df_delta_sums = pd.concat([df_delta_sums, pd.DataFrame(
            {"delta_sum": df_delta_prob_diff.sum(axis=1), "window_size": w, "diff_con": "ERK"})])
    return df_delta_sums


def plot_delta_distribution(diff_video_num, con_video_num, end_of_file_name, dir_name):
    df_delta_sums = get_df_delta_sums(diff_video_num, con_video_num, end_of_file_name)
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.violinplot(x="window_size", y="delta_sum", split=True, hue="diff_con", data=df_delta_sums, inner="quartile",
                   palette="bright")
    plt.title("Distribution of sum of the change in the probability to be differentiated")
    plt.ylabel("sum of delta")
    plt.xlabel("time window size")
    plt.legend()
    plt.show()
    plt.savefig(
        dir_name + "/" + f"distribution of sum of delta of the probability to be differentiated, video #{con_video_num},{diff_video_num}, {end_of_file_name}.png")

    # second plot
    sns.set_theme(style="darkgrid")
    # Plot the responses for different events and regions
    sns.lineplot(x="window_size", y="delta_sum", hue="diff_con", style="event", data=df_delta_sums)
    plt.title("Distribution of sum of the change in the probability to be differentiated")
    plt.ylabel("sum of delta")
    plt.xlabel("time window size")
    plt.legend()
    plt.show()
    plt.savefig(
        dir_name + "/" + f"line plot distribution of sum of delta of the probability to be differentiated, video #{con_video_num},{diff_video_num}, {end_of_file_name}.png")


if __name__ == '__main__':
    print(
        "Let's go! In this script, we will calculate the average delta of the probability of being differentiated over time")

    # params
    motility = False
    intensity = True
    video_num = 3
    window = 120
    wt_cols = [wt * 90 for wt in range(0, 950, window)]

    # print(f"window size is {window} frames, video number is{video_num}")

    # open a new directory to save the outputs in
    dir_name = f"manual_tracking_1,3_ motility-{motility}_intensity-{intensity}"
    print(dir_name)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    # # load ERK's tracks and dataframe
    # xml_path = get_path(
    #     fr"data/tracks_xml/manual_tracking/Experiment1_w1Widefield550_s{video_num}_all_manual_tracking.xml")
    # tracks, df = load_tracks_xml(xml_path)
    #
    # clf, X_train, X_test, y_train, y_test = load_data(dir_name)
    #
    # df_delta_prob = calc_prob_delta(window, tracks, clf, X_test, motility, intensity, wt_cols, aggregate_windows=True)
    # pickle.dump(df_delta_prob,
    #             open(dir_name + "/" + f"df_prob_w={window}, video_num={video_num} aggregate windows", 'wb'))

    plot_delta_distribution(diff_video_num=3, con_video_num=1, end_of_file_name=" aggregate windows", dir_name=dir_name)
    plot_avg_change_diff_prob(diff_video_num=3, con_video_num=1, end_of_file_name=" aggregate windows", dir_name=dir_name)
