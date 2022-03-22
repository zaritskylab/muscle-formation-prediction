import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from mastodon import get_prob_over_track, load_data
from mastodon import drop_columns, normalize_tracks
from mast_intensity import drop_columns as drop_col_int
from mast_intensity import normalize
pd.options.mode.chained_assignment = None


def calc_prob_delta(window, tracks, clf, X_test, motility, visual, wt_cols, moving_window=False,
                    aggregate_windows=False, calc_delta=True):
    wt_cols = [i for i in range(260)] if moving_window else wt_cols
    df_prob = pd.DataFrame(columns=wt_cols)

    # tracks = tracks[:5] #todo: remove

    for ind, t in enumerate(tracks):
        track = tracks[ind]
        if len(track) < window:
            continue
        else:
            _window = window

        step_size = 1 if moving_window or aggregate_windows else window

        time_windows = [(track.iloc[val]['Spot frame']) for val in range(0 + window, len(track), step_size)]
        time_windows.sort()
        track = track.sort_values("Spot frame")

        # track = track[:33]

        # normalize track:
        if motility:
            track = drop_columns(track, motility=motility, visual=visual)
            track = normalize_tracks(track, motility=motility, visual=visual)
        elif visual:
            track = drop_col_int(track)
            track = normalize(track)
        track.dropna(inplace=True)

        # calculate list of probabilities per window
        true_prob = get_prob_over_track(clf, track, _window, X_test, moving_window, aggregate_windows)
        if calc_delta:
            # calculate the difference in probability
            prob = [(true_prob[i] - true_prob[i - 1]) for i in range(1, len(true_prob))]
            time_windows = time_windows[1:]
        else:
            prob = true_prob

        # dic = {}
        # for (wt, prob) in zip(time_windows, prob):
        #     dic[int(wt)] = prob
        # data = [dic]
        # df_prob = df_prob.append(data, ignore_index=True, sort=False)
        df_prob = df_prob.append(prob, ignore_index=True, sort=False)

    return df_prob


def plot_avg_diff_prob(diff_video_num, con_video_num, end_of_file_name, dir_name, title):

    w=30
    df = pd.DataFrame()

    window_times = []
    wt_c = [wt * 1 for wt in range(0, 250, w)]
    window_times.append(np.array(wt_c) / 3600 * 300)
    df_delta_prob_diff = pickle.load(
        open(dir_name + "/" + f"df_prob_w={w}, video_num={diff_video_num}" + end_of_file_name, 'rb'))
    df_delta_prob_con = pickle.load(
        open(dir_name + "/" + f"df_prob_w={w}, video_num={con_video_num}" + end_of_file_name, 'rb'))

    # avg_vals_diff = ([df_delta_prob_diff[col].mean() for col in wt_c])
    # std_vals_diff = ([df_delta_prob_diff[col].std() for col in wt_c])
    # avg_vals_con = ([df_delta_prob_con[col].mean() for col in df_delta_prob_con.columns])
    # std_vals_con = ([df_delta_prob_con[col].std() for col in wt_c[:5]])

    df_delta_prob_diff.dropna(axis=1, how='all', inplace=True)
    avg_vals_diff = ([df_delta_prob_diff[col].mean() for col in df_delta_prob_diff.columns])
    std_vals_diff = ([df_delta_prob_diff[col].std() for col in df_delta_prob_diff.columns])

    df_delta_prob_con.dropna(axis=1, how='all', inplace=True)
    avg_vals_con = ([df_delta_prob_con[col].mean() for col in df_delta_prob_con.columns])
    std_vals_con = ([df_delta_prob_con[col].std() for col in df_delta_prob_con.columns])


    # plot it!

    fig, ax = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(8, 4), dpi=140)
    ax.plot(window_times[:5], avg_vals_con)
    ax.plot(window_times, avg_vals_diff)
    ax.set_title(f"window size={window}")


    p_std = np.asarray(avg_vals_con[:5]) + np.asarray(std_vals_con)
    m_std = np.asarray(avg_vals_con[:5]) - np.asarray(std_vals_con)
    ax.fill_between(window_times[:5], m_std, p_std, alpha=0.5, color="blue")

    p_std = np.asarray(avg_vals_diff) + np.asarray(std_vals_diff)
    m_std = np.asarray(avg_vals_diff) - np.asarray(std_vals_diff)
    ax.fill_between(window_times, m_std, p_std, alpha=0.5, color="DarkOrange")
    ax.grid()

    plt.suptitle(title, wrap=True)
    fig.legend(['control', 'ERK'], loc="lower left")

    for ax in fig.get_axes():
        ax.set_xlabel('Time [h]')
        ax.set_ylabel(' Avg confidence')
        ax.label_outer()
    plt.savefig(
        dir_name + "/" + f"{title} 30.png")
    plt.show()
    plt.close(fig)


def run_calc(dir_name, tracks, video_num, wt_cols, motility, intensity):
    clf, X_train, X_test, y_train, y_test = load_data(dir_name)

    if not os.path.exists(dir_name + "/" + f"df_prob_w={window}, video_num={video_num}"):
        df_prob = calc_prob_delta(window, tracks, clf, X_test, motility, intensity, wt_cols, calc_delta=False)
        pickle.dump(df_prob, open(dir_name + "/" + f"df_prob_w={window}, video_num={video_num}", 'wb'))


def run_plot_delta(dir_name, intensity, motility, diff_vid_num, con_vid_num):
    title = f"Averaged diff probability (intensity= {intensity}, motility={motility}, video #{con_vid_num},{diff_vid_num})"
    plot_avg_diff_prob(diff_video_num=diff_vid_num, con_video_num=con_vid_num, end_of_file_name="",
                       dir_name=dir_name, title=title)


def run_delta(diff_t_w, motility, visual, video_diff_num, video_con_num, con_windows):
    video_num = 1
    target = 0
    print(f"video_nam: {video_num}")
    # load tracks and dataframe
    csv_path = fr"muscle-formation-diff/data/mastodon/train/int_measures_s{video_num}.csv"
    # csv_path = fr"../data/mastodon/test/int_measures_s{video_num}.csv"
    # csv_path = fr"../data/mastodon/Nuclei_{video_num}-vertices.csv"


    df = pd.read_csv(csv_path, encoding="ISO-8859-1")
    df['Spot track ID'] = df["label"]
    df['target'] = target

    tracks = list()
    for label, labeld_df in df.groupby('Spot track ID'):
        tracks.append(labeld_df)

    # open a new directory to save the outputs in
    dir_name = f"24-01-2022-manual_mastodon_motility-{motility}_intensity-{visual}"
    time_frame = f"{diff_t_w[0]},{diff_t_w[1]} frames ERK, {con_windows} frames con,{40} winsize"
    complete_path = dir_name + "/" + time_frame

    run_calc(complete_path, tracks, video_num, wt_cols, motility, visual)
    run_plot_delta(complete_path, visual, motility, video_diff_num, video_con_num)


if __name__ == '__main__':
    print("calc_delta_mastodon")

    # df = pickle.load(open(fr"../data/mastodon/train/int_measures_s{5}", 'rb'))
    # df.to_csv(f'../data/mastodon/train/int_measures_s{5}.csv')


    # params

    motility = False
    visual = True
    video_diff_num = 3
    video_con_num = 2
    video_num = 2

    con_windows = [[0, 30], [140, 170], [180, 210], [240, 270], [300, 330]]

    # print(f"video: {video_num}, motility: {motility}, visual: {visual}, con_list = {con_windows}")

    diff_t_window = [140, 170]
    window = 30
    wt_cols = [wt for wt in range(0, 260, window)]

    run_delta(diff_t_window, motility, visual, video_diff_num, video_con_num, con_windows)
