import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from DataPreprocessing.load_tracks_xml import load_tracks_xml
from Motility.MotilityMeasurements import get_linearity, get_distance, get_total_distance, get_net_total_proportion, \
    get_monotonicity, get_msd
from TimeSeriesAnalysis.ts_fresh import load_data, get_path, get_prob_over_track, drop_columns, normalize_tracks


# from load_tracks_xml import load_tracks_xml
# from MotilityMeasurements import get_linearity, get_distance, get_total_distance, get_net_total_proportion, \
#     get_monotonicity, get_msd
# from ts_fresh import load_data, get_path, get_prob_over_track, drop_columns, normalize_tracks


def get_cell_speed(track, window_size):
    velocities = np.zeros(shape=(len(track), 2))
    total_velocity = []
    xs = track["x"].tolist()
    ys = track["y"].tolist()
    for i in range(len(track) - 1):
        velocities[i][0] = (xs[i + 1] - xs[i])  # * (10 ^ 4) / 58.4564 * 40  # um/h
        velocities[i][1] = (ys[i + 1] - ys[i])  # * (10 ^ 4) / 58.4564 * 40  # um/h
        total_velocity.append(np.sqrt(velocities[i][0] ** 2 + velocities[i][1] ** 2))
    # calculate averaged velocity for each time window
    avg_x = [np.mean(velocities[i:i + window_size, 0]) for i in range(0, len(xs), window_size)]
    avg_y = [np.mean(velocities[i:i + window_size, 1]) for i in range(0, len(ys), window_size)]
    avg_total = [np.mean(total_velocity[i:i + window_size]) for i in range(0, len(total_velocity), window_size)]
    return avg_x, avg_y, avg_total


def plot_measurement(data, dir_name, video_num, y, x="confidence"):
    fig = plt.Figure(figsize=(8, 6))
    ax = sns.scatterplot(data=data, x=x, y=y, hue=data.index, legend=False, palette="RdBu_r")

    norm = plt.Normalize(data.index.min(), data.index.max())
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])

    # ax.figure.colorbar(sm)
    plt.savefig(
        dir_name + "/" + f"{y} over confidence 30 all data, video {video_num}.png")
    plt.show()
    plt.close(fig)


if __name__ == '__main__':

    motility = True
    intensity = False
    video_num = 8  # 3,4 - control; 8,11 - ERKi
    window_size = 30
    t_windows_con = [[0, 30], [140, 170], [180, 210]]  # , [140, 170], [180, 210]

    dir_name = f"tmp_motility-{motility}_intensity-{intensity}/{140},{170} frames ERK, {t_windows_con} frames con"
    xml_path = get_path(fr"data/tracks_xml/260721/S{video_num}_Nuclei.xml")
    # xml_path = get_path(fr"../data/tracks_xml/0104/Experiment1_w1Widefield550_s{video_num}_all_0104.xml")

    # Load clf
    print(dir_name)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    clf, X_train, X_test, y_train, y_test = load_data(dir_name)

    # load tracks and dataframe
    tracks, _df = load_tracks_xml(xml_path)

    all_data = pd.DataFrame()
    for track in tracks:
        if len(track) > window_size * 3:
            linearity = []
            net_distance = []
            total_distance = []
            net_total_distance = []
            monotonicity = []
            msd_alpha = []

            avg_x, avg_y, avg_total = get_cell_speed(track, window_size)
            # calculate list of probabilities per window
            track = drop_columns(track, motility=motility, intensity=intensity)
            track = normalize_tracks(track, motility=motility, intensity=intensity)
            track_diff_confidence = get_prob_over_track(clf=clf, track=track, window=window_size, features_df=X_test)

            for i in range(0, len(track), window_size):
                track_portion = track[i:i + window_size]
                # linearity (to calculate persistence)
                linearity.append(get_linearity(track_portion))
                net_distance.append(
                    get_distance(x1=track_portion[track_portion["t"] == int(np.min(track_portion["t"]))]["x"].values[0],
                                 y1=track_portion[track_portion["t"] == int(np.min(track_portion["t"]))]["y"].values[0],
                                 x2=track_portion[track_portion["t"] == int(np.max(track_portion["t"]))]["x"].values[0],
                                 y2=track_portion[track_portion["t"] == int(np.max(track_portion["t"]))]["y"].values[
                                     0]))
                total_distance.append(get_total_distance(track_portion))
                for net, tot in zip(net_distance, total_distance):
                    net_total_distance.append(get_net_total_proportion(net, tot))
                monotonicity.append(get_monotonicity(track_portion))
                # msd_alpha.append(get_msd(track_portion))

            length = len(track_diff_confidence)
            # _df = pd.DataFrame(
            #     {"confidence": track_diff_confidence, "avg_x": avg_x[:length], "avg_y": avg_y[:length],
            #      "avg_total": avg_total[:length]})

            tmp_df = pd.DataFrame(
                {"confidence": track_diff_confidence, "avg_x": avg_x[:length], "avg_y": avg_y[:length],
                 "avg_total": avg_total[:length], "linearity": linearity[:length],
                 "net_distance": net_distance[:length], "total_distance": total_distance[:length],
                 "net_total_distance": net_total_distance[:length], "monotonicity": monotonicity[:length]
                 # , "msd_alpha": msd_alpha[:length]
                 })

            # all_data = pd.concat([all_data, _df], axis=0)
            all_data = pd.concat([all_data, tmp_df], axis=0)

            # plot_measurement(data=all_data, dir_name=dir_name, video_num=video_num, y="net_distance", x="confidence")
            # plot_measurement(data=all_data, dir_name=dir_name, video_num=video_num, y="linearity", x="confidence")
            # plot_measurement(data=all_data, dir_name=dir_name, video_num=video_num, y="avg_total", x="confidence")
            # plot_measurement(data=all_data, dir_name=dir_name, video_num=video_num, y="total_distance", x="confidence")
            # plot_measurement(data=all_data, dir_name=dir_name, video_num=video_num, y="net_total_distance",
            #                  x="confidence")
            # plot_measurement(data=all_data, dir_name=dir_name, video_num=video_num, y="monotonicity", x="confidence")
            # plot_measurement(data=all_data, dir_name=dir_name, video_num=video_num, y="msd_alpha", x="confidence")

    pickle.dump(all_data, open(dir_name + "/" + f"all_data , video {video_num}", 'wb'))
