import math
import os
import pickle
from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from skimage import io

# from TimeSeriesAnalysis.mastodon import load_data, drop_columns, normalize_tracks, get_prob_over_track


from mastodon import load_data, drop_columns, normalize_tracks, get_prob_over_track


def get_cell_speed(track, window_size):
    velocities = np.zeros(shape=(len(track), 2))
    total_velocity = []
    xs = track["Spot position X (µm)"].tolist()
    ys = track["Spot position Y (µm)"].tolist()
    for i in range(len(track) - 1):
        velocities[i][0] = (xs[i + 1] - xs[i])  # * (10 ^ 4) / 58.4564 * 40  # um/h
        velocities[i][1] = (ys[i + 1] - ys[i])  # * (10 ^ 4) / 58.4564 * 40  # um/h
        total_velocity.append(np.sqrt(velocities[i][0] ** 2 + velocities[i][1] ** 2))
    # calculate averaged velocity for each time window
    avg_x = [np.mean(velocities[i:i + window_size, 0]) for i in range(0, len(xs), 1)]
    avg_y = [np.mean(velocities[i:i + window_size, 1]) for i in range(0, len(ys), 1)]
    avg_total = [np.mean(total_velocity[i:i + window_size]) for i in range(0, len(total_velocity), 1)]
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


def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value ** 2


def get_linearity(track):
    x = np.array(track['Spot position X (µm)']).reshape((-1, 1))
    y = np.array(track['Spot position Y (µm)'])
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    return r_sq


def get_total_distance(track):
    xs = track['Spot position X (µm)'].tolist()
    ys = track['Spot position Y (µm)'].tolist()
    total_distance = 0
    for i in range(len(track) - 1):
        total_distance += get_distance(xs[i], ys[i], xs[i + 1], ys[i + 1])
    return total_distance


# Calculate distance between 2 points
def get_distance(x1, y1, x2, y2):
    return math.sqrt(math.pow(x2 - x1, 2) +
                     math.pow(y2 - y1, 2) * 1.0)


def get_net_total_proportion(net_distance, total_distance):
    if total_distance == 0:
        distance_prop = 0
    else:
        distance_prop = net_distance / total_distance
    return distance_prop


def get_monotonicity(track):
    x = np.array(track['Spot position X (µm)']).reshape((-1, 1))
    y = np.array(track['Spot position Y (µm)'])
    rho, p_value = stats.spearmanr(x, y)
    return rho


def get_cell_area(track, nuc_video):
    image_size = 32

    for i in range(len(track)):
        x = int(track.iloc[i]["Spot position X (µm)"])
        y = int(track.iloc[i]["Spot position Y (µm)"])

        single_cell_crop = nuc_video[int(track.iloc[i]['Spot frame']), y - image_size:y + image_size,
                           x - image_size:x + image_size]

        # def val(size, xy):
        #     if xy - size < 0:
        #         return xy
        #     else:
        #         return size
        #
        # image_size = 80
        # x = int(track.iloc[23]["Spot position X (µm)"] / 264.5833)
        # y = int(track.iloc[23]["Spot position Y (µm)"] / 264.5833)
        #
        # frame = nuc_video[int(track.iloc[i]['Spot frame'])]
        # single_cell_crop = frame[y - val(image_size, y):y + image_size,
        #                    x - val(image_size, x):x + image_size]
        #
        # plt.imshow(single_cell_crop)
        # plt.show()


def height_weight(track):
    pass


def calc_motility_measures(tracks, window_size, clf, X_test):
    motility = True
    visual = False
    all_data = pd.DataFrame()

    for track in tracks:
        if len(track) > window_size:
            track = track.sort_values('Spot frame')
            tmp_df = get_measueres(X_test, clf, motility, track, visual, window_size)
            tmp_df["Spot frame"] = track["Spot frame"].values[1:]
            tmp_df["Spot track ID"] = track["Spot track ID"].values[1:]
            all_data = pd.concat([all_data, tmp_df], axis=0)
    return all_data


def get_measueres(X_test, clf, motility, track, visual, window_size):
    t = 'Spot frame'
    x = 'Spot position X (µm)'
    y = 'Spot position Y (µm)'
    linearity = []
    net_distance = []
    total_distance = []
    net_total_distance = []
    monotonicity = []
    msd_alpha = []
    avg_x, avg_y, avg_total = get_cell_speed(track, window_size)
    length = 0
    for i in range(0, len(track), 1):
        length += 1
        track_portion = track[i:i + window_size]
        # linearity (to calculate persistence)
        linearity.append(get_linearity(track_portion))

        net_distance.append(
            get_distance(x1=track_portion[track_portion[t] == int(np.min(track_portion[t]))][x].values[0],
                         y1=track_portion[track_portion[t] == int(np.min(track_portion[t]))][y].values[0],
                         x2=track_portion[track_portion[t] == int(np.max(track_portion[t]))][x].values[0],
                         y2=track_portion[track_portion[t] == int(np.max(track_portion[t]))][y].values[0]))
        total_distance.append(get_total_distance(track_portion))
        for net, tot in zip(net_distance, total_distance):
            net_total_distance.append(get_net_total_proportion(net, tot))
        monotonicity.append(get_monotonicity(track_portion))
        # msd_alpha.append(get_msd(track_portion))

    # calculate list of probabilities per window
    # track = drop_columns(track, motility=motility, visual=visual)
    # track = normalize_tracks(track, motility=motility, visual=visual)
    # track_diff_confidence = get_prob_over_track(clf=clf, track=track, window=window_size, features_df=X_test)
    # length = len(track_diff_confidence)
    # length = len(track)
    length = length - 1
    tmp_df = pd.DataFrame(
        {"avg_x": avg_x[:length], "avg_y": avg_y[:length],  # "confidence": track_diff_confidence
         "avg_total": avg_total[:length], "linearity": linearity[:length],
         "net_distance": net_distance[:length], "total_distance": total_distance[:length],
         "net_total_distance": net_total_distance[:length], "monotonicity": monotonicity[:length]
         # , "msd_alpha": msd_alpha[:length]
         })
    return tmp_df


def calc_visual_measures(tracks, window_size, clf, X_test, video_num):
    vid_path = fr"C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Videos\06102021\ERK\S{video_num}_nuclei.tif"
    vid = io.imread(vid_path)
    motility = False
    visual = True
    all_data = pd.DataFrame()
    t = 'Spot frame'
    x = 'Spot position X (µm)'
    y = 'Spot position Y (µm)'
    for track in tracks:
        area = []
        if len(track) > window_size * 2:

            for i in range(0, len(track), window_size):
                track_portion = track[i:i + window_size]
                ar = get_cell_area(track_portion, vid)
            # calculate list of probabilities per window
            track = drop_columns(track, motility=motility, visual=visual)
            track = normalize_tracks(track, motility=motility, visual=visual)
            track_diff_confidence = get_prob_over_track(clf=clf, track=track, window=window_size,
                                                        features_df=X_test)

    return all_data


if __name__ == '__main__':
    lst_v = [1, 5]  # 1 = Control, 5 = ERK
    video_diff_num = 5  # 8
    video_con_num = 1  # 3
    dif_window = [140, 170]
    motility = True
    visual = False
    con_windows = [[0, 30], [140, 170], [180, 210], [240, 270], [300, 330]]
    video_num = 5

    dir_name = f"21-12-2021-manual_mastodon_motility-{motility}_intensity-{visual}"
    time_frame = f"{dif_window[0]},{dif_window[1]} frames ERK, {con_windows} frames con"
    complete_path = dir_name + "/" + time_frame

    # # load the model & train set & test set
    clf, X_train, X_test, y_train, y_test = load_data(complete_path)

    csv_path = fr"muscle-formation-diff/data/mastodon/Nuclei_{video_num}-vertices.csv"
    df = pd.read_csv(csv_path, encoding="cp1252")
    df = df[df["manual"] == 1]
    all_data = pd.DataFrame()

    for label, label_df in df.groupby('Spot track ID'):
        if len(label_df) > 140:
            track = drop_columns(label_df, motility=motility, visual=visual)
            track = normalize_tracks(track, motility=motility, visual=visual)
            track_diff_confidence = get_prob_over_track(clf=clf, track=track, window=30, features_df=X_test,
                                                        moving_window=True)

            all_data = pd.concat(
                [all_data, pd.DataFrame({"w_confidence": track_diff_confidence, "label": label})], axis=0)

    pickle.dump(all_data, open(
        complete_path + "/" + f"moving window confidence, motility={motility}, intensity={visual}, video #{video_num}",
        'wb'))

    # motility = True
    # visual = False
    # window_size = 30
    # control_videos = [3, 4]
    # erk_videos = [8, 11]
    # dif_window = [140, 170]
    #
    # video_num = 5  # 3,4 - control; 8,11 - ERKi
    # # con_windows = [[0, 30], [140, 170], [180, 210]]
    # # con_windows = [[0, 30], [30, 60], [60, 90], [90, 120],
    # #                [120, 150], [150, 180], [180, 210], [210, 240],
    # #                [240, 270], [270, 300], [300, 330]]
    # con_windows = [[0, 30], [140, 170], [180, 210], [240, 270], [300, 330]]
    #
    # dir_name = f"21-12-2021-manual_mastodon_motility-True_intensity-False"
    # time_frame = f"{dif_window[0]},{dif_window[1]} frames ERK, {con_windows} frames con"
    # complete_path = dir_name + "/" + time_frame
    #
    # csv_path = fr"muscle-formation-diff/data/mastodon/Nuclei_{video_num}-vertices.csv"
    # # csv_path = fr"../data/mastodon/Nuclei_{video_num}-vertices.csv"
    #
    # # load tracks and dataframe
    # df = pd.read_csv(csv_path, encoding="cp1252")
    #
    # # get only manual tracked cells:
    # df = df[df["manual"] == 1]
    #
    # # df["target"] = True if video_num in erk_videos else False
    # tracks = list()
    # for label, labeled_df in df.groupby('Spot track ID'):
    #     if len(labeled_df) > 180:  # consider only long tracks
    #         labeled_df["label"] = label
    #         tracks.append(labeled_df)
    #
    # # Load clf
    # print(complete_path)
    # clf, X_train, X_test, y_train, y_test = load_data(complete_path)
    #
    # if motility:
    #     all_data = calc_motility_measures(tracks, window_size, clf, X_test)
    # else:
    #     all_data = calc_visual_measures(tracks, window_size, clf, X_test, video_num)
    #
    # pickle.dump(all_data,
    #             open(complete_path + "/" + f"all_data_manual, motility={motility}, intensity={visual}, video #{video_num}",
    #                  'wb'))
