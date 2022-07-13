import pickle
from functools import reduce

import consts
import diff_tracker_utils as utils
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from build_models_on_transformed_tracks import get_to_run
from scipy.spatial import distance
import scipy
from skimage import io

ROLLING_VAL = 30


def correlations_percentage(df_mot, df_int, tagged_df, all_tracks, dir_path, s_name, actin_vid_path):
    second_dir = s_name + "_properties"
    save_dir = dir_path + "/" + second_dir
    all_correlations_df = pd.DataFrame()
    for label, track_score_mot in df_mot.groupby("Spot track ID"):
        track_score_int = df_int[df_int["Spot track ID"] == label]
        if len(track_score_mot.iloc[0]) < 60 or len(track_score_int) == 0:
            continue

        # correlations_df = get_correlations_df(track, tagged_df, label, all_tracks)
        correlations_df = get_single_cell_properties(track_score_mot, track_score_int,
                                                     tagged_df[tagged_df["Spot track ID"] == label],
                                                     all_tracks, actin_vid_path, actin_win_size=7)
        all_correlations_df = all_correlations_df.append(correlations_df, ignore_index=True)
    pickle.dump(all_correlations_df, open(save_dir, 'wb'))


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


def get_coordination_data(track_coord_data):
    try:
        coord = [np.nan for i in range(int(track_coord_data["t0"].max()))]
        coord.extend(list(track_coord_data["cos_theta"].values)[0][0])
    except:
        return pd.DataFrame()
    return coord


def get_local_density(label_track, all_tracks, track_coord_data):
    local_den_df = utils.add_features(label_track, local_density=True, df_s=all_tracks)
    local_den_df = local_den_df.sort_values("Spot frame")
    local_density = [np.nan for i in range(int(track_coord_data["t0"].max()))]
    local_density.extend(local_den_df["local density"].values.tolist())
    # local_density = [0 for i in range(len(coord))]
    return local_density


def get_speed(track_data):
    # calc speed
    track_data = track_data[["Spot frame", "Spot position X (µm)", "Spot position Y (µm)", "Spot track ID"]]
    track_data = track_data.sort_values("Spot frame")
    track_data["speed_x"] = track_data["Spot position X (µm)"].diff()
    track_data["speed_y"] = track_data["Spot position Y (µm)"].diff()
    track_data["speed"] = np.sqrt(np.square(track_data["speed_x"]) + np.square(track_data["speed_y"]))
    track_data["speed_change"] = track_data["speed"].diff()
    return track_data[["speed", "speed_x", "speed_y", "speed_change", "Spot frame", "Spot track ID"]]


def calc_speed_df(df_mot, dir_path, s_name):
        second_dir = s_name + "_speed"
        save_dir = dir_path + "/" + second_dir
        total_speed_df = pd.DataFrame()
        for label, track_score_mot in df_mot.groupby("Spot track ID"):
            total_speed_df = total_speed_df.append(get_speed(track_score_mot), ignore_index=True)

        # print(total_speed_df)

        pickle.dump(total_speed_df, open(save_dir, 'wb'))


def get_rolling_avg(feature_list, df, rolling_window_size=ROLLING_VAL):
    for feature_name in feature_list:
        df['rolling_' + feature_name] = df[feature_name].rolling(window=rolling_window_size).mean()
    return df


def get_path_len(x_vec, y_vec):
    delta_x = x_vec.diff()
    delta_y = y_vec.diff()
    path_len = np.sqrt(np.square(delta_x) + np.square(delta_y)).sum()
    return path_len


def get_displacement(x_vec, y_vec):
    delta_x = (x_vec.tail(1).values - x_vec.head(1).values)[0]
    delta_y = (y_vec.tail(1).values - y_vec.head(1).values)[0]
    displacement = np.sqrt(np.square(delta_x) + np.square(delta_y))
    return displacement


def get_directionality_ratio(track, window_size=ROLLING_VAL):
    time_windows = [(track.iloc[val]['Spot frame']) for val in range(0 + window_size, len(track), 1)]
    time_windows.sort()
    time_frames_chunks = [time_windows[i:i + window_size] for i in range(0, len(time_windows), 1) if
                          len(time_windows[i:i + window_size]) == window_size]

    directionality_ratios = []
    time = []
    for chunk in time_frames_chunks:
        track_portion = track[track["Spot frame"].isin(chunk)]
        x_vec = track_portion["Spot position X (µm)"]
        y_vec = track_portion["Spot position Y (µm)"]
        displacement = get_displacement(x_vec, y_vec)
        path_len = get_path_len(x_vec, y_vec)
        directionality_ratio = displacement / path_len
        directionality_ratios.append(directionality_ratio)
        time.append(track_portion["Spot frame"].max())

    directionality_ratios_df = pd.DataFrame({"directionality_ratio": directionality_ratios, "Spot frame": time})
    return directionality_ratios_df


def get_directionality(track, speed_df, window_size=ROLLING_VAL):
    directionality = get_directionality_ratio(track, window_size)
    track["directionality_cos_alpha"] = \
        speed_df["speed_x"] / np.sqrt(speed_df["speed_x"] ** 2 + speed_df["speed_y"] ** 2)
    directionality = pd.merge(directionality, track[["directionality_cos_alpha", "Spot frame", "Spot track ID"]], on=["Spot frame"])
    return directionality


def get_single_cell_properties(track_coord_data_mot_score, track_score_int, track, all_tracks, actin_vid_path,
                               actin_win_size=7):
    coord = get_coordination_data(track_coord_data_mot_score)
    local_density = get_local_density(track, all_tracks, track_coord_data_mot_score)
    score_mot = track_coord_data_mot_score.copy().iloc[0, :].drop(
        columns=["Spot track ID", "Track #", "t0", "cos_theta"])
    score_int = track_score_int.copy().iloc[0, :].drop(
        columns=["Spot track ID", "Track #", "t0", "cos_theta"])
    length = min([len(score_mot), len(coord), len(local_density)])
    spot_track_id = [track["Spot track ID"].values[0] for i in range(length)]
    spot_frame = [i for i in range(length)]

    properties_df = pd.DataFrame({
        'score_motility': score_mot[:length],
        'score_intensity': score_int[:length],
        'coordination': coord[:length],
        'local density': local_density[:length],
        'Spot frame': spot_frame,
        'time': np.array(spot_frame) * 5 / 60,
        'Spot track ID': spot_track_id,
    })

    speed_df = get_speed(track)
    actin_df = utils.get_single_cell_intensity_measures(label=spot_track_id[0], df=track,
                                                        im_actin=io.imread(actin_vid_path), window_size=actin_win_size)
    directionality_df = get_directionality(track, speed_df, ROLLING_VAL)

    properties_df = reduce(lambda df_left, df_right:
                           pd.merge(df_left, df_right, on=["Spot frame", "Spot track ID"]),
                           [properties_df, speed_df[["speed", "speed_change", "Spot frame", "Spot track ID"]], actin_df,
                            directionality_df])

    rolling_features_list = ['coordination', 'speed', 'speed_change', 'local density', 'mean',
                             'directionality_cos_alpha']
    properties_df = get_rolling_avg(rolling_features_list, properties_df, ROLLING_VAL)

    return properties_df


if __name__ == '__main__':
    print("single_cell_properties_calc")

    diff_window = [140, 170]
    tracks_len = 30
    con_windows = [[0, 30], [40, 70], [90, 120], [140, 170], [180, 210], [220, 250]]

    s_runs = [consts.s5, consts.s1, consts.s3, consts.s2]
    path = consts.cluster_path

    local_density = False
    for con_train_n, diff_train_n, con_test_n, diff_test_n in [(2, 3, 1, 5), (1, 5, 2, 3), ]:

        for s_run in s_runs:
            print(s_run)
            dir_path_score = get_dir_path("motility")
            df_merge_col_mot, df_tagged_mot, df_all_tracks_mot = load_correlations_data(s_run, dir_path_score)
            dir_path_score = get_dir_path("intensity")
            df_merge_col_int, df_tagged_int, df_all_tracks_int = load_correlations_data(s_run, dir_path_score)

            correlations_percentage(df_merge_col_mot, df_merge_col_int, df_tagged_mot, df_all_tracks_mot,
                                    dir_path_score, s_run["name"],
                                    path + s_run["actin_path"])

            # calc_speed_df(df_tagged_mot, dir_path_score, s_run["name"])
            # print(pickle.load(open(dir_path_score + r"\S5_speed", 'rb')))