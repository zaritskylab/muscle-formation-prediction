import pickle
from functools import reduce

import sys, os
import numpy as np
import pandas as pd
from skimage import io

ROLLING_VAL = 30
sys.path.append('/sise/home/shakarch/muscle-formation-diff')
sys.path.append(os.path.abspath('..'))
from TimeSeriesAnalysis import consts
from TimeSeriesAnalysis.utils.diff_tracker_utils import *
from TimeSeriesAnalysis.utils.data_load_save import *


def calc_properties_df(score_df_mot, score_df_int, tagged_df, all_tracks, save_dir_path, s_run, local_density,
                       coordination):
    """
    calculates cellular properties for a tracks dataframe
    :param local_density: (bool) calculates local density if True
    :param score_df_mot: scores dataframe of the motility-based predictor
    :param score_df_int:scores dataframe of the actin-intensity-based predictor
    :param tagged_df: dataframe of only tagged tracks
    :param all_tracks: dataframe of all tracks
    :param save_dir_path: path for saving the properties dataframe output
    :param s_run: dictionary, contains details of the experiment to calculate properties on
    :return:
    """
    second_dir = s_run['name'] + "_properties" + "_" + params.registration_method
    save_dir = save_dir_path + "/" + second_dir
    all_correlations_df = pd.DataFrame()
    actin_vid = io.imread(s_run['actin_path'])

    for label, track_score_mot in score_df_mot.groupby("Spot track ID"):
        track_score_int = score_df_int[score_df_int["Spot track ID"] == label]

        # skip track if ____
        if len(track_score_mot.iloc[0]) < params.tracks_len or len(track_score_int) == 0:
            continue

        correlations_df = get_single_cell_properties(track_score_mot, track_score_int,
                                                     tagged_df[tagged_df["Spot track ID"] == label],
                                                     all_tracks, actin_vid, actin_win_size=params.window_size,
                                                     local_density=local_density, coordination=coordination)
        all_correlations_df = all_correlations_df.append(correlations_df, ignore_index=True)
    print(f"saving into {save_dir}")
    print(f"size {all_correlations_df.shape}")

    pickle.dump(all_correlations_df, open(save_dir + ".pkl", 'wb'))


def load_correlations_data(s_run, dir_path_score, coordination_df_path, tracks_csv_path, get_coordination=True,
                           get_local_den=False):
    """
    loads scores dataframe, tracks dataframe and coordination dataframe
    :param s_run:
    :param dir_path_score:
    :param coordination_df_path:
    :param tracks_csv_path:
    :param get_coordination:
    :param get_local_den:
    :return:
    """
    print(f"load correlations data: {s_run['name']}", flush=True)
    # get single cells score df
    df_score = pickle.load(open(dir_path_score + f"/df_prob_w=30, video_num={s_run['name'][1]}", 'rb'))

    # get tagged tracks
    df_all_tracks, _ = get_tracks(tracks_csv_path, manual_tagged_list=False)
    df_tagged = df_all_tracks[df_all_tracks["manual"] == 1]

    if get_local_den:
        # calculate local density
        local_density_df = add_features_df(df_tagged, df_all_tracks, local_density=get_local_den)

    if get_coordination:
        # get coordination df
        df_coord = pickle.load((open(coordination_df_path, 'rb')))
        df_coord["Spot track ID"] = df_coord["Track #"].apply(lambda x: x)
        df_coord["Spot frame"] = df_coord["cos_theta"].apply(lambda x: x[0])
        df_coord["cos_theta_values"] = df_coord["cos_theta"].apply(lambda x: x[1])

        df_score['cos_theta'] = -1
        df_score['t0'] = -1
        df_score['cos_theta'] = df_score['cos_theta'].astype('object')

        for index, row in df_score.iterrows():
            track_id = row["Spot track ID"]
            cos_theta_list = df_coord[df_coord["Spot track ID"] == track_id]["cos_theta_values"].to_list()
            df_score['cos_theta'].iloc[index] = [cos_theta_list]
            df_score['t0'].iloc[index] = df_coord[df_coord["Spot track ID"] == track_id]["t0"].max()

    return df_score, df_tagged, df_all_tracks


def get_dir_path(modality, con_train_n, diff_train_n, ):
    dir_path = consts.storage_path + f"30-07-2022-{modality} local dens-{params.local_density}, s{con_train_n}, s{diff_train_n} train" + (
        f" win size {params.window_size}" if modality != "motility" else "")
    second_dir = f"track len {params.tracks_len}, impute_func-{params.impute_methodology}_{params.impute_func} reg {params.registration_method}"
    dir_path += "/" + second_dir
    return dir_path


def get_coordination_data(track_coord_data):
    try:
        # coord = [np.nan for i in range(int(track_coord_data["t0"].max()))]
        coord = [np.nan for i in range(int(track_coord_data.dropna(axis=1).columns[0]))]
        coord.extend(list(track_coord_data["cos_theta"].values)[0][0])
    except:
        return pd.DataFrame()
    return coord


def get_local_density(label_track, all_tracks, track_coord_data):
    local_den_df = add_features(label_track, local_density=True, df_s=all_tracks)
    local_den_df = local_den_df.sort_values("Spot frame")
    # local_density = [np.nan for i in range(int(track_coord_data["t0"].max()))]
    local_density = [np.nan for i in range(int(local_den_df["Spot frame"].min()))]
    local_density.extend(local_den_df["local density"].values.tolist())
    return local_density


def get_speed(track_data):
    # calc speed
    track_data = track_data[["Spot frame", "Spot position X", "Spot position Y", "Spot track ID"]]
    track_data = track_data.sort_values("Spot frame")
    track_data["speed_x"] = track_data["Spot position X"].diff()
    track_data["speed_y"] = track_data["Spot position Y"].diff()
    track_data["speed"] = np.sqrt(np.square(track_data["speed_x"]) + np.square(track_data["speed_y"]))
    track_data["speed_change"] = track_data["speed"].diff()
    speed_df = track_data[
        ["speed", "speed_change", "Spot frame", "Spot track ID", "Spot position X", "Spot position Y", "speed_x",
         "speed_y"]]

    return speed_df


def calc_speed_df(df_mot, dir_path, s_name):
    second_dir = s_name + "_speed"
    save_dir = dir_path + "/" + second_dir
    total_speed_df = pd.DataFrame()
    for label, track_score_mot in df_mot.groupby("Spot track ID"):
        total_speed_df = total_speed_df.append(get_speed(track_score_mot), ignore_index=True)

    pickle.dump(total_speed_df, open(save_dir, 'wb'))


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


def get_persistence_df(track, window_size=ROLLING_VAL):
    time_windows = [(track.iloc[val]['Spot frame']) for val in range(0 + window_size, len(track), 1)]
    time_windows.sort()
    time_frames_chunks = [time_windows[i:i + window_size] for i in range(0, len(time_windows), 1) if
                          len(time_windows[i:i + window_size]) == window_size]

    persistence_lst = []
    frames_lst = []
    for chunk in time_frames_chunks:
        track_portion = track[track["Spot frame"].isin(chunk)]
        x_vec = track_portion["Spot position X"]
        y_vec = track_portion["Spot position Y"]
        displacement = get_displacement(x_vec, y_vec)
        path_len = get_path_len(x_vec, y_vec)
        persistence = displacement / path_len
        persistence_lst.append(persistence)
        frames_lst.append(track_portion["Spot frame"].max())

    persistence_df = pd.DataFrame({"persistence": persistence_lst, "Spot frame": frames_lst})
    return persistence_df


def get_directionality_properties(track, speed_df, window_size=ROLLING_VAL):
    direct_props_df = get_persistence_df(track, window_size)
    track["directionality_cos_alpha"] = \
        speed_df["speed_x"] / np.sqrt(speed_df["speed_x"] ** 2 + speed_df["speed_y"] ** 2)
    direct_props_df = pd.merge(direct_props_df, track[["directionality_cos_alpha", "Spot frame", "Spot track ID"]],
                               on=["Spot frame"])
    return direct_props_df


def get_single_cell_properties(track_coord_data_mot_score, track_score_int, track, all_tracks, actin_vid,
                               actin_win_size, coordination, local_density=True):
    cols_to_drop = ["Spot track ID", "Track #", "t0", "cos_theta"]
    score_mot = track_coord_data_mot_score.copy().iloc[0, :].drop(columns=cols_to_drop)
    score_int = track_score_int.copy().iloc[0, :].drop(columns=cols_to_drop)
    spot_track_id = track["Spot track ID"].values[0]

    coord = get_coordination_data(track_coord_data_mot_score) if coordination else [np.nan for i in range(500)]
    local_density_df = get_local_density(track, all_tracks, track_coord_data_mot_score) if local_density else [np.nan
                                                                                                               for i in
                                                                                                               range(
                                                                                                                   500)]

    length = min([len(score_mot), len(coord), len(local_density_df)])
    spot_frame = [i for i in range(length)]

    properties_df = pd.DataFrame({
        'score_motility': score_mot[:length],
        'score_intensity': score_int[:length],
        'coordination': coord[:length],
        'local_density': local_density_df[:length],
        'Spot frame': spot_frame,
        'time': np.array(spot_frame) * 5 / 60,
        'Spot track ID': spot_track_id,
    })

    speed_df = get_speed(track).astype(float)  # calc speed
    persistence_df = get_directionality_properties(track, speed_df, ROLLING_VAL).astype(float)
    actin_df = get_single_cell_intensity_measures(label=spot_track_id, df=track,
                                                  im_actin=actin_vid, window_size=actin_win_size).astype(float)

    properties_df = reduce(lambda df_left, df_right:
                           pd.merge(df_left, df_right, on=["Spot track ID", "Spot frame"]),
                           # todo: I changed the order from ["Spot frame", "Spot track ID"] to ["Spot track ID", "Spot frame"]
                           [properties_df, actin_df, persistence_df, speed_df[
                               ["speed", "speed_change", "Spot frame", "Spot track ID", "Spot position X",
                                "Spot position Y"]],
                            ])
    return properties_df


if __name__ == '__main__':
    print("single_cell_properties_calc")
    os.chdir("/home/shakarch/muscle-formation-diff")

    print("\n"f"===== current working directory: {os.getcwd()} =====")

    s_run = consts.s_runs[sys.argv[1]]

    coord_df_path = f"Coordination/coordination_outputs/coordination_dfs/manual_tracking/ring_size_30/coord_mastodon_S{s_run['name'][1]} reg {params.registration_method}_n_dist={30}"  # .pkl

    csv_path = consts.data_csv_path % (params.registration_method, s_run['name'])

    for con_train_n, diff_train_n in [(1, 5), (2, 3)]:
        print(s_run, flush=True)
        path_scores_df = get_dir_path("motility", con_train_n, diff_train_n)
        df_score_mot, df_tagged_mot, all_tracks_mot = load_correlations_data(s_run,
                                                                             path_scores_df,
                                                                             coord_df_path,
                                                                             csv_path,
                                                                             get_coordination=False)

        path_scores_df = get_dir_path("actin_intensity", con_train_n, diff_train_n)
        df_score_int, df_tagged_int, all_tracks_int = load_correlations_data(s_run, path_scores_df,
                                                                             coord_df_path,
                                                                             csv_path,
                                                                             get_coordination=False)

        calc_properties_df(df_score_mot, df_score_int, df_tagged_mot, all_tracks_mot,
                           path_scores_df, s_run, local_density=True, coordination=False)
