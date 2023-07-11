"""The "calc_single_cell_properties" module provides functions for calculating and retrieving various properties of
single cells based on their tracks and scores. It includes functions for adding spot position columns, calculating
speed, mean actin intensity, persistence, local density, monotonicity, and terminal differentiation time. These
properties are computed using the provided data frames and video information. """

import sys, os

from scipy.stats import spearmanr
from skimage import io
import pymannkendall as mk

from data_layer.features_calculator import LocalDensityFeaturesCalculator

ROLLING_VAL = 30
sys.path.append('/sise/home/shakarch/muscle-formation-regeneration')
sys.path.append(os.path.abspath('..'))
from data_layer.utils import *


def add_spot_position_columns(scores_df, vid_name):
    """
    Receives differentiation scores dataframe and video name, and adds the spot positions of the cells at each timepoint.
    :param scores_df: (pd.DataFrame) differntiation scores by motility & actin intensity models
    :param vid_name: (Str) name of the data's video stage ("S1"/"S2"/"S3"/"S5"/"S6"/"S8")
    :return: (pd.DataFrame) scores with spot positions (x,y) for each timepoint.
    """
    csv_path = consts.data_csv_path % (consts.REG_METHOD, vid_name)
    tracks_df, _ = get_tracks(csv_path, manual_tagged_list=False)
    tracks_df = tracks_df.drop_duplicates(subset=["Spot track ID", "Spot frame"])
    tracks_df_cols = ["Spot track ID", "Spot frame", "Spot position X", "Spot position Y"]
    scores_df = pd.merge(left=scores_df, right=tracks_df[tracks_df_cols], on=["Spot track ID", "Spot frame"],
                         how='inner')

    return scores_df


def get_speed(scores_df):
    """
    Calculates the speed of single cell through its track. Speed = sqrt( (x2-x1)^2 + (y2-y1)^2 )
    :param scores_df: (pd.DataFrame) differntiation scores by motility & actin intensity models
    :return: (pd.DataFrame) scores with speed for each timepoint.
    """
    pd.options.display.float_format = '{:,.4f}'.format

    scores_df = scores_df.sort_values("Spot frame")
    speed_x = scores_df.groupby(['Spot track ID'])["Spot position X"].transform(lambda x: x.diff())
    speed_y = scores_df.groupby(['Spot track ID'])["Spot position Y"].transform(lambda x: x.diff())
    scores_df["speed"] = np.sqrt(np.square(speed_x) + np.square(speed_y))

    return scores_df


def add_actin_intensity_mean(scores_df, actin_vid_path):
    """
    Calculates mean actin intensity of aingle cell through its track, within a window determined by a given window size (taken from the parameters file).
    :param scores_df: (pd.DataFrame) differntiation scores by motility & actin intensity models
    :param actin_vid_path: (Str) path of the actin video matches the data.
    :return: (pd.DataFrame) scores with mean actin intensity for each timepoint.
    """

    def calc_mean_actin(track, actin_vid):
        mean_int_lst = [get_centered_image(i, track, actin_vid, consts.WIN_SIZE).mean() for i in range(len(track))]
        return mean_int_lst

    actin_vid = io.imread(actin_vid_path)
    means = scores_df.groupby(['Spot track ID']).apply(lambda x: calc_mean_actin(x, actin_vid))
    scores_df["mean"] = scores_df.groupby(['Spot track ID'])['Spot track ID'].transform(
        lambda x: means[x.iloc[0]])

    return scores_df


def add_persistence(scores_df):
    """
    Persistence is the ratio between a single cell’s displacement and its full path length.
    Persistence of 1 implies that the cell migrated in a straight line.
    :param scores_df: (pd.DataFrame) differntiation scores by motility & actin intensity models
    :return: (pd.DataFrame) scores with persistence values for each timepoint.
    """

    def calc_persistence(track):
        persistence_data = list(get_persistence_df(track, 30)["persistence"])
        persistence_data = [np.nan for i in range(len(track) - len(persistence_data))] + persistence_data
        return persistence_data

    persistence_lst = scores_df.groupby(['Spot track ID']).apply(lambda x: calc_persistence(x))
    scores_df["persistence"] = scores_df.groupby(['Spot track ID'])['Spot track ID'].transform(
        lambda x: persistence_lst[x.iloc[0]])

    return scores_df


def get_local_density(scores_df, vid_num):
    """
    Local density: the number of nuclei within a radius of 50 µm around the cell.
    :param scores_df: (pd.DataFrame) differntiation scores by motility & actin intensity models
    :param vid_num: (int) number of the data's video stage (1/2/3/5/6/8)
    :return: (pd.DataFrame) scores with persistence values for each timepoint.
    """

    def calc_local_den(track, vid_num):
        track_id = track['Spot track ID'].max()
        local_den = list(calculator.get_single_cell_measures(track_id, track, None, None, vid_num)["local density"])
        return local_den

    calculator = LocalDensityFeaturesCalculator()
    local_den_lst = scores_df.groupby(['Spot track ID']).apply(lambda x: calc_local_den(x, vid_num))
    scores_df["local density"] = scores_df.groupby(['Spot track ID'])['Spot track ID'].transform(
        lambda x: local_den_lst[x.iloc[0]])

    return scores_df


def get_position(ind, df):
    """
    Retrieves the position coordinates and frame number for a single cell at the specified index.
    :param ind: (int) Index of the spot.
    :param df: (pandas.DataFrame) DataFrame containing track's data.
    :return: (tuple) Tuple containing the x-coordinate, y-coordinate, and frame number.
    """
    x = int(df.iloc[ind]["Spot position X"] / 0.462)
    y = int(df.iloc[ind]["Spot position Y"] / 0.462)
    spot_frame = int(df.iloc[ind]["Spot frame"])
    return x, y, spot_frame


def get_centered_image(ind, df, im_actin, window_size):
    """
    Retrieves a centered image patch from the specified spot index.

    :param ind: (int) Index of the spot.
    :param df: (pandas.DataFrame) DataFrame containing track's data.
    :param im_actin: (numpy.ndarray) Array of images.
    :param window_size: (int) Half-size of the window around the spot.
    :return: (numpy.ndarray) Cropped image patch.
    """
    x, y, spot_frame = get_position(ind, df)
    cropped = im_actin[spot_frame][x - window_size:x + window_size, y - window_size: y + window_size]
    return cropped


def get_path_len(x_vec, y_vec):
    """
    Calculates the path length based on the given x and y vectors.

    :param x_vec: (pandas.Series) Vector of x-coordinates.
    :param y_vec: (pandas.Series) Vector of y-coordinates.
    :return: (float) Path length.
    """
    delta_x = x_vec.diff()
    delta_y = y_vec.diff()
    path_len = np.sqrt(np.square(delta_x) + np.square(delta_y)).sum()
    return path_len


def get_displacement(x_vec, y_vec):
    """
    Calculates the displacement based on the given x and y vectors.

    :param x_vec: (pandas.Series) Vector of x-coordinates.
    :param y_vec: (pandas.Series) Vector of y-coordinates.
    :return: (float) Displacement.
    """
    delta_x = (x_vec.tail(1).values - x_vec.head(1).values)[0]
    delta_y = (y_vec.tail(1).values - y_vec.head(1).values)[0]
    displacement = np.sqrt(np.square(delta_x) + np.square(delta_y))
    return displacement


def get_persistence_df(track, temporal_window_len=ROLLING_VAL):
    """
    Calculates the persistence values for a given track and returns a DataFrame with persistence values and frame 
    numbers.

    :param track: (pandas.DataFrame) DataFrame containing track data.
    :param temporal_window_len: (int) Size of the time window for calculating persistence. Default is ROLLING_VAL.
    :return: (pandas.DataFrame) DataFrame with persistence values and frame numbers.
    """
    time_windows = [(track.iloc[val]['Spot frame']) for val in range(0 + temporal_window_len, len(track), 1)]
    time_windows.sort()
    time_frames_chunks = [time_windows[i:i + temporal_window_len] for i in range(0, len(time_windows), 1) if
                          len(time_windows[i:i + temporal_window_len]) == temporal_window_len]

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


def get_properties(scores_df, vid_name, actin_vid_path, exist_ok=True):
    """
    Retrieves single cell properties including spot position columns, speed, actin intensity mean, persistence,
    and local density. The properties are calculated and stored in a pickle file for future use.

    :param scores_df: (pandas.DataFrame) DataFrame containing scores data, tracks Ids, etc.
    :param vid_name: (str) Name of the video the data was taken from.
    :param actin_vid_path: (str) Path to the actin video for mean actin intensity computation.
    :param exist_ok: (bool) Flag indicating whether to load from the existing file if available. Default is True.
    :return: (pandas.DataFrame) DataFrame with calculated properties.
    """
    saving_path = consts.storage_path + f"data/properties_dfs/scores_df_{vid_name}.pkl"
    if exist_ok and os.path.exists(saving_path):
        scores_df = pickle.load(open(saving_path, 'rb'))
    else:
        scores_df = add_spot_position_columns(scores_df, vid_name)
        scores_df = get_speed(scores_df)
        scores_df = add_actin_intensity_mean(scores_df, actin_vid_path)
        scores_df = add_persistence(scores_df)
        scores_df = get_local_density(scores_df, vid_name[1])
        pickle.dump(scores_df, open(saving_path, 'wb'))

    return scores_df


def get_monotonicity_value(track_df, modality, time, rolling_w=1, return_p_val=False):
    """
    Calculates monotonicity, as defined by the Spearman correlation between differentiation score and time
    :param return_p_val: if True, return the p-value instead of the coefficients.
    :param track_df: (pd.DataFrame) single cell's track dataframe, with differentiation scores by motility & actin intensity models
    :param modality: (Str) "motility"/"intensity".
    :param time: (tuple) time range to calculate correlation in (start_time, end_time).
    :param rolling_w: (int) size of window for rolling average smoothing.
    :return: (float) correlation coefficient.
    """
    track_df = track_df[(track_df["time"] >= time[0]) & (track_df["time"] <= time[1])]
    track_df = track_df.astype('float64')

    val = 1 if return_p_val else 0
    corr = track_df.groupby('Spot track ID').apply(lambda df: df[f"score_{modality}"].rolling(rolling_w).mean()
                                                   .corr(df["time"].rolling(rolling_w).mean(),
                                                         method=lambda x, y: spearmanr(x, y)[val]))
    corr = np.array(corr)
    corr = corr[0] if len(corr) > 0 else np.nan

    return round(corr, 4)


def get_monotonicity(scores_df, modality, time, rolling_w=1, return_p_val=False):
    """
       Calculates and adds the monotonicity values to the scores DataFrame for the specified modality.
        :param return_p_val: if True, return the p-value instead of the coefficients.
       :param scores_df: (pandas.DataFrame) DataFrame containing scores data.
       :param modality: (str) Modality for which monotonicity values are calculated,
       Choose from "motility" or "intensity".
       :param time: (str) Time column in the scores DataFrame.
       :param rolling_w: (int) Rolling window size for calculating monotonicity. Default is 1.
       :return: (pandas.DataFrame) DataFrame with added monotonicity values.
       """

    def calc_monotonicity_value(track, modality, time, rolling_w):
        monotonicity = get_monotonicity_value(track, modality, time, rolling_w, return_p_val)
        return monotonicity

    monotonicity_lst = scores_df.groupby(['Spot track ID']).apply(
        lambda x: calc_monotonicity_value(x, modality, time, rolling_w))
    scores_df[f"monotonicity_{modality}"] = scores_df.groupby(['Spot track ID'])['Spot track ID'].transform(
        lambda x: monotonicity_lst[x.iloc[0]])

    return scores_df


def get_longest_sequences(df):
    """gets the lobgest sequence of consequatative time points within a single track's dataframe
    :param df: (pd.DataFrame) single cell's track dataframe, with differntiation scores by motility & actin intensity models higher then a threshold.
    :return longest_sequence: (pd.DataFrame) portion of the dataframe, with the longest sequance of acores.
    """
    df["Spot frame diff"] = df["Spot frame"].diff()
    sequences = [v for k, v in df.groupby((df["Spot frame diff"].shift() != df["Spot frame diff"]).cumsum())]
    if len(sequences) == 0:
        return None
    else:
        longest_sequence = max(sequences, key=len)
        return longest_sequence


def get_terminal_differentiation_time(track_df, modality, diff_threshold=0.78):
    """
    Calculates the terminal differentiation time for a track based on the specified modality.

    :param track_df: (pandas.DataFrame) DataFrame containing track data.
    :param modality: (str) Modality for which differentiation time is calculated.
                      Choose from "motility" or "intensity".
    :param diff_threshold: (float) Differentiation threshold value. Default is 0.78.
    :return: (float) Terminal differentiation time in hours.
    """
    track_df = track_df[(track_df["time"] <= track_df["fusion_time"])]
    track_df = track_df.sort_values("Spot frame", ascending=True)
    high_then_thresh = track_df[(track_df[f"score_{modality}"] >= diff_threshold)]
    high_then_thresh["Spot frame diff"] = high_then_thresh["Spot frame"].diff()

    longest_sequence = get_longest_sequences(high_then_thresh)
    if longest_sequence is None:
        diff_time = np.nan
    else:
        diff_frame = np.floor(longest_sequence["Spot frame"].min())
        diff_time = diff_frame * 1 / 12
    return diff_time


def get_stable_threshold_time(track_df, threshold, modality, time_point, time_thresh=24):
    """
    returns the time (in hours) of reaching the longest stable sequence of differentiation scores within a given range.
    :param track_df: (pd.DataFrame) single cell's track dataframe, with differentiation scores by motility & actin intensity models
    :param threshold: (int, int) threshold values for determining sequence of stable scores.
    :param modality: (Str) "motility"/"intensity".
    :param time_thresh: (float) time threshold to limit the track's data.
    :return diff_time: (float) time of reaching terminal differentiation (hours)
    """
    thresholded_data = track_df[(track_df[f"score_{modality}"] >= threshold[0])
                                & (track_df[f"score_{modality}"] <= threshold[1])
                                & (track_df[f"time"] <= time_thresh)]

    longest_sequence = get_longest_sequences(thresholded_data)
    if longest_sequence is None:
        threshold_time = np.nan
    else:
        threshold_frame = longest_sequence.head(1)["Spot frame"].values[0] if time_point == "first" else \
            longest_sequence.tail(1)["Spot frame"].values[0]
        threshold_time = threshold_frame * 1 / 12

    return threshold_time


def get_terminal_diff_time(scores_df, modality, diff_threshold):
    """
        Calculates the terminal differentiation time for each track based on the specified modality.

        :param scores_df: (pandas.DataFrame) DataFrame containing differentiation scores data.
        :param modality: (str) Modality for which differentiation time is calculated.
                          Choose from "motility" or "intensity".
        :param diff_threshold: (float) Differentiation threshold value.
        :return: (pandas.DataFrame) DataFrame with added terminal differentiation time column.
        """

    def calc_terminal_diff_time(track, modality, diff_threshold):
        terminal_diff_time = get_terminal_differentiation_time(track, modality, diff_threshold)
        return terminal_diff_time

    terminal_diff_time_lst = scores_df.groupby(['Spot track ID']).apply(
        lambda x: calc_terminal_diff_time(x, modality, diff_threshold))
    scores_df[f"terminal_diff_time_{modality}"] = scores_df.groupby(['Spot track ID'])['Spot track ID'].transform(
        lambda x: terminal_diff_time_lst[x.iloc[0]])

    return scores_df


def get_differentiation_fusion_duration(scores_df, modality):
    """
    Calculates the duration between differentiation and fusion for each track based on the specified modality.

    :param scores_df: (pandas.DataFrame) DataFrame containing differentiation scores data.
    :param modality: (str) Modality for which differentiation-fusion duration is calculated.
                     Choose from "motility" or "intensity".
    :return: (pandas.DataFrame) DataFrame with added differentiation-fusion duration column.
    """

    def calc_diff_fusion_duration(track, modality):
        duration = track["fusion_time"].iloc[0] - track[f"terminal_diff_time_{modality}"].iloc[0]
        return duration

    diff_fusion_duration_lst = scores_df.groupby(['Spot track ID']).apply(
        lambda x: calc_diff_fusion_duration(x, modality))
    scores_df[f"diff_to_fusion_duration_{modality}"] = scores_df.groupby(['Spot track ID'])['Spot track ID'].transform(
        lambda x: diff_fusion_duration_lst[x.iloc[0]])

    return scores_df


def get_diff_duration(scores_df, modality, low_thresh, high_thresh):
    """
    Calculates the duration between the last occurrence of a low threshold scores sequence sequence and the first
    occurrence of a high threshold scores sequence for each track, based on the specified modality.

    :param scores_df: (pandas.DataFrame) DataFrame containing differentiation scores data.
    :param modality: (str) Modality for which differentiation duration is calculated. Choose from "motility" or "intensity".
    :param low_thresh: (tuple) The lower threshold value.
    :param high_thresh: (tuple) The higher threshold value.
    :return: (pandas.DataFrame) DataFrame with added differentiation duration column.
    """

    def calc_diff_duration(track, modality, high_thresh, low_thresh):
        high_thresh_time = get_stable_threshold_time(track, high_thresh, modality, time_point="first")
        low_thresh_time = get_stable_threshold_time(track, low_thresh, modality, time_point="last",
                                                    time_thresh=high_thresh_time)
        duration = high_thresh_time - low_thresh_time
        return duration

    duration_lst = scores_df.groupby(['Spot track ID']).apply(
        lambda x: calc_diff_duration(x, modality, high_thresh, low_thresh))
    scores_df[f"diff_duration_{modality}"] = scores_df.groupby(['Spot track ID'])['Spot track ID'].transform(
        lambda x: duration_lst[x.iloc[0]])

    return scores_df


def get_diff_start_time(scores_df, modality, low_thresh, high_thresh):
    """
    Calculates the start time of differentiation based on the last occurrence of a low threshold following
    the first occurrence of a high threshold for each track based on the specified modality.

    :param scores_df: (pandas.DataFrame) DataFrame containing differentiation scores data.
    :param modality: (str) Modality for which differentiation start time is calculated. Choose from "motility" or "intensity".
    :param low_thresh: (tuple) The lower threshold value.
    :param high_thresh: (tuple) The higher threshold value.
    :return: (pandas.DataFrame) DataFrame with added differentiation start time column.
    """

    def calc_diff_start_time(track, modality, high_thresh, low_thresh):
        high_thresh_time = get_stable_threshold_time(track, high_thresh, modality, time_point="first")
        low_thresh_time = get_stable_threshold_time(track, low_thresh, modality, time_point="last",
                                                    time_thresh=high_thresh_time)
        return low_thresh_time

    start_time_lst = scores_df.groupby(['Spot track ID']).apply(
        lambda x: calc_diff_start_time(x, modality, high_thresh, low_thresh))
    scores_df[f"diff_start_time_{modality}"] = scores_df.groupby(['Spot track ID'])['Spot track ID'].transform(
        lambda x: start_time_lst[x.iloc[0]])

    return scores_df


def get_mannkendall(scores_df, modality):
    """
    Calculates the Mann-Kendall trend test statistic for each track based on the specified modality.

    :param scores_df: DataFrame containing differentiation scores data.
    :param modality: Modality for which Mann-Kendall test is calculated. Choose from "motility" or "intensity".
    :return: DataFrame with added Mann-Kendall test slope column.
    """

    def calc_mannkendall(track, modality):
        try:
            mannkendall = mk.original_test(track[f"score_{modality}"], alpha=0.05).slope
        except:
            return np.nan
        return mannkendall

    mannkendall_lst = scores_df.groupby(['Spot track ID']).apply(
        lambda x: calc_mannkendall(x, modality))
    scores_df[f"mannkendall_{modality}"] = scores_df.groupby(['Spot track ID'])['Spot track ID'].transform(
        lambda x: mannkendall_lst[x.iloc[0]])

    return scores_df


def get_property(scores_df, feature_name, modality, func, func_args):
    """
    Calculates a custom property for each track based on the specified function and arguments.

    :param scores_df: (pandas.DataFrame) DataFrame containing differentiation scores data.
    :param feature_name: (str) Name of the custom property.
    :param modality: (str) Modality for which the property is calculated.
    :param func: (callable) Custom function to calculate the property.
    :param func_args: (tuple) Arguments to be passed to the custom function.
    :return: (pandas.DataFrame) DataFrame with the added custom property column.
    """
    lst = scores_df.groupby(['Spot track ID']).apply(lambda x: func(x, *func_args))
    scores_df[f"{feature_name}_{modality}"] = scores_df.groupby(['Spot track ID'])['Spot track ID'].transform(
        lambda x: lst[x.iloc[0]])

    return scores_df


if __name__ == '__main__':
    print("single_cell_properties_calc")

    print("\n"f"===== current working directory: {os.getcwd()} =====")

    s_run = consts.vid_info_dict[sys.argv[1]]

    coord_df_path = f"Coordination/coordination_outputs/coordination_dfs/manual_tracking/ring_size_30/coord_mastodon_S{s_run['name'][1]} reg {consts.REG_METHOD}_n_dist={30}"  # .pkl

    csv_path = consts.data_csv_path % (consts.REG_METHOD, s_run['name'])

    for con_train_n, diff_train_n in [(1, 5), (2, 3)]:
        print(s_run, flush=True)
