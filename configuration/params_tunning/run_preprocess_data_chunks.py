import os
import sys

sys.path.append('/sise/home/reutme/muscle-formation-regeneration')
sys.path.append(os.path.abspath('../..'))
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from preprocess_data_chunks import preprocess_data
from data_layer.utils import *

if __name__ == '__main__':
    print("running prepare_data")

    # get arguments from sbatch
    modality = sys.argv[1]
    n_tasks = int(sys.argv[2])
    s_run = consts.vid_info_dict[sys.argv[3]]

    # feature type is the name of the feature that we change
    feature_type = sys.argv[4]
    task_id = int(os.getenv('SLURM_ARRAY_TASK_ID')[1:])

    # if this is the original model - not sensitive analysis
    if feature_type is None:
        # build model by original feature
        preprocess_data(n_tasks=n_tasks, job_id=task_id, s_run=s_run, modality=modality,
                        win_size=params.window_size, track_len=params.tracks_len)


    # build model by sensitive analysis on crop window size feature
    elif feature_type == "window_size_arr":
        for win_size in params.feature_calc_types[feature_type]:
            print(f"start preprocess_data with window size: {win_size}")
            preprocess_data(n_tasks=n_tasks, job_id=task_id, s_run=s_run, modality=modality, win_size=win_size,
                            track_len=params.tracks_len, feature_type=feature_type, specific_feature_calc=win_size)
            print(f"finish preprocess_data with window size: {win_size}")

    # build model by sensitive analysis on temporal_segment feature
    elif feature_type == "temporal_segment_arr":
        # array of arrays that each array is one of temporal segment array, differentiation_window array and control window array
        list_of_tempural_diff_win_con_win = params.feature_calc_types[feature_type]
        temporal_segment_array = list_of_tempural_diff_win_con_win[0][0]
        diff_wind_array = list_of_tempural_diff_win_con_win[0][1]
        con_wind_array = list_of_tempural_diff_win_con_win[0][2]

        for temporal_segment, diff_wind, con_wind in zip(temporal_segment_array, diff_wind_array, con_wind_array):
            feature_specific = temporal_segment

            print(f"start preprocess_data with temporal segment size: {temporal_segment}")
            print(f"diff_window: {diff_wind}")
            print(f"con_wind: {con_wind}")
            preprocess_data(n_tasks=n_tasks, job_id=task_id, s_run=s_run, modality=modality,
                            win_size=params.window_size, track_len=temporal_segment,
                            feature_type=feature_type, specific_feature_calc=feature_specific)
            print(f"finish preprocess_data with temporal segment size: {temporal_segment}")

    # build model by sensitive analysis on diff_win feature
    elif feature_type == "diff_window_arr":
        for diff_win in params.feature_calc_types[feature_type]:
            feature_specific = diff_win
            preprocess_data(n_tasks=n_tasks, job_id=task_id, s_run=s_run, modality=modality,
                            win_size=params.window_size, track_len=params.tracks_len, feature_type=feature_type,
                            specific_feature_calc=feature_specific)
