import os
import sys

sys.path.append('/sise/home/reutme/muscle-formation-regeneration')
sys.path.append(os.path.abspath('../..'))
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from TimeSeriesAnalysis.data_preprocessing.preprocess_data_chunks import preprocess_data
import consts
from utils.diff_tracker_utils import *
from utils.data_load_save import *



if __name__ == '__main__':
    print("running prepare_data")

    # get arguments from sbatch
    modality = sys.argv[1]
    n_tasks = int(sys.argv[2])
    s_run = consts.s_runs[sys.argv[3]]
    feature_calc = sys.argv[4]
    task_id = int(os.getenv('SLURM_ARRAY_TASK_ID')[1:])


    # for tempural_diffwin_conwin in params.feature_calc_types[feature_calc]:
    #     print(tempural_diffwin_conwin)
    #     temporal_segment = tempural_diffwin_conwin[0]
    #     feature_specific = temporal_segment
    #     diff_wind = tempural_diffwin_conwin[1]
    #     con_wind = tempural_diffwin_conwin[2]

    list_of_tempural_diff_win_con_win = params.feature_calc_types[feature_calc]
    temporal_segment_array = list_of_tempural_diff_win_con_win[0][0]
    diff_wind_array = list_of_tempural_diff_win_con_win[0][1]
    con_wind_array = list_of_tempural_diff_win_con_win[0][2]

    for temporal_segment, diff_wind, con_wind in zip(temporal_segment_array, diff_wind_array, con_wind_array):
        feature_specific = temporal_segment

        print(f"start preprocess_data with temporal segment size: {temporal_segment}")
        print(f"diff_window: {diff_wind}")
        print(f"con_wind: {con_wind}")
        preprocess_data(n_tasks=n_tasks, job_id=task_id, s_run=s_run, modality=modality,
                        win_size=params.window_size, local_den=params.local_density,
                        diff_win=diff_wind, con_win=con_wind,
                        track_len=temporal_segment, feature_type=feature_calc, specific_feature_calc=feature_specific)
        print(f"finish preprocess_data with temporal segment size: {temporal_segment}")



    # win_size = 35
    # print(f"start preprocess_data with window size: {win_size}")
    # preprocess_data(n_tasks=n_tasks, job_id=task_id, s_run=s_run, modality=modality,
    #                 win_size=win_size, local_den=params.local_density,
    #                 diff_win=params.diff_window, con_win=params.con_window,
    #                 track_len=params.tracks_len, feature_type=feature_calc, specific_feature_calc=win_size)
    # print(f"finish preprocess_data with window size: {win_size}")
    #
    #
    #


