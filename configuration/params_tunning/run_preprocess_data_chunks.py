import os
import sys

sys.path.append('/sise/home/reutme/muscle-formation-regeneration')
sys.path.append(os.path.abspath('../..'))
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from data_preparation import prepare_data_in_parallel_chunks
from data_layer.utils import *
from configuration import consts, params

if __name__ == '__main__':
    print("running prepare_data")

    # get arguments from sbatch
    task_id = int(os.getenv('SLURM_ARRAY_TASK_ID')[1:])
    modality = sys.argv[1]  # The modality of the data (e.g., actin_intensity, motility)
    n_tasks = int(sys.argv[2])  # The number of tasks
    vid_info = consts.vid_info_dict[sys.argv[3]]  # Video information dictionary

    # The feature type is the name of the feature that will be changed
    feature_type = sys.argv[4]

    # Define base directory path and video path based on modality and video information
    base_dir_path = consts.storage_path + f"data/mastodon/ts_transformed/{modality}/{consts.IMPUTE_METHOD}_{consts.IMPUTE_FUNC}/{vid_info['name']} "
    vid_path = vid_info["actin_path"] if modality == "actin_intensity" else vid_info["nuc_path"]

    # set paths for saving transformed data
    save_data_path = f"/{vid_info['name']}_reg={consts.REG_METHOD},local_den=False" + \
                     (f"win size {consts.WIN_SIZE}" if modality != "motility" else "")

    # Load tracks data from CSV file
    print("\n===== load data =====")
    tracks_csv_path = consts.data_csv_path % (consts.REG_METHOD, vid_info['name'])
    tracks_df, _ = get_tracks(tracks_csv_path, tagged_only=True)

    # If this is the original model - not sensitivity analysis
    if feature_type is None:
        save_data_dir_path = base_dir_path
        os.makedirs(save_data_dir_path, exist_ok=True)

        # Build model using the original feature
        prepare_data_in_parallel_chunks(tracks_df=tracks_df, vid_path=vid_path, modality=modality,
                                        target=vid_info["target"], n_tasks=n_tasks,
                                        job_id=task_id, win_size=consts.WIN_SIZE,
                                        segment_length=consts.SEGMENT_LEN,
                                        save_data_dir_path=save_data_dir_path,
                                        save_data_path=save_data_dir_path + save_data_path)

    # Build model by sensitivity analysis on crop window size feature
    elif feature_type == "window_size_arr":
        save_data_dir_path = base_dir_path + f"/{feature_type}/"
        for win_size in params.feature_calc_types[feature_type]:
            print(f"start preprocess_data with window size: {win_size}")
            prepare_data_in_parallel_chunks(tracks_df=tracks_df, vid_path=vid_path, modality=modality,
                                            target=vid_info["target"], n_tasks=n_tasks,
                                            job_id=task_id, win_size=consts.WIN_SIZE,
                                            segment_length=consts.SEGMENT_LEN,
                                            save_data_dir_path=save_data_dir_path,
                                            save_data_path=save_data_dir_path + save_data_path)
            print(f"finish preprocess_data with window size: {win_size}")

    # Build model by sensitivity analysis on temporal_segment feature
    elif feature_type == "temporal_segment_arr":
        # array of arrays that each array is one of temporal segment array, differentiation_window array and control
        # window array
        list_of_tempural_diff_win_con_win = params.feature_calc_types[feature_type]
        temporal_segment_array = list_of_tempural_diff_win_con_win[0][0]
        diff_wind_array = list_of_tempural_diff_win_con_win[0][1]
        con_wind_array = list_of_tempural_diff_win_con_win[0][2]

        for temporal_segment, diff_wind, con_wind in zip(temporal_segment_array, diff_wind_array, con_wind_array):
            save_data_dir_path = base_dir_path + f"/{feature_type}/{temporal_segment}"

            print(f"start preprocess_data with temporal segment size: {temporal_segment}")
            print(f"diff_window: {diff_wind}")
            print(f"con_wind: {con_wind}")
            prepare_data_in_parallel_chunks(tracks_df=tracks_df, vid_path=vid_path, modality=modality,
                                            target=vid_info["target"], n_tasks=n_tasks,
                                            job_id=task_id, win_size=consts.WIN_SIZE,
                                            segment_length=consts.SEGMENT_LEN,
                                            save_data_dir_path=save_data_dir_path,
                                            save_data_path=save_data_dir_path + save_data_path)
            print(f"finish preprocess_data with temporal segment size: {temporal_segment}")

    # Build model by sensitivity analysis on diff_win feature
    elif feature_type == "diff_window_arr":
        for diff_win in params.feature_calc_types[feature_type]:
            save_data_dir_path = base_dir_path + f"/{feature_type}/{diff_win}"
            prepare_data_in_parallel_chunks(tracks_df=tracks_df, vid_path=vid_path, modality=modality,
                                            target=vid_info["target"], n_tasks=n_tasks,
                                            job_id=task_id, win_size=consts.WIN_SIZE,
                                            segment_length=consts.SEGMENT_LEN,
                                            save_data_dir_path=save_data_dir_path,
                                            save_data_path=save_data_dir_path + save_data_path)
