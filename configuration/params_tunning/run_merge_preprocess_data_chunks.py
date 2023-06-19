import os, sys

sys.path.append('/sise/home/reutme/muscle-formation-regeneration')
sys.path.append(os.path.abspath('../..'))
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from data_layer.merge_preprocessed_data_chunks import concat_data_portions


from configuration import consts, params



if __name__ == '__main__':
    print("running merge_prepare_data_chunks")

    # Get arguments from sbatch
    modality = sys.argv[1]  # The modality of the data (e.g., actin_intensity, motility)
    vid_info = consts.vid_info_dict[os.getenv('SLURM_ARRAY_TASK_ID')]  # Video information dictionary
    feature_type = sys.argv[2]  # The type of feature for sensitivity analysis

    # If this is the original model - not sensitivity analysis
    if feature_type == "original":
        feature_specific = 'original'

        files_path = consts.storage_path + f"data/mastodon/ts_transformed/{modality}/{consts.IMPUTE_METHOD}_{consts.IMPUTE_FUNC}/{vid_info['name']}/feature_type_{feature_type}_{feature_specific}"
        data_save_csv_path = files_path + f"merged_chunks_reg={consts.REG_METHOD},local_den=False,win size={consts.WIN_SIZE}.pkl"
        concat_data_portions(files_path, data_save_csv_path)

    # Build model by sensitivity analysis on temporal_segment feature
    elif feature_type == "temporal_segment_arr":
        # Array of arrays where each array contains temporal segment, differentiation window, and control window arrays
        list_of_tempural_diff_win_con_win = params.feature_calc_types[feature_type]
        temporal_segment_array = list_of_tempural_diff_win_con_win[0][0]
        diff_wind_array = list_of_tempural_diff_win_con_win[0][1]
        con_wind_array = list_of_tempural_diff_win_con_win[0][2]

        # Iterate over the arrays of temporal segment, differentiation window, and control window
        for temporal_segment, diff_wind, con_wind in zip(temporal_segment_array, diff_wind_array, con_wind_array):
            print(f"start_merge with feature_type: {feature_type}, temporal_segment: {temporal_segment}")
            print(f"diff_wind: {diff_wind}")
            print(f"con_wind: {con_wind}")

            feature_specific = temporal_segment

            files_path = consts.storage_path + f"data/mastodon/ts_transformed/{modality}/{consts.IMPUTE_METHOD}_{consts.IMPUTE_FUNC}/{vid_info['name']}/feature_type_{feature_type}_{feature_specific}"
            data_save_csv_path = files_path + f"merged_chunks_reg={consts.REG_METHOD},local_den=False,win size={consts.WIN_SIZE}.pkl"

            # Concatenates data portions of transformed tsfresh data from multiple files into a single DataFrame.
            concat_data_portions(files_path, data_save_csv_path)

    # Build model by sensitivity analysis on diff_win feature
    elif feature_type == "diff_window_arr":
        # Iterate over the diff_win values
        for diff_win in params.feature_calc_types[feature_type]:
            feature_specific = diff_win

            files_path = consts.storage_path + f"data/mastodon/ts_transformed/{modality}/{consts.IMPUTE_METHOD}_{consts.IMPUTE_FUNC}/{vid_info['name']}/feature_type_{feature_type}_{feature_specific}"
            data_save_csv_path = files_path + f"merged_chunks_reg={consts.REG_METHOD},local_den=False,win size={consts.WIN_SIZE}.pkl"

            # Concatenates data portions of transformed tsfresh data from multiple files into a single DataFrame.
            concat_data_portions(files_path, data_save_csv_path)

    # Build model by sensitivity analysis on crop window size feature
    elif feature_type == "window_size_arr":

        # Iterate over the win_size values
        for win_size in params.feature_calc_types[feature_type]:
            feature_specific = win_size

            files_path = consts.storage_path + f"data/mastodon/ts_transformed/{modality}/{consts.IMPUTE_METHOD}_{consts.IMPUTE_FUNC}/{vid_info['name']}/feature_type_{feature_type}_{feature_specific}"
            data_save_csv_path = files_path + f"merged_chunks_reg={consts.REG_METHOD},local_den=False,win size={consts.WIN_SIZE}.pkl"

            # Concatenates data portions of transformed tsfresh data from multiple files into a single DataFrame.
            concat_data_portions(files_path, data_save_csv_path)




