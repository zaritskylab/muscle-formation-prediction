import os, sys
from datetime import datetime

sys.path.append('/sise/home/reutme/muscle-formation-regeneration')
sys.path.append(os.path.abspath('../../TimeSeriesAnalysis'))

from data_layer.utils import *
from model_layer.build_model import build_state_prediction_model_light
from configuration import consts, params

if __name__ == '__main__':
    # Get the modality and feature type from command-line arguments
    modality = sys.argv[1]  # The modality of the data (e.g., actin_intensity, motility)
    feature_type = sys.argv[2]  # The type of feature for sensitivity analysis

    # Iterate over different train-test configurations
    for con_train_n, diff_train_n, con_test_n, diff_test_n in [(1, 5, 2, 3), (2, 3, 1, 5), ]:
        print(f"\n train: con_n-{con_train_n},dif_n-{diff_train_n}; test: con_n-{con_test_n},dif_n-{diff_test_n}")

        today = datetime.datetime.now().strftime('%d-%m-%Y')
        dir_path = f"{consts.storage_path}/{today}-{modality} local dens-False, s{con_train_n}, s{diff_train_n} train" \
                   + (f" win size {consts.WIN_SIZE}" if modality != "motility" else "")

        second_dir = f"track len {consts.SEGMENT_LEN}, impute_func-{consts.IMPUTE_METHOD}_{consts.IMPUTE_FUNC} reg {consts.REG_METHOD}"
        os.makedirs(dir_path, exist_ok=True)
        os.makedirs(os.path.join(dir_path, second_dir), exist_ok=True)
        save_dir_path = dir_path + "/" + second_dir + "/"

        # Build the model using original features (not sensitive analysis)
        if feature_type is None:
            build_state_prediction_model_light(save_dir_path=save_dir_path,
                                                con_window=params.con_window,
                                               diff_window=params.diff_window, modality=modality,
                                               transformed_data_path=consts.transformed_data_path)


        # Build the model by sensitive analysis on temporal_segment feature
        elif feature_type == "temporal_segment_arr":
            # Array of arrays where each array contains temporal segment, differentiation window, and control window arrays
            list_of_tempural_diff_win_con_win = params.feature_calc_types[feature_type]


            temporal_segment_array = list_of_tempural_diff_win_con_win[0][0]
            diff_wind_array = list_of_tempural_diff_win_con_win[0][1]
            con_wind_array = list_of_tempural_diff_win_con_win[0][2]

            # diff_wind - The time windows for the experimental group - the differentiated group
            # con_wind - The time windows for the control group - the group that did not undergo differentiation
            # temporal_segment - The size of each partitioned trajectories - these sub-tracks overlap each other
            for temporal_segment, diff_wind, con_wind in zip(temporal_segment_array, diff_wind_array, con_wind_array):
                print(f"start_merge with feature_type: {feature_type}, temporal_segment: {temporal_segment}")
                print(f"diff_wind: {diff_wind}")
                print(f"con_wind: {con_wind}")

                feature_specific = temporal_segment

                build_state_prediction_model_light(save_dir_path=save_dir_path,
                                                   con_window=params.con_window,
                                                   diff_window=params.diff_window, modality=modality,
                                                   transformed_data_path=consts.transformed_data_path)

        # Build the model by sensitive analysis on diff_win feature
        elif feature_type == "diff_window_arr":

            # Iterate over the diff_win values
            for diff_win in params.feature_calc_types[feature_type]:
                feature_specific = diff_win
                transformed_data_path = consts.storage_path + f"data/mastodon/ts_transformed/%s/{consts.IMPUTE_METHOD}_{consts.IMPUTE_FUNC}/S%s/feature_type_{feature_type}_{feature_specific}/merged_chunks_reg={consts.REG_METHOD},local_den=False,win size={consts.WIN_SIZE}.pkl"

                build_state_prediction_model_light(save_dir_path=save_dir_path,
                                                   con_window=params.con_window,
                                                   diff_window=params.diff_window, modality=modality,
                                                   transformed_data_path=consts.transformed_data_path)

        # Build the model by sensitive analysis on crop window size feature
        elif feature_type == "window_size_arr":

            # Iterate over the size of crop window size
            for win_size in params.feature_calc_types[feature_type]:
                feature_specific = win_size

                transformed_data_path = consts.storage_path + f"data/mastodon/ts_transformed/%s/{consts.IMPUTE_METHOD}_{consts.IMPUTE_FUNC}/S%s/feature_type_{feature_type}_{feature_specific}/merged_chunks_reg={consts.REG_METHOD},local_den=False,win size={consts.WIN_SIZE}.pkl"

                build_state_prediction_model_light(save_dir_path=save_dir_path,
                                                   con_window=params.con_window,
                                                   diff_window=params.diff_window, modality=modality,
                                                   transformed_data_path=consts.transformed_data_path)
