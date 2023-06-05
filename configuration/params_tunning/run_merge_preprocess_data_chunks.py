import os, sys

sys.path.append('/sise/home/reutme/muscle-formation-regeneration')
sys.path.append(os.path.abspath('../..'))
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from data_layer.data_preprocessing import concat_data_portions
import configuration.params as params
import configuration.consts as consts



if __name__ == '__main__':
    print("running merge_prepare_data_chunks")

    # get arguments from sbatch
    modality = sys.argv[1]

    s_run = consts.vid_info_dict[os.getenv('SLURM_ARRAY_TASK_ID')]

    feature_type = sys.argv[2]

    # if this is the original model - not sensitive analysis
    if feature_type == "original":
        feature_specific = 'original'
        concat_data_portions(params.window_size, s_run, modality, feature_specific, feature_type)

    # build model by sensitive analysis on temporal_segment feature
    elif feature_type == "temporal_segment_arr":
        #array of arrays that each array is one of temporal segment array, differentiation_window array and control window array
        list_of_tempural_diff_win_con_win = params.feature_calc_types[feature_type]
        temporal_segment_array = list_of_tempural_diff_win_con_win[0][0]
        diff_wind_array = list_of_tempural_diff_win_con_win[0][1]
        con_wind_array = list_of_tempural_diff_win_con_win[0][2]

        for temporal_segment, diff_wind, con_wind in zip(temporal_segment_array, diff_wind_array, con_wind_array):
            print(f"start_merge with feature_type: {feature_type}, temporal_segment: {temporal_segment}")
            print(f"diff_wind: {diff_wind}")
            print(f"con_wind: {con_wind}")

            feature_specific = temporal_segment

            concat_data_portions(params.window_size, s_run, modality, feature_specific, feature_type)

    # build model by sensitive analysis on diff_win feature
    elif feature_type == "diff_window_arr":

        for diff_win in params.feature_calc_types[feature_type]:
            feature_specific = diff_win

            concat_data_portions(params.window_size, s_run, modality, feature_specific, feature_type)

    # build model by sensitive analysis on crop window size feature
    elif feature_type == "window_size_arr":

        for win_size in params.feature_calc_types[feature_type]:
            feature_specific = win_size
            concat_data_portions(params.window_size, s_run, modality, feature_specific, feature_type)




