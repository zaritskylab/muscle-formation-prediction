from collections import Counter
import os, sys

sys.path.append('/sise/home/reutme/muscle-formation-regeneration')
sys.path.append(os.path.abspath('..'))

import TimeSeriesAnalysis.consts as consts
from TimeSeriesAnalysis.utils.data_load_save import *
from TimeSeriesAnalysis.build_models_on_transformed_tracks import build_model_trans_tracks

if __name__ == '__main__':
    modality = sys.argv[1]
    feature_type = sys.argv[2]

# build model by sensitive analysis on crop window size feature
    # for win_size in params.feature_calc_types[feature_type]:
    #     build_model_trans_tracks(path=consts.storage_path, local_density=params.local_density,
    #                              window_size=params.window_size,
    #                              tracks_len=params.tracks_len, con_window=params.con_window,
    #                              diff_window=params.diff_window, feature_type=feature_type,
    #                              specific_feature_type=win_size, modality=modality)


# # build model by sensitive analysis on temporal_segment feature
#     #array of arrays that each array is one of temporal segment array, differentiation_window array and control window array
#     list_of_tempural_diff_win_con_win = params.feature_calc_types[feature_type]
#
#     temporal_segment_array = list_of_tempural_diff_win_con_win[0][0]
#     diff_wind_array = list_of_tempural_diff_win_con_win[0][1]
#     con_wind_array = list_of_tempural_diff_win_con_win[0][2]
#
#     for temporal_segment, diff_wind, con_wind in zip(temporal_segment_array, diff_wind_array, con_wind_array):
#         print(f"start_merge with feature_type: {feature_type}, temporal_segment: {temporal_segment}")
#         print(f"diff_wind: {diff_wind}")
#         print(f"con_wind: {con_wind}")
#
#         feature_specific = temporal_segment
#
#         build_model_trans_tracks(path=consts.storage_path, local_density=params.local_density,
#                                  window_size=params.window_size,
#                                  tracks_len=temporal_segment, con_window=con_wind,
#                                  diff_window=diff_wind, feature_type=feature_type,
#                                  specific_feature_type=feature_specific, modality=modality)


# build model by sensitive analysis on diff_win feature
    for diff_win in params.feature_calc_types[feature_type]:

        feature_specific = diff_win

        build_model_trans_tracks(path=consts.storage_path, local_density=params.local_density,
                                 window_size=params.window_size,
                                 tracks_len=params.tracks_len, con_window=params.con_window,
                                 diff_window=diff_win, feature_type=feature_type,
                                 specific_feature_type=feature_specific, modality=modality)