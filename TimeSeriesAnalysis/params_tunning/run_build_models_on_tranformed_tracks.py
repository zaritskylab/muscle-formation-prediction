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

    # for win_size in params.feature_calc_types[feature_type]:
    #     build_model_trans_tracks(path=consts.storage_path, local_density=params.local_density, window_size=win_size,
    #                              tracks_len=params.tracks_len, con_window=params.con_window,
    #                              diff_window=params.diff_window, feature_type=feature_type,
    #                              specific_feature_type=win_size)

    win_size=16
    build_model_trans_tracks(path=consts.storage_path, local_density=params.local_density, window_size=win_size,
                             tracks_len=params.tracks_len, con_window=params.con_window,
                             diff_window=params.diff_window, feature_type=feature_type,
                             specific_feature_type=win_size, modality=modality)
