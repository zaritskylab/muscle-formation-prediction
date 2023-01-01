import glob
import pickle
import os, sys

sys.path.append('/sise/home/reutme/muscle-formation-regeneration')
sys.path.append(os.path.abspath('../..'))
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from TimeSeriesAnalysis.data_preprocessing.merge_preprocessed_data_chunks import concat_data_portions
import TimeSeriesAnalysis.params as params
import TimeSeriesAnalysis.consts as consts



if __name__ == '__main__':
    print("running merge_prepare_data_chunks")

    # get arguments from sbatch
    modality = sys.argv[1]

    s_run = consts.s_runs[os.getenv('SLURM_ARRAY_TASK_ID')]

    feature_type = sys.argv[2]

    list_of_tempural_diff_win_con_win = params.feature_calc_types[feature_type]
    temporal_segment_array = list_of_tempural_diff_win_con_win[0][0]
    diff_wind_array = list_of_tempural_diff_win_con_win[0][1]
    con_wind_array = list_of_tempural_diff_win_con_win[0][2]

    for temporal_segment, diff_wind, con_wind in zip(temporal_segment_array, diff_wind_array, con_wind_array):
        print(f"start_merge with feature_type: {feature_type}, temporal_segment: {temporal_segment}")
        print(f"diff_wind: {diff_wind}")
        print(f"con_wind: {con_wind}")

        feature_specific = temporal_segment


        concat_data_portions(params.local_density, params.window_size, s_run, modality, temporal_segment, feature_type)






